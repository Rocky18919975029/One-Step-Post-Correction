import argparse
from pathlib import Path

import pandas as pd


def read_csv_if_exists(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def short_text(text, max_chars):
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--completions_per_prefix", type=int, default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=500)
    parser.add_argument("--show_examples", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.csv"
    samples_path = output_dir / "samples.csv"
    eval_path = output_dir / "eval_metrics.csv"
    checkpoint_state_path = output_dir / "checkpoint_latest" / "training_state.pt"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")

    metrics = pd.read_csv(metrics_path)
    samples = pd.read_csv(samples_path)
    eval_metrics = read_csv_if_exists(eval_path)

    print("== Artifacts ==")
    block_dirs = sorted(path.name for path in output_dir.glob("block_*") if path.is_dir())
    print(f"output_dir: {output_dir}")
    print(f"metrics.csv: {metrics_path.exists()}")
    print(f"samples.csv: {samples_path.exists()}")
    print(f"eval_metrics.csv: {eval_path.exists()}")
    print(f"checkpoint_latest: {(output_dir / 'checkpoint_latest').exists()}")
    print(f"final: {(output_dir / 'final').exists()}")
    print(f"block checkpoints: {', '.join(block_dirs) if block_dirs else 'none'}")
    if checkpoint_state_path.exists():
        try:
            import torch

            state = torch.load(checkpoint_state_path, map_location="cpu", weights_only=False)["state"]
            print(
                "checkpoint state: "
                f"next_block_idx={state.get('next_block_idx')} "
                f"global_step={state.get('global_step')} "
                f"wandb_id={state.get('wandb_id')}"
            )
            saved_args = state.get("args", {})
            if args.completions_per_prefix is None and "completions_per_prefix" in saved_args:
                args.completions_per_prefix = int(saved_args["completions_per_prefix"])
            if args.max_completion_tokens is None and saved_args.get("max_completion_tokens") is not None:
                args.max_completion_tokens = int(saved_args["max_completion_tokens"])
        except Exception as exc:
            print(f"checkpoint state: unreadable ({exc})")

    print("== Metrics by block ==")
    print(
        metrics.groupby("block_idx")
        .agg(
            steps=("step", "count"),
            loss_mean=("loss", "mean"),
            reward_mean=("reward_mean", "mean"),
            logp_theta_mean=("logp_theta_mean", "mean"),
            logp_ref_mean=("logp_ref_mean", "mean"),
        )
        .reset_index()
        .to_string(index=False)
    )

    if eval_metrics is not None:
        print("\n== Eval by block ==")
        print(eval_metrics.to_string(index=False))

    print("\n== Samples by block ==")
    summary_columns = {
        "samples": ("step", "count"),
        "reward_mean": ("reward", "mean"),
        "boxed_rate": ("has_boxed_answer", "mean"),
        "prefix_tokens_mean": ("prefix_token_len", "mean"),
        "completion_tokens_mean": ("completion_token_len", "mean"),
        "logp_theta_mean": ("logp_theta", "mean"),
        "logp_ref_mean": ("logp_ref", "mean"),
    }
    print(
        samples.groupby("block_idx")
        .agg(**summary_columns)
        .reset_index()
        .to_string(index=False)
    )

    print("\n== Flow checks ==")
    grouping = ["block_idx", "rank", "step", "example_idx"]
    counts = samples.groupby(grouping).size().reset_index(name="samples_per_prefix")
    expected = args.completions_per_prefix
    if expected is None:
        expected = int(counts["samples_per_prefix"].mode().iloc[0])
        print(f"inferred completions_per_prefix: {expected}")
    bad_counts = counts[counts["samples_per_prefix"] != expected]
    print(f"sample count per prefix: {'ok' if bad_counts.empty else 'mismatch'}")
    if not bad_counts.empty:
        print(bad_counts.head(20).to_string(index=False))

    if args.max_completion_tokens is not None:
        too_long = samples[samples["completion_token_len"] > args.max_completion_tokens]
        print(f"completion length <= {args.max_completion_tokens}: {'ok' if too_long.empty else 'mismatch'}")
        if not too_long.empty:
            print(
                too_long[
                    ["block_idx", "rank", "step", "example_idx", "sample_idx", "completion_token_len"]
                ]
                .head(20)
                .to_string(index=False)
            )

    no_box = samples[~samples["has_boxed_answer"].astype(bool)]
    print(f"boxed answer parse rate: {1.0 - len(no_box) / max(len(samples), 1):.3f}")
    print(f"reward mean: {samples['reward'].mean():.3f}")

    scored = samples.copy()
    scored["log_z_term"] = args.alpha * scored["logp_ref"] - scored["logp_theta"] + scored["reward"] / args.beta
    scored["target_logp"] = args.alpha * scored["logp_ref"] + scored["reward"] / args.beta
    scored["log_z_hat"] = scored.groupby(grouping)["log_z_term"].transform("mean")
    scored["tb_residual"] = scored["log_z_hat"] + scored["logp_theta"] - scored["target_logp"]
    scored["tb_loss_term"] = scored["tb_residual"] ** 2
    recomputed = (
        scored.groupby(["block_idx", "rank", "step"])["tb_loss_term"]
        .mean()
        .reset_index(name="loss_from_samples")
    )
    loss_compare = metrics.merge(recomputed, on=["block_idx", "rank", "step"], how="left")
    loss_compare["loss_abs_diff"] = (loss_compare["loss"] - loss_compare["loss_from_samples"]).abs()
    print("\n== Loss Consistency ==")
    print(
        loss_compare[
            ["block_idx", "rank", "step", "loss", "loss_from_samples", "loss_abs_diff"]
        ]
        .to_string(index=False)
    )
    print(f"max loss abs diff: {loss_compare['loss_abs_diff'].max():.6g}")

    print("\n== Example completions ==")
    for block_idx in sorted(samples["block_idx"].unique()):
        block_samples = samples[samples["block_idx"] == block_idx].head(args.show_examples)
        for _, row in block_samples.iterrows():
            print(
                f"\n[block={row['block_idx']} example={row['example_idx']} sample={row['sample_idx']} "
                f"reward={row['reward']} parsed={row['parsed_answer']}]"
            )
            print(f"prefix_tokens={row.get('prefix_token_len', 'n/a')} completion_tokens={row.get('completion_token_len', 'n/a')}")
            print("prefix:", short_text(row["prefix_text"], args.max_chars))
            print("completion:", short_text(row["completion"], args.max_chars))


if __name__ == "__main__":
    main()
