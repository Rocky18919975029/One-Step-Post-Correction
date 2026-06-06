import argparse
from pathlib import Path

import pandas as pd


def short_text(text, max_chars):
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--max_chars", type=int, default=500)
    parser.add_argument("--show_examples", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.csv"
    samples_path = output_dir / "samples.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")

    metrics = pd.read_csv(metrics_path)
    samples = pd.read_csv(samples_path)

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
