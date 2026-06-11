#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def add_rolling_columns(df, window):
    df = df.copy()
    for column in ["loss", "reward_mean", "logp_theta_mean", "logp_ref_mean", "logp_gap"]:
        if column in df.columns:
            df[f"{column}_roll"] = (
                df.groupby("block_idx", sort=False)[column]
                .transform(lambda values: values.rolling(window, min_periods=1).mean())
            )
    return df


def plot_lines(df, output_dir, window):
    plt = require_matplotlib()
    plots = [
        ("loss", "Training loss", "loss_by_step.png"),
        ("reward_mean", "Reward mean", "reward_by_step.png"),
        ("logp_theta_mean", "Policy logprob mean", "logp_theta_by_step.png"),
        ("logp_ref_mean", "Reference logprob mean", "logp_ref_by_step.png"),
        ("logp_gap", "Policy minus reference logprob", "logp_gap_by_step.png"),
    ]

    for column, title, filename in plots:
        if column not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(13, 5))
        for block_idx, block_df in df.groupby("block_idx", sort=False):
            ax.plot(block_df["step"], block_df[column], alpha=0.22, linewidth=0.9, label=f"block {block_idx} raw")
            roll_col = f"{column}_roll"
            if roll_col in block_df.columns:
                ax.plot(block_df["step"], block_df[roll_col], linewidth=2.0, label=f"block {block_idx} roll{window}")
        ax.set_title(title)
        ax.set_xlabel("global optimizer step")
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def plot_combined(df, output_dir, window):
    plt = require_matplotlib()
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    columns = [
        ("loss", "loss"),
        ("reward_mean", "reward_mean"),
        ("logp_theta_mean", "logp_theta_mean"),
        ("logp_gap", "logp_theta_mean - logp_ref_mean"),
    ]
    for ax, (column, ylabel) in zip(axes, columns):
        for block_idx, block_df in df.groupby("block_idx", sort=False):
            roll_col = f"{column}_roll"
            y = block_df[roll_col] if roll_col in block_df.columns else block_df[column]
            ax.plot(block_df["step"], y, linewidth=1.8, label=f"block {block_idx}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("global optimizer step")
    fig.suptitle(f"Training diagnostics, rolling window={window}", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "training_diagnostics_overview.png", dpi=180)
    plt.close(fig)


def plot_histograms(df, output_dir):
    plt = require_matplotlib()
    columns = ["loss", "reward_mean", "logp_theta_mean", "logp_ref_mean", "logp_gap"]
    for column in columns:
        if column not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for block_idx, block_df in df.groupby("block_idx", sort=False):
            ax.hist(block_df[column].dropna(), bins=40, alpha=0.45, label=f"block {block_idx}")
        ax.set_title(f"{column} distribution by block")
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"{column}_hist_by_block.png", dpi=180)
        plt.close(fig)


def plot_scatter(df, output_dir):
    plt = require_matplotlib()
    if "logp_gap" not in df.columns or "loss" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for block_idx, block_df in df.groupby("block_idx", sort=False):
        ax.scatter(block_df["logp_gap"], block_df["loss"], s=14, alpha=0.55, label=f"block {block_idx}")
    ax.set_title("Loss versus policy-reference logprob gap")
    ax.set_xlabel("logp_theta_mean - logp_ref_mean")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "loss_vs_logp_gap.png", dpi=180)
    plt.close(fig)


def write_summaries(df, output_dir):
    numeric_columns = [
        column
        for column in ["loss", "reward_mean", "logp_theta_mean", "logp_ref_mean", "logp_gap"]
        if column in df.columns
    ]
    summary = (
        df.groupby("block_idx", sort=False)[numeric_columns]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(6)
    )
    summary.to_csv(output_dir / "summary_by_block.csv")

    tail_summary = (
        df.groupby("block_idx", sort=False)
        .tail(min(20, max(1, len(df))))
        .groupby("block_idx", sort=False)[numeric_columns]
        .mean()
        .round(6)
    )
    tail_summary.to_csv(output_dir / "last20_mean_by_block.csv")

    with (output_dir / "diagnostic_notes.txt").open("w") as handle:
        handle.write("Diagnostics to inspect:\n")
        handle.write("1. A strongly negative logp_gap means the trained policy assigns lower probability than ref to sampled completions.\n")
        handle.write("2. A block whose reward_mean is normal but eval collapses likely has an update-scale or target-scale issue.\n")
        handle.write("3. Compare summary_by_block.csv with last20_mean_by_block.csv to see late-block drift.\n")


def main():
    parser = argparse.ArgumentParser(description="Plot diagnostics from an OSPC metrics.csv file.")
    parser.add_argument(
        "metrics",
        type=str,
        help="Path to metrics.csv, or a result directory containing metrics.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for plots. Defaults to <result_dir>/diagnostics/metrics.",
    )
    parser.add_argument("--rolling_window", type=int, default=10)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / "diagnostics" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)
    required = {"step", "block_idx", "loss", "reward_mean", "logp_theta_mean", "logp_ref_mean"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"metrics file is missing columns: {missing}")

    df = df.sort_values(["step"]).reset_index(drop=True)
    df["logp_gap"] = df["logp_theta_mean"] - df["logp_ref_mean"]
    df = add_rolling_columns(df, max(1, args.rolling_window))

    plot_lines(df, output_dir, max(1, args.rolling_window))
    plot_combined(df, output_dir, max(1, args.rolling_window))
    plot_histograms(df, output_dir)
    plot_scatter(df, output_dir)
    write_summaries(df, output_dir)

    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
