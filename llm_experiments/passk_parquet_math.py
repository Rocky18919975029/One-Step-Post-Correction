import argparse
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grader_utils.math_grader import grade_answer


_LAST_NUM_RE = re.compile(r"_(\d+)(?=\.[^.]+$)")


def safe_grade_math(ans, correct_ans):
    try:
        return int(grade_answer(str(ans), str(correct_ans)))
    except Exception:
        return 0


def seed_from_fname(fname):
    match = _LAST_NUM_RE.search(fname)
    if not match:
        return None
    return int(match.group(1))


def load_correct_matrix(fnames):
    rows = []
    for fname in fnames:
        seed = seed_from_fname(fname)
        if seed is None:
            continue
        df = pd.read_csv(fname)
        for _, row in df.iterrows():
            rows.append({
                "seed": seed,
                "problem_id": int(row["problem_id"]),
                "correct": safe_grade_math(row["mcmc_answer"], row["correct_answer"]),
            })

    if not rows:
        return pd.DataFrame()

    scored = pd.DataFrame(rows)
    return scored.pivot_table(
        index="problem_id",
        columns="seed",
        values="correct",
        aggfunc="max",
    )


def plot_passk(fnames, output_path=None):
    correct_by_problem = load_correct_matrix(fnames)
    if correct_by_problem.empty:
        print("No scored files found.")
        return []

    seeds = sorted(correct_by_problem.columns.tolist())
    best_of_n_acc = []

    for n in range(1, len(seeds) + 1):
        accs = []
        subsets = itertools.combinations(seeds, n)
        subsets = list(subsets) if len(seeds) <= 5 else None

        if subsets is None:
            rng = np.random.default_rng(0)
            sampled_subsets = [
                tuple(rng.choice(seeds, size=n, replace=False))
                for _ in range(200)
            ]
        else:
            sampled_subsets = subsets

        for subset in sampled_subsets:
            subset_df = correct_by_problem.loc[:, list(subset)]
            complete_rows = subset_df.notna().all(axis=1)
            if complete_rows.sum() == 0:
                continue
            subset_correct = subset_df.loc[complete_rows].max(axis=1)
            accs.append(subset_correct.mean())

        best_of_n_acc.append((n, float(np.mean(accs)) if accs else 0.0))

    for n, mean_acc in best_of_n_acc:
        print(f"Best-of-{n}: {mean_acc:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(
        [n for n, _ in best_of_n_acc],
        [mean for _, mean in best_of_n_acc],
        "o-",
    )
    plt.xlabel("k")
    plt.ylabel("Pass@k Accuracy")
    plt.title("Parquet Math Train")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200)
    else:
        plt.show()

    return best_of_n_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    plot_passk(fnames, args.output_path)
