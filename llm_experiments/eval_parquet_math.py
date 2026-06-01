import argparse
from pathlib import Path

import pandas as pd

from grader_utils.math_grader import grade_answer


def safe_grade(ans, correct_ans):
    try:
        return int(grade_answer(str(ans), str(correct_ans)))
    except Exception:
        return 0


def eval_parquet_math(fname):
    df = pd.read_csv(fname)
    base_correct = 0
    temp_correct = 0
    mcmc_correct = 0
    total = len(df)

    for i in range(total):
        correct_answer = df["correct_answer"][i]
        base_correct += safe_grade(df["std_answer"][i], correct_answer)
        temp_correct += safe_grade(df["naive_answer"][i], correct_answer)
        mcmc_correct += safe_grade(df["mcmc_answer"][i], correct_answer)

    return base_correct, temp_correct, mcmc_correct, total


def parquet_math_results(fnames):
    base_total = 0
    temp_total = 0
    mcmc_total = 0
    total = 0

    for fname in fnames:
        base, temp, mcmc, n = eval_parquet_math(fname)
        base_total += base
        temp_total += temp
        mcmc_total += mcmc
        total += n

    denom = max(total, 1)
    base_acc = base_total / denom
    temp_acc = temp_total / denom
    mcmc_acc = mcmc_total / denom

    print(f"Files evaluated: {len(fnames)}")
    print(f"Total questions: {total}")
    print(f"Base accuracy:  {base_acc:.3f}")
    print(f"Temp accuracy:  {temp_acc:.3f}")
    print(f"MCMC accuracy:  {mcmc_acc:.3f}")

    return {
        "base_acc": base_acc,
        "temp_acc": temp_acc,
        "mcmc_acc": mcmc_acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    folder = Path(args.folder)
    fnames = sorted(str(p) for p in folder.glob("*.csv"))
    parquet_math_results(fnames)
