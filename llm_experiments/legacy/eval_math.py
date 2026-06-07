import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
from grader_utils.math_grader import grade_answer


def safe_grade(ans, correct_ans):
    try:
        return int(grade_answer(str(ans), str(correct_ans)))
    except Exception:
        return 0


def eval_math(fname):
    print(fname)
    df = pd.read_csv(fname)
    base_correct = 0
    temp_correct = 0
    mcmc_correct = 0
    total = len(df)

    for i in range(total):
        base_correct += safe_grade(df["std_answer"][i], df["correct_answer"][i])
        temp_correct += safe_grade(df["naive_answer"][i], df["correct_answer"][i])
        mcmc_correct += safe_grade(df["mcmc_answer"][i], df["correct_answer"][i])


    return base_correct, temp_correct, mcmc_correct, total


def math_results(fnames):
    base_total = 0
    temp_total = 0
    mcmc_total = 0
    total = 0

    for fname in fnames:
        base, temp, mcmc, n = eval_math(fname)
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


def collect_fnames(path):
    path = Path(path)
    if path.is_file():
        return [str(path)]
    return sorted(
        str(p)
        for p in path.glob("*_math_base_power_samp_results_*.csv")
        if "merged" not in p.name
    )


def verify_math_files(fnames):
    if not fnames:
        raise FileNotFoundError("No MATH shard CSV files found.")

    total = 0
    for fname in fnames:
        df = pd.read_csv(fname)
        total += len(df)
        required = {
            "question",
            "correct_answer",
            "naive_answer",
            "std_answer",
            "mcmc_answer",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{fname} is missing columns: {sorted(missing)}")
        print(f"Verified {fname}: {len(df)} rows")

    print(f"Verified total rows: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()

    fnames = collect_fnames(args.path)
    verify_math_files(fnames)
    if args.verify_only:
        raise SystemExit(0)
    math_results(fnames)
