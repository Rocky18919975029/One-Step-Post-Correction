import argparse
import platform
import tempfile
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/train.parquet")
    parser.add_argument("--passk_plot", type=str, default=None)
    args = parser.parse_args()

    print(f"Python: {platform.python_version()}")

    import matplotlib
    import numpy as np
    import pyarrow
    import pylatexenc
    import sympy
    import torch
    import transformers

    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"pyarrow: {pyarrow.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"sympy: {sympy.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"pylatexenc: {pylatexenc.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")

    from eval_parquet_math import parquet_math_results
    from grader_utils.math_grader import grade_answer
    from grader_utils.parse_utils import parse_answer
    from passk_parquet_math import plot_passk
    from power_samp_utils import AutoregressiveSampler, mcmc_power_samp

    assert AutoregressiveSampler is not None
    assert mcmc_power_samp is not None
    assert parse_answer(r"The answer is \boxed{42}.") == "42"
    assert grade_answer("42", "42")
    print("Project imports and native math grader: ok")

    data_path = Path(args.data_path)
    df = pd.read_parquet(data_path)
    required_columns = {"question", "answer"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required parquet columns: {sorted(missing)}")
    print(f"Parquet read: ok ({data_path}, rows={len(df)}, columns={len(df.columns)})")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for seed in [0, 1]:
            pd.DataFrame([
                {
                    "problem_id": 0,
                    "question": "q0",
                    "correct_answer": "1",
                    "std_answer": "1",
                    "naive_answer": "2",
                    "mcmc_answer": "1",
                },
                {
                    "problem_id": 1,
                    "question": "q1",
                    "correct_answer": "2",
                    "std_answer": "0",
                    "naive_answer": "2",
                    "mcmc_answer": "2" if seed == 1 else "0",
                },
            ]).to_csv(
                tmp_path / f"qwen_math_parquet_math_power_samp_results_10_0.25_0_{seed}.csv",
                index=False,
            )

        parquet_math_results(sorted(str(p) for p in tmp_path.glob("*.csv")))
        plot_path = args.passk_plot or str(tmp_path / "passk.png")
        plot_passk(sorted(str(p) for p in tmp_path.glob("*.csv")), plot_path)
        print(f"Pass@k smoke plot: {plot_path}")

    print("Environment smoke test: ok")


if __name__ == "__main__":
    main()
