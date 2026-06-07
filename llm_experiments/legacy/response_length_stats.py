import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
import transformers

from blockwise_power_tb_train import MODEL_NAME_BY_KEY
from eval_math import collect_fnames


def summarize_lengths(frame, label, token_col, char_col, max_tokens):
    lengths = frame[token_col].dropna()
    chars = frame[char_col].dropna()
    if lengths.empty:
        return {
            "response_type": label,
            "n": 0,
        }

    summary = {
        "response_type": label,
        "n": int(lengths.shape[0]),
        "token_mean": float(lengths.mean()),
        "token_std": float(lengths.std(ddof=0)),
        "token_min": int(lengths.min()),
        "token_p25": float(lengths.quantile(0.25)),
        "token_p50": float(lengths.quantile(0.50)),
        "token_p75": float(lengths.quantile(0.75)),
        "token_p90": float(lengths.quantile(0.90)),
        "token_p95": float(lengths.quantile(0.95)),
        "token_p99": float(lengths.quantile(0.99)),
        "token_max": int(lengths.max()),
        "char_mean": float(chars.mean()) if not chars.empty else 0.0,
    }
    if max_tokens is not None:
        summary["hit_max_count"] = int((lengths >= max_tokens).sum())
        summary["hit_max_rate"] = float((lengths >= max_tokens).mean())
    return summary


def add_lengths(df, tokenizer, columns):
    for column in columns:
        df[f"{column}_chars"] = df[column].fillna("").astype(str).str.len()
        df[f"{column}_tokens"] = [
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in df[column].fillna("").astype(str)
        ]
    return df


def read_result_files(path, mode):
    path = Path(path)
    if mode == "math":
        files = collect_fnames(path)
    elif path.is_file():
        files = [str(path)]
    else:
        files = sorted(str(p) for p in path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {path}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Summarize completion length distributions.")
    parser.add_argument("path", type=str)
    parser.add_argument("--mode", choices=["math", "blockwise"], default="math")
    parser.add_argument("--model", choices=sorted(MODEL_NAME_BY_KEY), default="qwen")
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--save_with_lengths", type=str, default=None)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME_BY_KEY[args.model],
        trust_remote_code=True,
    )

    files = read_result_files(args.path, args.mode)
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    print(f"Loaded {len(files)} files, {len(df)} rows")

    if args.mode == "math":
        columns = ["std_completion", "naive_completion", "mcmc_completion"]
    else:
        columns = ["completion"]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing completion columns: {missing}")

    df = add_lengths(df, tokenizer, columns)
    summaries = [
        summarize_lengths(
            df,
            column.replace("_completion", ""),
            f"{column}_tokens",
            f"{column}_chars",
            args.max_tokens,
        )
        for column in columns
    ]
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))

    if args.output_csv:
        summary_df.to_csv(args.output_csv, index=False)
    if args.save_with_lengths:
        df.to_csv(args.save_with_lengths, index=False)


if __name__ == "__main__":
    main()
