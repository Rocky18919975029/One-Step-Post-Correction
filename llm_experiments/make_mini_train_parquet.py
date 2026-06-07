import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sample a deterministic mini parquet from a larger training parquet.")
    parser.add_argument("--input", type=str, default="../data/train.parquet")
    parser.add_argument("--output", type=str, default="../data/mini_train.parquet")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    frame = pd.read_parquet(input_path)
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.count > len(frame):
        raise ValueError(f"Requested {args.count} rows, but only {len(frame)} are available in {input_path}")

    sampled = frame.sample(n=args.count, random_state=args.seed, replace=False).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(output_path, index=False)

    print(f"Saved {len(sampled)} rows to {output_path}", flush=True)
    print(f"Source rows: {len(frame)} | seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
