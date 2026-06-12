import argparse
from pathlib import Path

import pandas as pd
import transformers

from blockwise_power_tb_train import resolve_model_name


SCORE_PREFIXES = (
    "logp_",
    "token_logp_",
    "token_tb_",
    "tb_target",
    "ref_policy",
)


def is_score_column(column):
    return column == "log_z_hat" or column == "score_version" or any(column.startswith(prefix) for prefix in SCORE_PREFIXES)


def convert_buffer(args):
    src = Path(args.input_buffer)
    dst = Path(args.output_buffer)
    if not src.exists():
        raise FileNotFoundError(src)

    model_name = resolve_model_name(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    df = pd.read_csv(src)
    records = []
    stage_start = min(max((args.block_idx - 1) * args.block_size, 0), args.max_completion_tokens)
    stage_limit = min(args.block_idx * args.block_size, args.max_completion_tokens)

    for row in df.to_dict("records"):
        partial_completion = str(row["completion"])
        partial_ids = tokenizer.encode(partial_completion, add_special_tokens=False)
        stage_end = min(stage_limit, len(partial_ids))
        prefix_piece_ids = partial_ids[:stage_start]
        completion_ids = partial_ids[stage_start:stage_end]

        if args.drop_empty and not completion_ids:
            continue

        prefix_piece = tokenizer.decode(prefix_piece_ids, skip_special_tokens=False) if prefix_piece_ids else ""
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False) if completion_ids else ""
        prompt_text = str(row["prefix_text"])
        prefix_text = prompt_text + prefix_piece

        clean_row = {key: value for key, value in row.items() if not is_score_column(str(key))}
        clean_row.update(
            {
                "block_idx": args.block_idx,
                "prefix_text": prefix_text,
                "prefix_token_len": len(tokenizer.encode(prefix_text)),
                "completion": completion,
                "completion_token_len": len(completion_ids),
                "stage_start_token": stage_start,
                "stage_end_token": stage_start + len(completion_ids),
                "full_partial_completion": partial_completion,
                "full_partial_completion_token_len": len(partial_ids),
            }
        )
        records.append(clean_row)

    if not records:
        raise ValueError("No rows left after conversion.")

    out_df = pd.DataFrame(records)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dst, index=False)
    print(
        f"Wrote incremental buffer {dst}: rows={len(out_df)} "
        f"stage_start={stage_start} stage_limit={stage_limit}",
        flush=True,
    )
    print(
        "completion_token_len min/mean/max:",
        int(out_df["completion_token_len"].min()),
        float(out_df["completion_token_len"].mean()),
        int(out_df["completion_token_len"].max()),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert a cumulative sampled block buffer into an incremental buffer.")
    parser.add_argument("--input_buffer", type=str, required=True)
    parser.add_argument("--output_buffer", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--block_idx", type=int, required=True)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--drop_empty", action="store_true")
    args = parser.parse_args()
    convert_buffer(args)


if __name__ == "__main__":
    main()
