import argparse
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from blockwise_power_tb_buffer_train import encode_buffer_group
from blockwise_power_tb_train import (
    completion_logprob,
    load_lora_model,
    load_reference_model,
    resolve_model_name,
    sync_cuda_if_available,
)


SCORE_COLUMNS = {
    "logp_ref",
    "logp_theta_score",
    "log_z_hat",
    "tb_target",
}


def score_buffer(args):
    buffer_path = Path(args.buffer_path)
    if not buffer_path.exists():
        raise FileNotFoundError(f"Missing buffer: {buffer_path}")

    df = pd.read_csv(buffer_path)
    if SCORE_COLUMNS.issubset(df.columns) and not args.force:
        print(f"Buffer already has score columns: {buffer_path}", flush=True)
        return

    model_name = resolve_model_name(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    actor = load_lora_model(
        model_name,
        args.torch_dtype,
        device,
        Path(args.adapter_path) if args.adapter_path else None,
        attn_implementation=args.attn_implementation,
    )
    actor.eval()
    ref_model = load_reference_model(
        model_name,
        args.torch_dtype,
        device,
        attn_implementation=args.attn_implementation,
    )

    df = df.sort_values(["example_idx", "sample_idx"]).reset_index(drop=True)
    logp_ref_values = torch.empty(len(df), dtype=torch.float64)
    logp_theta_values = torch.empty(len(df), dtype=torch.float64)
    batch_size = max(1, int(args.score_batch_size))

    with torch.no_grad():
        for start in tqdm(range(0, len(df), batch_size), desc=f"score {buffer_path.name}"):
            end = min(start + batch_size, len(df))
            batch_df = df.iloc[start:end]
            sequences, prompt_lens, attention_masks, _ = encode_buffer_group(
                tokenizer,
                batch_df,
                device,
            )
            logp_ref = completion_logprob(
                ref_model,
                sequences,
                prompt_lens,
                attention_masks,
                tokenizer.eos_token_id,
            )
            logp_theta = completion_logprob(
                actor,
                sequences,
                prompt_lens,
                attention_masks,
                tokenizer.eos_token_id,
            )
            sync_cuda_if_available()
            logp_ref_values[start:end] = logp_ref.detach().cpu().double()
            logp_theta_values[start:end] = logp_theta.detach().cpu().double()

    df["logp_ref"] = logp_ref_values.numpy()
    df["logp_theta_score"] = logp_theta_values.numpy()
    df["tb_target"] = args.alpha * df["logp_ref"] + df["reward"].astype(float) / args.beta
    log_z_terms = args.alpha * df["logp_ref"] - df["logp_theta_score"] + df["reward"].astype(float) / args.beta
    df["log_z_hat"] = log_z_terms.groupby(df["example_idx"]).transform("mean")

    tmp_path = buffer_path.with_suffix(buffer_path.suffix + ".scored.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(buffer_path)
    print(f"Wrote scored buffer {buffer_path}: {len(df)} rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Precompute blockwise buffer logprobs and Z estimates.")
    parser.add_argument("--buffer_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--score_batch_size", type=int, default=1)
    parser.add_argument("--completions_per_prefix", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    score_buffer(args)


if __name__ == "__main__":
    main()
