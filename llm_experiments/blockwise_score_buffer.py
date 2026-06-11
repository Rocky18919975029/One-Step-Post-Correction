import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from blockwise_power_tb_buffer_train import encode_buffer_group
from blockwise_power_tb_train import (
    completion_logprob,
    completion_end,
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

TOKEN_SCORE_COLUMNS = {
    "token_logp_ref",
    "token_logp_theta_score",
    "token_log_z_hat",
    "token_tb_target",
}


def completion_token_logprob_lists(model, sequences, prompt_lens, attention_masks, eos_token_id):
    output = model(sequences, attention_mask=attention_masks)
    logits = output.logits[:, :-1, :]
    labels = sequences[:, 1:]

    rows = []
    for row_idx, prompt_len in enumerate(prompt_lens):
        start = max(prompt_len - 1, 0)
        end = completion_end(sequences[row_idx], prompt_len, eos_token_id)
        slice_end = max(end - 1, start)
        row_logits = logits[row_idx, start:slice_end]
        if row_logits.numel() == 0:
            rows.append(torch.empty(0, device=logits.device, dtype=torch.float32))
            continue
        row_labels = labels[row_idx, start:slice_end]
        row_logprobs = torch.nn.functional.log_softmax(row_logits.float(), dim=-1)
        rows.append(row_logprobs.gather(-1, row_labels.unsqueeze(-1)).squeeze(-1))
    return rows


def dump_float_list(values):
    return json.dumps([float(x) for x in values], separators=(",", ":"))


def score_buffer(args):
    buffer_path = Path(args.buffer_path)
    if not buffer_path.exists():
        raise FileNotFoundError(f"Missing buffer: {buffer_path}")

    df = pd.read_csv(buffer_path)
    required_columns = SCORE_COLUMNS | (TOKEN_SCORE_COLUMNS if args.loss_level == "token" else set())
    if required_columns.issubset(df.columns) and not args.force:
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
    token_ref_values = [None] * len(df)
    token_theta_values = [None] * len(df)
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
            if args.loss_level == "token":
                token_ref = completion_token_logprob_lists(
                    ref_model,
                    sequences,
                    prompt_lens,
                    attention_masks,
                    tokenizer.eos_token_id,
                )
                token_theta = completion_token_logprob_lists(
                    actor,
                    sequences,
                    prompt_lens,
                    attention_masks,
                    tokenizer.eos_token_id,
                )
                logp_ref = torch.stack([values.sum() for values in token_ref])
                logp_theta = torch.stack([values.sum() for values in token_theta])
            else:
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
            if args.loss_level == "token":
                for offset, values in enumerate(token_ref):
                    token_ref_values[start + offset] = [float(x) for x in values.detach().cpu().double().tolist()]
                for offset, values in enumerate(token_theta):
                    token_theta_values[start + offset] = [float(x) for x in values.detach().cpu().double().tolist()]

    df["logp_ref"] = logp_ref_values.numpy()
    df["logp_theta_score"] = logp_theta_values.numpy()
    df["tb_target"] = args.alpha * df["logp_ref"] + df["reward"].astype(float) / args.beta
    log_z_terms = args.alpha * df["logp_ref"] - df["logp_theta_score"] + df["reward"].astype(float) / args.beta
    df["log_z_hat"] = log_z_terms.groupby(df["example_idx"]).transform("mean")

    if args.loss_level == "token":
        token_targets = []
        token_z_values = [None] * len(df)
        for row_idx, (ref_values, reward) in enumerate(zip(token_ref_values, df["reward"].astype(float).tolist())):
            if ref_values is None:
                raise RuntimeError(f"Missing token ref scores for row {row_idx}")
            token_targets.append([args.alpha * ref_value + reward / args.beta for ref_value in ref_values])

        for _, group in df.groupby("example_idx", sort=False):
            group_indices = list(group.index)
            max_len = max(len(token_ref_values[idx]) for idx in group_indices)
            for token_idx in range(max_len):
                terms = []
                present_indices = []
                for row_idx in group_indices:
                    if token_idx < len(token_ref_values[row_idx]):
                        reward = float(df.at[row_idx, "reward"])
                        terms.append(
                            args.alpha * token_ref_values[row_idx][token_idx]
                            - token_theta_values[row_idx][token_idx]
                            + reward / args.beta
                        )
                        present_indices.append(row_idx)
                mean_value = float(sum(terms) / len(terms))
                for row_idx in present_indices:
                    if token_z_values[row_idx] is None:
                        token_z_values[row_idx] = [0.0] * len(token_ref_values[row_idx])
                    token_z_values[row_idx][token_idx] = mean_value

        token_z_values = [[] if values is None else values for values in token_z_values]
        df["token_logp_ref"] = [dump_float_list(values or []) for values in token_ref_values]
        df["token_logp_theta_score"] = [dump_float_list(values or []) for values in token_theta_values]
        df["token_tb_target"] = [dump_float_list(values) for values in token_targets]
        df["token_log_z_hat"] = [dump_float_list(values) for values in token_z_values]

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
    parser.add_argument("--loss_level", type=str, default="sequence", choices=["sequence", "token"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    score_buffer(args)


if __name__ == "__main__":
    main()
