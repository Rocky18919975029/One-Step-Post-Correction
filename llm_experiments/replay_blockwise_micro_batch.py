import argparse
import os
from pathlib import Path
import re

import pandas as pd
import torch
import transformers

from blockwise_power_tb_buffer_train import encode_buffer_group
from blockwise_power_tb_train import (
    MODEL_NAME_BY_KEY,
    enable_gradient_checkpointing,
    load_checkpoint_state,
    load_lora_model,
    vargrad_tb_loss,
)


def natural_key(text):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def main():
    parser = argparse.ArgumentParser(description="Replay one dumped blockwise micro-batch against a saved checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--micro_batch_csv", type=str, action="append", required=True)
    parser.add_argument("--micro_batch_glob", type=str, default=None)
    parser.add_argument("--max_micro_batches", type=int, default=0)
    parser.add_argument("--model", type=str, default=None, choices=sorted(MODEL_NAME_BY_KEY))
    parser.add_argument("--torch_dtype", type=str, default=None, choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--completions_per_prefix", type=int, default=None)
    parser.add_argument("--score_micro_batch_size", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--run_backward", action="store_true")
    parser.add_argument("--run_optimizer_step", action="store_true")
    parser.add_argument("--disable_backward_scaling", action="store_true")
    args = parser.parse_args()

    micro_batch_csvs = list(args.micro_batch_csv or [])
    if args.micro_batch_glob:
        matched = sorted(
            Path().glob(args.micro_batch_glob),
            key=lambda p: natural_key(str(p)),
        )
        micro_batch_csvs.extend(str(path) for path in matched)
    if not micro_batch_csvs:
        raise ValueError("Provide at least one --micro_batch_csv or --micro_batch_glob.")
    if args.max_micro_batches and args.max_micro_batches > 0:
        micro_batch_csvs = micro_batch_csvs[: args.max_micro_batches]

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )["state"]
    saved_args = checkpoint_state.get("args", {})

    model_key = args.model or saved_args.get("model")
    if model_key is None:
        raise ValueError("Could not determine --model from args or checkpoint.")
    model_name = MODEL_NAME_BY_KEY[model_key]

    torch_dtype = args.torch_dtype or saved_args.get("torch_dtype", "bfloat16")
    attn_implementation = args.attn_implementation
    if attn_implementation is None:
        attn_implementation = saved_args.get("attn_implementation")

    alpha = args.alpha if args.alpha is not None else saved_args["alpha"]
    beta = args.beta if args.beta is not None else saved_args["beta"]
    completions_per_prefix = (
        args.completions_per_prefix
        if args.completions_per_prefix is not None
        else saved_args["completions_per_prefix"]
    )
    score_micro_batch_size = (
        args.score_micro_batch_size
        if args.score_micro_batch_size is not None
        else saved_args.get("score_micro_batch_size")
    )
    gradient_checkpointing = args.gradient_checkpointing or bool(saved_args.get("gradient_checkpointing", False))

    print(f"checkpoint_dir={checkpoint_dir}", flush=True)
    print(f"micro_batch_csvs={micro_batch_csvs}", flush=True)
    print(
        f"model={model_key} torch_dtype={torch_dtype} attn_implementation={attn_implementation} "
        f"alpha={alpha} beta={beta} completions_per_prefix={completions_per_prefix} "
        f"score_micro_batch_size={score_micro_batch_size} gradient_checkpointing={gradient_checkpointing}",
        flush=True,
    )
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_lora_model(
        model_name,
        torch_dtype,
        device=device,
        adapter_path=checkpoint_dir / "adapter",
        attn_implementation=attn_implementation,
    )
    if gradient_checkpointing:
        enable_gradient_checkpointing(model)
    model.train()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(saved_args.get("lr", 1e-5)),
    )
    load_checkpoint_state(checkpoint_dir, optimizer if args.run_optimizer_step else None, device)

    if args.run_backward or args.run_optimizer_step:
        optimizer.zero_grad(set_to_none=True)

    total_sequences = int(saved_args.get("batch_size", 1)) * completions_per_prefix
    for idx, micro_batch_csv in enumerate(micro_batch_csvs, start=1):
        print(f"=== replay {idx}/{len(micro_batch_csvs)}: {micro_batch_csv} ===", flush=True)
        micro_df = pd.read_csv(micro_batch_csv)
        sequences, prompt_lens, attention_masks, rewards = encode_buffer_group(
            tokenizer,
            micro_df,
            device,
        )
        print(
            f"rows={len(micro_df)} seq_shape={tuple(sequences.shape)} max_prompt_len={max(prompt_lens)} "
            f"reward_mean={float(rewards.mean().detach().cpu()):.4f}",
            flush=True,
        )

        print("loss forward begin", flush=True)
        loss, logp_theta, logp_ref = vargrad_tb_loss(
            model,
            tokenizer,
            sequences,
            prompt_lens,
            attention_masks,
            rewards,
            alpha,
            beta,
            completions_per_prefix,
            score_micro_batch_size,
        )
        print("loss forward end", flush=True)
        print(
            f"loss={float(loss.detach().cpu()):.6f} "
            f"logp_theta_mean={float(logp_theta.mean().detach().cpu()):.6f} "
            f"logp_ref_mean={float(logp_ref.mean().detach().cpu()):.6f}",
            flush=True,
        )

        if args.run_backward or args.run_optimizer_step:
            backward_loss = loss
            if not args.disable_backward_scaling:
                backward_loss = loss * (len(micro_df) / total_sequences)
                print(
                    f"backward scale={len(micro_df)}/{total_sequences}={len(micro_df) / total_sequences:.6f}",
                    flush=True,
                )
            print("backward begin", flush=True)
            backward_loss.backward()
            print("backward end", flush=True)

    if args.run_optimizer_step:
        print("optimizer step begin", flush=True)
        optimizer.step()
        print("optimizer step end", flush=True)


if __name__ == "__main__":
    main()
