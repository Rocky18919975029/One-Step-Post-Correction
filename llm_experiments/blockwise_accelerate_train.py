import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from blockwise_power_tb_buffer_train import PRECOMPUTED_SCORE_COLUMNS
from blockwise_power_tb_train import (
    completion_logprob,
    enable_gradient_checkpointing,
    load_lora_model,
    resolve_model_name,
    seed_everything,
    unwrap_model,
)


def has_precomputed_scores(df):
    return PRECOMPUTED_SCORE_COLUMNS.issubset(df.columns) and not df[list(PRECOMPUTED_SCORE_COLUMNS)].isna().any().any()


class ScoredBufferDataset(Dataset):
    def __init__(self, buffer_df, completions_per_prefix):
        df = buffer_df.sort_values(["example_idx", "sample_idx"]).copy()
        if completions_per_prefix is not None:
            df = df[df["sample_idx"] < completions_per_prefix].copy()
        self.rows = df.to_dict("records")
        if not self.rows:
            raise ValueError("Scored buffer has no rows after filtering by completions_per_prefix.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class ScoredBufferCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        rows = list(examples)
        texts = [str(row["prefix_text"]) + str(row["completion"]) for row in rows]
        encoded = self.tokenizer(texts, padding=True, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "prompt_lens": [int(row["prefix_token_len"]) for row in rows],
            "log_z_hat": torch.tensor([float(row["log_z_hat"]) for row in rows], dtype=torch.float32),
            "tb_target": torch.tensor([float(row["tb_target"]) for row in rows], dtype=torch.float32),
            "logp_ref": torch.tensor([float(row["logp_ref"]) for row in rows], dtype=torch.float32),
            "reward": torch.tensor([float(row["reward"]) for row in rows], dtype=torch.float32),
        }


def save_accelerate_checkpoint(output_dir, model, tokenizer, optimizer, state, accelerator):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoint_latest"
    tmp_dir = output_dir / "checkpoint_latest_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    unwrap_model(unwrapped).save_pretrained(tmp_dir / "adapter")
    tokenizer.save_pretrained(tmp_dir / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "state": state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        tmp_dir / "training_state.pt",
    )

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def save_block_adapter(output_dir, block_idx, model, tokenizer, accelerator):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    block_dir = Path(output_dir) / f"block_{block_idx}"
    block_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrap_model(unwrapped).save_pretrained(block_dir)
    tokenizer.save_pretrained(block_dir)


def load_resume_state(checkpoint_dir, optimizer):
    if checkpoint_dir is None:
        return 0
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        return 0
    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    state = checkpoint.get("state", {})
    return int(state.get("global_step", 0) or 0)


def train_scored_block(args):
    accumulation_rows = max(1, args.batch_size * args.completions_per_prefix)
    accelerator = Accelerator(
        gradient_accumulation_steps=max(1, math.ceil(accumulation_rows / args.micro_batch_size))
    )
    seed_everything(args.seed + accelerator.process_index)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer_path = Path(args.buffer_path)
    if not buffer_path.exists():
        raise FileNotFoundError(f"Missing scored buffer: {buffer_path}")

    buffer_df = pd.read_csv(buffer_path)
    if not has_precomputed_scores(buffer_df):
        missing = sorted(PRECOMPUTED_SCORE_COLUMNS.difference(buffer_df.columns))
        raise ValueError(f"Buffer is not pre-scored. Missing/invalid columns: {missing or sorted(PRECOMPUTED_SCORE_COLUMNS)}")

    if args.max_examples is not None:
        allowed = set(buffer_df["example_idx"].drop_duplicates().head(args.max_examples))
        buffer_df = buffer_df[buffer_df["example_idx"].isin(allowed)].copy()

    model_name = resolve_model_name(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ScoredBufferDataset(buffer_df, args.completions_per_prefix)
    collator = ScoredBufferCollator(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    if adapter_path is not None and not adapter_path.exists():
        adapter_path = None

    model = load_lora_model(
        model_name,
        args.torch_dtype,
        accelerator.device,
        adapter_path,
        attn_implementation=args.attn_implementation,
    )
    model.train()
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr)
    checkpoint_step = load_resume_state(args.resume_from_checkpoint, optimizer)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    global_step = int(args.start_step or checkpoint_step)
    metrics = []
    for epoch in range(args.epochs):
        total_optimizer_steps = math.ceil(len(dataloader) / accelerator.gradient_accumulation_steps)
        progress = tqdm(
            total=total_optimizer_steps,
            desc=f"block {args.block_idx} epoch {epoch}",
            disable=not accelerator.is_main_process,
        )
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                logp_theta = completion_logprob(
                    model,
                    batch["input_ids"],
                    batch["prompt_lens"],
                    batch["attention_mask"],
                    tokenizer.eos_token_id,
                )
                log_z_hat = batch["log_z_hat"].to(logp_theta.device, dtype=logp_theta.dtype)
                target = batch["tb_target"].to(logp_theta.device, dtype=logp_theta.dtype)
                loss = (log_z_hat + logp_theta - target).pow(2).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    gathered = accelerator.gather_for_metrics(
                        torch.stack(
                            [
                                loss.detach().float(),
                                batch["reward"].to(logp_theta.device).float().mean(),
                                logp_theta.detach().float().mean(),
                                batch["logp_ref"].to(logp_theta.device).float().mean(),
                            ]
                        ).unsqueeze(0)
                    )
                    metric = gathered.mean(dim=0).cpu().tolist()
                    record = {
                        "step": global_step,
                        "epoch": epoch,
                        "block_idx": args.block_idx,
                        "loss": metric[0],
                        "reward_mean": metric[1],
                        "logp_theta_mean": metric[2],
                        "logp_ref_mean": metric[3],
                    }
                    metrics.append(record)
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{metric[0]:.4f}",
                        reward=f"{metric[1]:.4f}",
                    )
                    if accelerator.is_main_process and (args.log_every <= 0 or global_step % args.log_every == 0):
                        print(record, flush=True)
        progress.close()

    state = {
        "global_step": global_step,
        "next_block_idx": args.block_idx + 1,
        "current_block_idx": None,
        "wandb_id": None,
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_run_name": None,
    }
    if args.save_every_block:
        save_block_adapter(output_dir, args.block_idx, model, tokenizer, accelerator)
    save_accelerate_checkpoint(output_dir, model, tokenizer, optimizer, state, accelerator)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metrics_path = output_dir / "metrics.csv"
        old_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        new_metrics = pd.DataFrame(metrics)
        pd.concat([old_metrics, new_metrics], ignore_index=True).to_csv(metrics_path, index=False)
        print(f"[block {args.block_idx}] accelerate checkpoint_latest updated", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train one pre-scored block with a standard Accelerate/DDP loop.")
    parser.add_argument("--buffer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--block_idx", type=int, required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--completions_per_prefix", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_every_block", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()
    train_scored_block(args)


if __name__ == "__main__":
    main()
