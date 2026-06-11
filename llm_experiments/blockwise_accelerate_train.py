import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from blockwise_power_tb_buffer_train import PRECOMPUTED_SCORE_COLUMNS, PRECOMPUTED_TOKEN_SCORE_COLUMNS
from blockwise_power_tb_train import (
    completion_logprob,
    completion_end,
    enable_gradient_checkpointing,
    load_lora_model,
    maybe_init_wandb,
    resolve_model_name,
    seed_everything,
    unwrap_model,
)


def has_precomputed_scores(df):
    return PRECOMPUTED_SCORE_COLUMNS.issubset(df.columns) and not df[list(PRECOMPUTED_SCORE_COLUMNS)].isna().any().any()


def has_precomputed_token_scores(df):
    return PRECOMPUTED_TOKEN_SCORE_COLUMNS.issubset(df.columns) and not df[list(PRECOMPUTED_TOKEN_SCORE_COLUMNS)].isna().any().any()


def parse_float_list(value):
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(x) for x in json.loads(value)]


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
    def __init__(self, tokenizer, loss_level):
        self.tokenizer = tokenizer
        self.loss_level = loss_level

    def __call__(self, examples):
        rows = list(examples)
        texts = [str(row["prefix_text"]) + str(row["completion"]) for row in rows]
        encoded = self.tokenizer(texts, padding=True, return_tensors="pt")
        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "prompt_lens": [int(row["prefix_token_len"]) for row in rows],
            "log_z_hat": torch.tensor([float(row["log_z_hat"]) for row in rows], dtype=torch.float32),
            "tb_target": torch.tensor([float(row["tb_target"]) for row in rows], dtype=torch.float32),
            "logp_ref": torch.tensor([float(row["logp_ref"]) for row in rows], dtype=torch.float32),
            "reward": torch.tensor([float(row["reward"]) for row in rows], dtype=torch.float32),
        }
        if self.loss_level == "token":
            token_z = [parse_float_list(row["token_log_z_hat"]) for row in rows]
            token_target = [parse_float_list(row["token_tb_target"]) for row in rows]
            token_ref = [parse_float_list(row["token_logp_ref"]) for row in rows]
            max_len = max(len(values) for values in token_target) if token_target else 0
            token_mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
            token_z_tensor = torch.zeros((len(rows), max_len), dtype=torch.float32)
            token_target_tensor = torch.zeros((len(rows), max_len), dtype=torch.float32)
            token_ref_tensor = torch.zeros((len(rows), max_len), dtype=torch.float32)
            for row_idx, (z_values, target_values, ref_values) in enumerate(zip(token_z, token_target, token_ref)):
                length = len(target_values)
                token_mask[row_idx, :length] = True
                token_z_tensor[row_idx, :length] = torch.tensor(z_values, dtype=torch.float32)
                token_target_tensor[row_idx, :length] = torch.tensor(target_values, dtype=torch.float32)
                token_ref_tensor[row_idx, :length] = torch.tensor(ref_values, dtype=torch.float32)
            batch.update(
                {
                    "token_log_z_hat": token_z_tensor,
                    "token_tb_target": token_target_tensor,
                    "token_logp_ref": token_ref_tensor,
                    "token_mask": token_mask,
                }
            )
        return batch


def completion_token_logprob_padded(model, sequences, prompt_lens, attention_masks, eos_token_id, target_width):
    output = model(sequences, attention_mask=attention_masks)
    logits = output.logits[:, :-1, :]
    labels = sequences[:, 1:]
    padded = logits.new_zeros((sequences.shape[0], target_width), dtype=torch.float32)
    mask = torch.zeros((sequences.shape[0], target_width), dtype=torch.bool, device=sequences.device)

    for row_idx, prompt_len in enumerate(prompt_lens):
        start = max(prompt_len - 1, 0)
        end = completion_end(sequences[row_idx], prompt_len, eos_token_id)
        slice_end = min(max(end - 1, start), start + target_width)
        row_logits = logits[row_idx, start:slice_end]
        if row_logits.numel() == 0:
            continue
        row_labels = labels[row_idx, start:slice_end]
        row_logprobs = F.log_softmax(row_logits.float(), dim=-1)
        gathered = row_logprobs.gather(-1, row_labels.unsqueeze(-1)).squeeze(-1)
        length = gathered.shape[0]
        padded[row_idx, :length] = gathered
        mask[row_idx, :length] = True
    return padded, mask


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
        return 0, None
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        return 0, None
    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    state = checkpoint.get("state", {})
    return int(state.get("global_step", 0) or 0), state


def should_log_wandb(args, step):
    if args.wandb_log_every <= 0:
        return True
    return args.wandb_log_every == 1 or step % args.wandb_log_every == 0


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
    if args.loss_level == "token":
        required = PRECOMPUTED_SCORE_COLUMNS | PRECOMPUTED_TOKEN_SCORE_COLUMNS
        if not (has_precomputed_scores(buffer_df) and has_precomputed_token_scores(buffer_df)):
            missing = sorted(required.difference(buffer_df.columns))
            raise ValueError(f"Buffer is not token-scored. Missing/invalid columns: {missing or sorted(required)}")
    elif not has_precomputed_scores(buffer_df):
        missing = sorted(PRECOMPUTED_SCORE_COLUMNS.difference(buffer_df.columns))
        raise ValueError(f"Buffer is not sequence-scored. Missing/invalid columns: {missing or sorted(PRECOMPUTED_SCORE_COLUMNS)}")

    if args.max_examples is not None:
        allowed = set(buffer_df["example_idx"].drop_duplicates().head(args.max_examples))
        buffer_df = buffer_df[buffer_df["example_idx"].isin(allowed)].copy()

    model_name = resolve_model_name(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = ScoredBufferDataset(buffer_df, args.completions_per_prefix)
    collator = ScoredBufferCollator(tokenizer, args.loss_level)
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
    checkpoint_step, resume_state = load_resume_state(args.resume_from_checkpoint, optimizer)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    wandb_run = maybe_init_wandb(args, 0 if accelerator.is_main_process else accelerator.process_index, resume_state)
    if accelerator.is_main_process and wandb_run is not None and args.wandb_id is None:
        args.wandb_id = wandb_run.id

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
                if args.loss_level == "token":
                    token_mask = batch["token_mask"].to(batch["input_ids"].device)
                    token_logp_theta, model_token_mask = completion_token_logprob_padded(
                        model,
                        batch["input_ids"],
                        batch["prompt_lens"],
                        batch["attention_mask"],
                        tokenizer.eos_token_id,
                        batch["token_tb_target"].shape[1],
                    )
                    token_mask = token_mask & model_token_mask
                    token_z_hat = batch["token_log_z_hat"].to(token_logp_theta.device, dtype=token_logp_theta.dtype)
                    token_target = batch["token_tb_target"].to(token_logp_theta.device, dtype=token_logp_theta.dtype)
                    token_residual = token_z_hat + token_logp_theta - token_target
                    token_mask_float = token_mask.to(token_residual.dtype)
                    token_counts = token_mask_float.sum(dim=1)
                    valid_rows = token_counts > 0
                    token_sq = token_residual.pow(2) * token_mask_float
                    per_row_loss = token_sq.sum(dim=1) / token_counts.clamp_min(1.0)
                    loss = per_row_loss[valid_rows].mean()
                    logp_theta = (token_logp_theta * token_mask.to(token_logp_theta.dtype)).sum(dim=1)
                    logp_ref_metric = (
                        batch["token_logp_ref"].to(token_logp_theta.device, dtype=token_logp_theta.dtype)
                        * token_mask.to(token_logp_theta.dtype)
                    ).sum(dim=1)
                else:
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
                    logp_ref_metric = batch["logp_ref"].to(logp_theta.device).float()
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
                                logp_ref_metric.detach().float().mean(),
                            ]
                        ).unsqueeze(0)
                    )
                    metric = gathered.mean(dim=0).cpu().tolist()
                    record = {
                        "step": global_step,
                        "epoch": epoch,
                        "block_idx": args.block_idx,
                        "loss_level": args.loss_level,
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
                    if wandb_run is not None and should_log_wandb(args, global_step):
                        wandb_run.log(record, step=global_step)
        progress.close()

    state = {
        "global_step": global_step,
        "next_block_idx": args.block_idx + 1,
        "current_block_idx": None,
        "wandb_id": wandb_run.id if wandb_run is not None else args.wandb_id,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
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
    if wandb_run is not None:
        wandb_run.finish()


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
    parser.add_argument("--loss_level", type=str, default="sequence", choices=["sequence", "token"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="one-step-post-correction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default="allow")
    parser.add_argument("--wandb_log_every", type=int, default=0)
    args = parser.parse_args()
    train_scored_block(args)


if __name__ == "__main__":
    main()
