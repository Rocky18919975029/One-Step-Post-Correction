import argparse
import gc
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from tqdm import tqdm

from blockwise_power_tb_train import (
    MODEL_NAME_BY_KEY,
    cleanup_distributed,
    completion_end,
    enable_gradient_checkpointing,
    evaluate_model,
    init_distributed,
    load_checkpoint_state,
    load_lora_model,
    load_math_dataset,
    maybe_init_wandb,
    parse_answer,
    read_csv_if_nonempty,
    save_checkpoint,
    score_completion,
    seed_everything,
    unwrap_model,
    vargrad_tb_loss,
)
from power_samp_utils import format_prompt


def resolve_data_path(path):
    path = Path(path)
    if path.exists():
        return path
    return Path(__file__).resolve().parent / path


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_vllm(model_name, args, adapter_path=None):
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise ImportError("Install vLLM in the psamp environment to use buffer sampling.") from exc

    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "dtype": args.vllm_dtype,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "enable_lora": adapter_path is not None,
        "max_model_len": args.vllm_max_model_len,
    }
    llm = LLM(**llm_kwargs)
    lora_request = None
    if adapter_path is not None:
        lora_request = LoRARequest("adapter", 1, str(adapter_path))
    return llm, SamplingParams, lora_request


def vllm_generate_texts(llm, sampling_params_cls, lora_request, prompts, args, max_tokens=None, n=1, desc="vLLM generate"):
    max_tokens = args.max_completion_tokens if max_tokens is None else max_tokens
    sampling_params = sampling_params_cls(
        n=n,
        temperature=args.temperature,
        max_tokens=max_tokens,
    )
    generated = []
    batch_size = max(1, args.vllm_batch_size)
    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start:start + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
        generated.extend([[choice.text for choice in output.outputs] for output in outputs])
    return generated


def build_vllm_prefixes(llm, sampling_params_cls, lora_request, prompts, block_idx, args):
    if block_idx == 1:
        return prompts

    prefix_new_tokens = (block_idx - 1) * args.block_size
    prefix_outputs = vllm_generate_texts(
        llm,
        sampling_params_cls,
        lora_request,
        prompts,
        args,
        max_tokens=prefix_new_tokens,
        n=1,
        desc=f"block {block_idx} prefix generation",
    )
    return [prompt + choices[0] for prompt, choices in zip(prompts, prefix_outputs)]


def generate_stage_buffer(model_name, tokenizer, dataset, block_idx, args, adapter_path, output_dir):
    prompts = [format_prompt(row["prompt"], args.model, tokenizer, cot=True) for row in dataset]
    llm, sampling_params_cls, lora_request = load_vllm(model_name, args, adapter_path)
    try:
        prefixes = build_vllm_prefixes(llm, sampling_params_cls, lora_request, prompts, block_idx, args)
        completion_outputs = vllm_generate_texts(
            llm,
            sampling_params_cls,
            lora_request,
            prefixes,
            args,
            max_tokens=args.max_completion_tokens,
            n=args.completions_per_prefix,
            desc=f"block {block_idx} completion generation",
        )
    finally:
        del llm
        clear_cuda()

    records = []
    for example_idx, (row, prefix_text, completions) in enumerate(zip(dataset, prefixes, completion_outputs)):
        prefix_token_len = len(tokenizer.encode(prefix_text))
        for sample_idx, completion in enumerate(completions):
            reward, parsed = score_completion(completion, row["answer"])
            records.append(
                {
                    "block_idx": block_idx,
                    "example_idx": example_idx,
                    "sample_idx": sample_idx,
                    "question": row["prompt"],
                    "correct_answer": row["answer"],
                    "prefix_token_len": prefix_token_len,
                    "prefix_text": prefix_text,
                    "completion_token_len": len(tokenizer.encode(completion, add_special_tokens=False)),
                    "completion": completion,
                    "parsed_answer": parsed,
                    "has_boxed_answer": parsed is not None,
                    "reward": reward,
                }
            )

    buffer_dir = output_dir / "buffers"
    buffer_dir.mkdir(parents=True, exist_ok=True)
    buffer_path = buffer_dir / f"block_{block_idx}.csv"
    pd.DataFrame(records).to_csv(buffer_path, index=False)
    print(f"Saved buffer {buffer_path}: {len(records)} samples", flush=True)
    return buffer_path


def encode_buffer_group(tokenizer, rows, device):
    sequences = []
    prompt_lens = []
    rewards = []
    for _, row in rows.iterrows():
        prefix_ids = tokenizer.encode(str(row["prefix_text"]))
        completion_ids = tokenizer.encode(str(row["completion"]), add_special_tokens=False)
        seq = prefix_ids + completion_ids
        sequences.append(seq)
        prompt_lens.append(len(prefix_ids))
        rewards.append(float(row["reward"]))

    max_len = max(len(seq) for seq in sequences)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padded = []
    attention_masks = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_token_id] * pad_len)
        attention_masks.append([1] * len(seq) + [0] * pad_len)

    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        prompt_lens,
        torch.tensor(attention_masks, dtype=torch.long, device=device),
        torch.tensor(rewards, dtype=torch.float32, device=device),
    )


def train_stage_from_buffer(
    model,
    tokenizer,
    optimizer,
    buffer_df,
    dataset,
    block_idx,
    args,
    rank,
    world_size,
    global_step,
    metrics,
    sample_records,
    wandb_run,
):
    order = list(range(len(dataset)))
    for epoch in range(args.epochs):
        random.shuffle(order)
        rank_order = order
        if world_size > 1:
            remainder = len(rank_order) % world_size
            if remainder:
                rank_order.extend(rank_order[: world_size - remainder])
            rank_order = rank_order[rank::world_size]

        for start in tqdm(range(0, len(rank_order), args.batch_size), desc=f"block {block_idx} epoch {epoch}"):
            batch_indices = rank_order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            step_id = global_step + 1

            loss_sum = 0.0
            reward_sum = 0.0
            logp_theta_sum = 0.0
            logp_ref_sum = 0.0
            total_sequences = len(batch_indices) * args.completions_per_prefix

            for micro_start in range(0, len(batch_indices), args.micro_batch_size):
                micro_indices = batch_indices[micro_start:micro_start + args.micro_batch_size]
                micro_rows = []
                for example_idx in micro_indices:
                    rows = buffer_df[buffer_df["example_idx"] == example_idx].sort_values("sample_idx")
                    if len(rows) < args.completions_per_prefix:
                        rows = rows.sample(
                            n=args.completions_per_prefix,
                            replace=True,
                            random_state=args.seed + block_idx + example_idx,
                        ).sort_values("sample_idx")
                    else:
                        rows = rows.head(args.completions_per_prefix)
                    micro_rows.append(rows)
                micro_df = pd.concat(micro_rows, ignore_index=True)

                sequences, prompt_lens, attention_masks, rewards = encode_buffer_group(
                    tokenizer,
                    micro_df,
                    next(unwrap_model(model).parameters()).device,
                )
                loss, logp_theta, logp_ref = vargrad_tb_loss(
                    model,
                    tokenizer,
                    sequences,
                    prompt_lens,
                    attention_masks,
                    rewards,
                    args.alpha,
                    args.beta,
                    args.completions_per_prefix,
                    args.score_micro_batch_size,
                )
                micro_sequences = len(micro_df)
                (loss * (micro_sequences / total_sequences)).backward()

                loss_sum += float(loss.detach().cpu()) * micro_sequences
                reward_sum += float(rewards.sum().detach().cpu())
                logp_theta_sum += float(logp_theta.sum().cpu())
                logp_ref_sum += float(logp_ref.sum().cpu())

                if args.save_samples:
                    for row_idx, row in micro_df.iterrows():
                        completion_len = int(
                            completion_end(
                                sequences[row_idx],
                                prompt_lens[row_idx],
                                tokenizer.eos_token_id,
                            )
                            - prompt_lens[row_idx]
                        )
                        sample_records.append(
                            {
                                "step": step_id,
                                "epoch": epoch,
                                "block_idx": block_idx,
                                "rank": rank,
                                "example_idx": int(row["example_idx"]),
                                "sample_idx": int(row["sample_idx"]),
                                "question": row["question"],
                                "correct_answer": row["correct_answer"],
                                "prefix_token_len": int(row["prefix_token_len"]),
                                "prefix_text": row["prefix_text"],
                                "completion_token_len": completion_len,
                                "completion": row["completion"],
                                "parsed_answer": row["parsed_answer"],
                                "has_boxed_answer": bool(row["has_boxed_answer"]),
                                "reward": float(row["reward"]),
                                "logp_theta": float(logp_theta[row_idx].cpu()),
                                "logp_ref": float(logp_ref[row_idx].cpu()),
                            }
                        )

            optimizer.step()
            global_step = step_id
            record = {
                "step": global_step,
                "epoch": epoch,
                "block_idx": block_idx,
                "rank": rank,
                "loss": loss_sum / total_sequences,
                "reward_mean": reward_sum / total_sequences,
                "logp_theta_mean": logp_theta_sum / total_sequences,
                "logp_ref_mean": logp_ref_sum / total_sequences,
            }
            metrics.append(record)
            print(record, flush=True)
            if wandb_run is not None:
                wandb_run.log(record, step=global_step)
    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--eval_data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--output_dir", type=str, default="results/blockwise_tb_buffer")
    parser.add_argument("--model", type=str, default="qwen", choices=sorted(MODEL_NAME_BY_KEY))
    parser.add_argument("--max_examples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--score_micro_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--completions_per_prefix", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_every_block", action="store_true")
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="one-step-post-correction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default="allow")
    parser.add_argument("--eval_every_block", action="store_true")
    parser.add_argument("--eval_examples", type=int, default=100)
    parser.add_argument("--eval_max_new_tokens", type=int, default=3072)
    parser.add_argument("--eval_temperature", type=float, default=0.25)
    parser.add_argument("--eval_do_sample", action="store_true")
    parser.add_argument("--ddp_timeout_minutes", type=int, default=120)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_batch_size", type=int, default=8)
    args = parser.parse_args()

    distributed, rank, local_rank, world_size, distributed_device = init_distributed(args.ddp_timeout_minutes)
    seed_everything(args.seed + rank)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    dataset = load_math_dataset(args.data_path)[: args.max_examples]
    eval_rows = load_math_dataset(args.eval_data_path)[: args.eval_examples] if args.eval_every_block else []
    model_name = MODEL_NAME_BY_KEY[args.model]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resume_state = None
    start_block_idx = 1
    global_step = 0
    adapter_path = None
    if args.resume_from_checkpoint:
        checkpoint_dir = Path(args.resume_from_checkpoint)
        adapter_path = checkpoint_dir / "adapter"
        resume_state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)["state"]
        start_block_idx = int(resume_state.get("next_block_idx", 1))
        global_step = int(resume_state.get("global_step", 0))

    wandb_run = maybe_init_wandb(args, rank, resume_state)
    if rank == 0 and wandb_run is not None and args.wandb_id is None:
        args.wandb_id = wandb_run.id

    metrics = []
    sample_records = []
    eval_records = []

    for block_idx in range(start_block_idx, args.num_blocks + 1):
        stage_adapter_path = adapter_path
        if rank == 0:
            generate_stage_buffer(model_name, tokenizer, dataset, block_idx, args, stage_adapter_path, output_dir)
        if distributed:
            dist.barrier()

        buffer_path = output_dir / "buffers" / f"block_{block_idx}.csv"
        buffer_df = pd.read_csv(buffer_path)

        model = load_lora_model(model_name, args.torch_dtype, distributed_device, stage_adapter_path)
        if args.gradient_checkpointing:
            enable_gradient_checkpointing(model)
        model.train()
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr,
        )
        if args.resume_from_checkpoint and block_idx == start_block_idx:
            load_checkpoint_state(args.resume_from_checkpoint, optimizer, distributed_device)

        global_step = train_stage_from_buffer(
            model,
            tokenizer,
            optimizer,
            buffer_df,
            dataset,
            block_idx,
            args,
            rank,
            world_size,
            global_step,
            metrics,
            sample_records,
            wandb_run,
        )

        if args.save_every_block:
            block_dir = output_dir / f"block_{block_idx}"
            if rank == 0:
                unwrap_model(model).save_pretrained(block_dir)
                tokenizer.save_pretrained(block_dir)
            if distributed:
                dist.barrier()

        if args.eval_every_block and rank == 0:
            eval_metrics = evaluate_model(model, tokenizer, eval_rows, args)
            eval_metrics = {**eval_metrics, "block_idx": block_idx, "step": global_step}
            eval_records.append(eval_metrics)
            print(eval_metrics, flush=True)
            if wandb_run is not None:
                wandb_run.log(eval_metrics, step=global_step)

        if rank == 0:
            checkpoint_state = {
                "next_block_idx": block_idx + 1,
                "global_step": global_step,
                "wandb_id": wandb_run.id if wandb_run is not None else args.wandb_id,
                "args": vars(args),
            }
            save_checkpoint(output_dir, model, tokenizer, optimizer, checkpoint_state, distributed=False)
            adapter_path = output_dir / "checkpoint_latest" / "adapter"
        if distributed:
            dist.barrier()
            adapter_path = output_dir / "checkpoint_latest" / "adapter"

        del model
        del optimizer
        clear_cuda()

    pd.DataFrame(metrics).to_csv(output_dir / f"metrics_rank{rank}.csv", index=False)
    if args.save_samples:
        pd.DataFrame(sample_records).to_csv(output_dir / f"samples_rank{rank}.csv", index=False)
    if rank == 0 and eval_records:
        pd.DataFrame(eval_records).to_csv(output_dir / "eval_metrics.csv", index=False)

    if distributed:
        dist.barrier()

    if rank == 0:
        metric_frames = [
            frame
            for path in sorted(output_dir.glob("metrics_rank*.csv"))
            for frame in [read_csv_if_nonempty(path)]
            if frame is not None
        ]
        if metric_frames:
            pd.concat(metric_frames, ignore_index=True).to_csv(output_dir / "metrics.csv", index=False)

        if args.save_samples:
            sample_frames = [
                frame
                for path in sorted(output_dir.glob("samples_rank*.csv"))
                for frame in [read_csv_if_nonempty(path)]
                if frame is not None
            ]
            if sample_frames:
                pd.concat(sample_frames, ignore_index=True).to_csv(output_dir / "samples.csv", index=False)

        if wandb_run is not None:
            wandb_run.finish()

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
