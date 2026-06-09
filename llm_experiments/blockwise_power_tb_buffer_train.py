import argparse
import faulthandler
import gc
import json
import os
import random
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from blockwise_power_tb_train import (
    MODEL_NAME_BY_KEY,
    completion_logprob_chunks,
    completion_end,
    enable_gradient_checkpointing,
    evaluate_model,
    load_checkpoint_state,
    load_lora_model,
    load_math_dataset,
    maybe_init_wandb,
    parse_torch_dtype,
    save_checkpoint,
    score_completion,
    seed_everything,
    sync_cuda_if_available,
    unwrap_model,
    vargrad_tb_loss,
)
from power_samp_utils import format_prompt


DEBUG_HANDLE = None


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


def setup_debug_file(output_dir, name, dump_timeout_seconds=0):
    global DEBUG_HANDLE
    debug_dir = Path(output_dir) / "debug_logs"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / name
    DEBUG_HANDLE = debug_path.open("a", buffering=1)
    faulthandler.enable(DEBUG_HANDLE)
    if dump_timeout_seconds and dump_timeout_seconds > 0:
        faulthandler.dump_traceback_later(dump_timeout_seconds, repeat=True, file=DEBUG_HANDLE)
    return debug_path


def close_debug_file():
    global DEBUG_HANDLE
    faulthandler.cancel_dump_traceback_later()
    if DEBUG_HANDLE is not None:
        DEBUG_HANDLE.close()
        DEBUG_HANDLE = None


def debug_log(message, rank=None):
    timestamp = datetime.now().isoformat(timespec="seconds")
    prefix = f"[debug {timestamp} pid={os.getpid()}"
    if rank is not None:
        prefix += f" rank={rank}"
    prefix += "]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    if DEBUG_HANDLE is not None:
        print(line, file=DEBUG_HANDLE, flush=True)


def log_point(label, rank=0):
    debug_log(f"{label}", rank=rank)


def should_log_wandb_step(args, step):
    if args.wandb_log_every <= 0:
        return False
    return args.wandb_log_every == 1 or step % args.wandb_log_every == 0


def dump_micro_batch_debug(output_dir, block_idx, step_id, micro_start, micro_indices, micro_df, prompt_lens, sequences):
    debug_dir = Path(output_dir) / "debug_logs" / "micro_batches"
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = f"block{block_idx}_step{step_id}_micro{micro_start}"
    micro_df.to_csv(debug_dir / f"{stem}.csv", index=False)
    meta = {
        "block_idx": block_idx,
        "step": step_id,
        "micro_start": micro_start,
        "example_indices": [int(idx) for idx in micro_indices],
        "rows": int(len(micro_df)),
        "seq_shape": tuple(int(x) for x in sequences.shape),
        "prompt_lens": [int(x) for x in prompt_lens],
    }
    torch.save(meta, debug_dir / f"{stem}.pt")


def dump_active_forward_debug(
    output_dir,
    block_idx,
    step_id,
    micro_start,
    chunk_start,
    chunk_end,
    micro_df,
    prompt_lens,
    sequences,
    attention_masks,
):
    debug_dir = Path(output_dir) / "debug_logs" / "active_forward"
    debug_dir.mkdir(parents=True, exist_ok=True)
    chunk_df = micro_df.iloc[chunk_start:chunk_end].copy()
    chunk_sequences = sequences[chunk_start:chunk_end].detach().cpu()
    chunk_attention = attention_masks[chunk_start:chunk_end].detach().cpu()
    chunk_prompt_lens = [int(value) for value in prompt_lens[chunk_start:chunk_end]]

    meta = {
        "block_idx": int(block_idx),
        "step": int(step_id),
        "micro_start": int(micro_start),
        "chunk_start": int(chunk_start),
        "chunk_end": int(chunk_end),
        "rows": int(len(chunk_df)),
        "seq_shape": [int(value) for value in chunk_sequences.shape],
        "prompt_lens": chunk_prompt_lens,
        "attention_sums": [int(value) for value in chunk_attention.sum(dim=1).tolist()],
        "input_id_min": int(chunk_sequences.min().item()) if chunk_sequences.numel() else None,
        "input_id_max": int(chunk_sequences.max().item()) if chunk_sequences.numel() else None,
        "example_indices": [int(value) for value in chunk_df["example_idx"].tolist()] if "example_idx" in chunk_df else [],
        "sample_indices": [int(value) for value in chunk_df["sample_idx"].tolist()] if "sample_idx" in chunk_df else [],
    }

    chunk_df.to_csv(debug_dir / "active_forward.csv", index=False)
    torch.save(
        {
            "meta": meta,
            "input_ids": chunk_sequences,
            "attention_mask": chunk_attention,
        },
        debug_dir / "active_forward.pt",
    )
    with (debug_dir / "active_forward.json").open("w") as handle:
        json.dump(meta, handle, indent=2)

    stem = f"block{block_idx}_step{step_id}_micro{micro_start}_rows{chunk_start}_{chunk_end}"
    chunk_df.to_csv(debug_dir / f"{stem}.csv", index=False)
    with (debug_dir / f"{stem}.json").open("w") as handle:
        json.dump(meta, handle, indent=2)


def dump_active_backward_debug(
    output_dir,
    block_idx,
    step_id,
    micro_start,
    micro_df,
    prompt_lens,
    sequences,
    attention_masks,
):
    debug_dir = Path(output_dir) / "debug_logs" / "active_backward"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cpu_sequences = sequences.detach().cpu()
    cpu_attention = attention_masks.detach().cpu()
    meta = {
        "block_idx": int(block_idx),
        "step": int(step_id),
        "micro_start": int(micro_start),
        "rows": int(len(micro_df)),
        "seq_shape": [int(value) for value in cpu_sequences.shape],
        "prompt_lens": [int(value) for value in prompt_lens],
        "attention_sums": [int(value) for value in cpu_attention.sum(dim=1).tolist()],
        "input_id_min": int(cpu_sequences.min().item()) if cpu_sequences.numel() else None,
        "input_id_max": int(cpu_sequences.max().item()) if cpu_sequences.numel() else None,
        "example_indices": [int(value) for value in micro_df["example_idx"].tolist()] if "example_idx" in micro_df else [],
        "sample_indices": [int(value) for value in micro_df["sample_idx"].tolist()] if "sample_idx" in micro_df else [],
    }
    micro_df.to_csv(debug_dir / "active_backward.csv", index=False)
    torch.save(
        {
            "meta": meta,
            "input_ids": cpu_sequences,
            "attention_mask": cpu_attention,
        },
        debug_dir / "active_backward.pt",
    )
    with (debug_dir / "active_backward.json").open("w") as handle:
        json.dump(meta, handle, indent=2)

    stem = f"block{block_idx}_step{step_id}_micro{micro_start}"
    micro_df.to_csv(debug_dir / f"{stem}.csv", index=False)
    with (debug_dir / f"{stem}.json").open("w") as handle:
        json.dump(meta, handle, indent=2)


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
        "enforce_eager": args.vllm_enforce_eager,
        "disable_custom_all_reduce": args.vllm_disable_custom_all_reduce,
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


def partial_completion_token_limit(block_idx, args):
    return min(block_idx * args.block_size, args.max_completion_tokens)


def evaluate_model_with_vllm(model_name, tokenizer, eval_rows, args, adapter_path=None):
    if not eval_rows:
        return {}

    prompts = [format_prompt(row["prompt"], args.model, tokenizer, cot=True) for row in eval_rows]
    llm, sampling_params_cls, lora_request = load_vllm(model_name, args, adapter_path)
    try:
        sampling_kwargs = {
            "n": 1,
            "max_tokens": args.eval_max_new_tokens,
        }
        if args.eval_do_sample:
            sampling_kwargs["temperature"] = args.eval_temperature
        else:
            sampling_kwargs["temperature"] = 0.0
        sampling_params = sampling_params_cls(**sampling_kwargs)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        completions = [output.outputs[0].text for output in outputs]
    finally:
        del llm
        clear_cuda()

    rewards = []
    boxed = []
    for row, completion in zip(eval_rows, completions):
        reward, parsed = score_completion(completion, row["answer"])
        rewards.append(reward)
        boxed.append(parsed is not None)

    return {
        "eval/accuracy": float(np.mean(rewards)),
        "eval/boxed_rate": float(np.mean(boxed)),
        "eval/examples": len(eval_rows),
    }


def generate_stage_buffer(
    model_name,
    tokenizer,
    dataset,
    block_idx,
    args,
    adapter_path,
    output_dir,
    *,
    example_idx_offset=0,
    buffer_path_override=None,
):
    prompts = [format_prompt(row["prompt"], args.model, tokenizer, cot=True) for row in dataset]
    stage_token_limit = partial_completion_token_limit(block_idx, args)
    future_completions_per_partial = (
        args.future_completions_per_partial
        if args.future_completions_per_partial is not None
        else args.completions_per_prefix
    )
    future_token_budget = max(args.max_completion_tokens - stage_token_limit, 0)
    llm, sampling_params_cls, lora_request = load_vllm(model_name, args, adapter_path)
    try:
        partial_completion_outputs = vllm_generate_texts(
            llm,
            sampling_params_cls,
            lora_request,
            prompts,
            args,
            max_tokens=stage_token_limit,
            n=args.completions_per_prefix,
            desc=f"block {block_idx} partial completion generation",
        )

        future_prompts = []
        future_metadata = []
        for local_example_idx, (row, prompt_text, partials) in enumerate(zip(dataset, prompts, partial_completion_outputs)):
            example_idx = example_idx_offset + local_example_idx
            for sample_idx, partial_completion in enumerate(partials):
                partial_prefix = prompt_text + partial_completion
                partial_token_len = len(tokenizer.encode(partial_completion, add_special_tokens=False))
                future_prompts.append(partial_prefix)
                future_metadata.append(
                    {
                        "example_idx": example_idx,
                        "sample_idx": sample_idx,
                        "question": row["prompt"],
                        "correct_answer": row["answer"],
                        "prefix_text": prompt_text,
                        "prefix_token_len": len(tokenizer.encode(prompt_text)),
                        "completion": partial_completion,
                        "completion_token_len": partial_token_len,
                    }
                )

        future_outputs = []
        batch_size = max(1, args.vllm_batch_size)
        for start in tqdm(range(0, len(future_prompts), batch_size), desc=f"block {block_idx} future reward estimation"):
            batch_prompts = future_prompts[start:start + batch_size]
            if future_token_budget > 0:
                outputs = vllm_generate_texts(
                    llm,
                    sampling_params_cls,
                    lora_request,
                    batch_prompts,
                    args,
                    max_tokens=future_token_budget,
                    n=future_completions_per_partial,
                    desc=f"block {block_idx} future reward generation",
                )
            else:
                outputs = [[ "" for _ in range(future_completions_per_partial)] for _ in batch_prompts]
            future_outputs.extend(outputs)
    finally:
        del llm
        clear_cuda()

    records = []
    for meta, futures in zip(future_metadata, future_outputs):
        future_rewards = []
        first_successful_parsed = None
        for future in futures:
            full_completion = meta["completion"] + future
            future_reward, future_parsed = score_completion(full_completion, meta["correct_answer"])
            future_rewards.append(future_reward)
            if future_reward > 0 and first_successful_parsed is None:
                first_successful_parsed = future_parsed

        reward = 1.0 if any(r > 0 for r in future_rewards) else 0.0
        records.append(
            {
                "block_idx": block_idx,
                "example_idx": meta["example_idx"],
                "sample_idx": meta["sample_idx"],
                "question": meta["question"],
                "correct_answer": meta["correct_answer"],
                "prefix_token_len": meta["prefix_token_len"],
                "prefix_text": meta["prefix_text"],
                "completion_token_len": meta["completion_token_len"],
                "completion": meta["completion"],
                "parsed_answer": first_successful_parsed,
                "has_boxed_answer": first_successful_parsed is not None,
                "reward": reward,
                "future_reward_mean": float(np.mean(future_rewards)) if future_rewards else 0.0,
                "future_any_correct": bool(reward > 0),
            }
        )

    if buffer_path_override is None:
        buffer_dir = output_dir / "buffers"
        buffer_dir.mkdir(parents=True, exist_ok=True)
        buffer_path = buffer_dir / f"block_{block_idx}.csv"
    else:
        buffer_path = Path(buffer_path_override)
        buffer_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(buffer_path, index=False)
    print(f"Saved buffer {buffer_path}: {len(records)} samples", flush=True)
    return buffer_path


def vllm_subprocess_env(args):
    env = os.environ.copy()
    for key in [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_MAX_RESTARTS",
    ]:
        env.pop(key, None)
    env["TORCHELASTIC_RESTART_COUNT"] = "0"

    if args.vllm_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.vllm_visible_devices
    elif args.vllm_tensor_parallel_size == 1 and env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"].split(",")[0]
    return env


def generate_stage_buffer_subprocess(block_idx, args, adapter_path):
    command = [
        sys.executable,
        "blockwise_vllm_sample_buffer.py",
        "--data_path",
        args.data_path,
        "--output_dir",
        args.output_dir,
        "--model",
        args.model,
        "--block_idx",
        str(block_idx),
        "--max_examples",
        str(args.max_examples),
        "--block_size",
        str(args.block_size),
        "--completions_per_prefix",
        str(args.completions_per_prefix),
        "--max_completion_tokens",
        str(args.max_completion_tokens),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--vllm_dtype",
        args.vllm_dtype,
        "--vllm_tensor_parallel_size",
        str(args.vllm_tensor_parallel_size),
        "--vllm_gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm_max_model_len",
        str(args.vllm_max_model_len),
        "--vllm_batch_size",
        str(args.vllm_batch_size),
    ]
    if args.vllm_enforce_eager:
        command.append("--vllm_enforce_eager")
    if args.vllm_disable_custom_all_reduce:
        command.append("--vllm_disable_custom_all_reduce")
    if adapter_path is not None:
        command.extend(["--adapter_path", str(adapter_path)])

    env = vllm_subprocess_env(args)
    print("Launching vLLM sampler:", " ".join(command), flush=True)
    print("vLLM CUDA_VISIBLE_DEVICES:", env.get("CUDA_VISIBLE_DEVICES"), flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env, check=True)


def build_checkpoint_state(args, wandb_run, global_step, next_block_idx, current_block_idx=None, resume_epoch=None, resume_batch_start=None, resume_rank_order=None):
    state = {
        "next_block_idx": next_block_idx,
        "global_step": global_step,
        "wandb_id": wandb_run.id if wandb_run is not None else args.wandb_id,
        "args": vars(args),
    }
    if current_block_idx is not None:
        state["current_block_idx"] = current_block_idx
    if resume_epoch is not None:
        state["resume_epoch"] = resume_epoch
    if resume_batch_start is not None:
        state["resume_batch_start"] = resume_batch_start
    if resume_rank_order is not None:
        state["resume_rank_order"] = list(resume_rank_order)
    return state


def encode_buffer_group(tokenizer, rows, device):
    sequences = []
    prompt_lens = []
    rewards = []
    precomputed_logp_refs = []
    has_precomputed_logp_ref = "logp_ref" in rows.columns
    for _, row in rows.iterrows():
        prefix_ids = tokenizer.encode(str(row["prefix_text"]))
        completion_ids = tokenizer.encode(str(row["completion"]), add_special_tokens=False)
        seq = prefix_ids + completion_ids
        sequences.append(seq)
        prompt_lens.append(len(prefix_ids))
        rewards.append(float(row["reward"]))
        if has_precomputed_logp_ref:
            precomputed_logp_refs.append(float(row["logp_ref"]))

    max_len = max(len(seq) for seq in sequences)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padded = []
    attention_masks = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_token_id] * pad_len)
        attention_masks.append([1] * len(seq) + [0] * pad_len)

    precomputed_logp_ref_tensor = None
    if has_precomputed_logp_ref:
        precomputed_logp_ref_tensor = torch.tensor(precomputed_logp_refs, dtype=torch.float32, device=device)

    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        prompt_lens,
        torch.tensor(attention_masks, dtype=torch.long, device=device),
        torch.tensor(rewards, dtype=torch.float32, device=device),
        precomputed_logp_ref_tensor,
    )


def has_complete_logp_ref(buffer_df):
    if "logp_ref" not in buffer_df.columns:
        return False
    values = pd.to_numeric(buffer_df["logp_ref"], errors="coerce")
    return bool(values.notna().all() and np.isfinite(values.to_numpy(dtype=np.float64)).all())


def precompute_buffer_logp_ref(model_name, tokenizer, buffer_df, buffer_path, args, device, rank):
    if has_complete_logp_ref(buffer_df):
        debug_log(f"reference logp already present in {buffer_path}", rank=rank)
        return buffer_df
    if args.disable_precompute_ref_logp:
        debug_log("reference logp precompute disabled; falling back to adapter-disabled training reference", rank=rank)
        return buffer_df

    debug_log(f"precomputing reference logp for {len(buffer_df)} buffer rows", rank=rank)
    model_kwargs = {
        "torch_dtype": parse_torch_dtype(args.torch_dtype),
        "trust_remote_code": True,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    ref_model.eval()

    ref_values = []
    batch_size = max(1, int(args.ref_logp_batch_size))
    chunk_size = args.score_micro_batch_size if args.score_micro_batch_size is not None else batch_size
    try:
        with torch.no_grad():
            row_iter = range(0, len(buffer_df), batch_size)
            if not args.disable_tqdm:
                row_iter = tqdm(row_iter, desc="precompute ref logp")
            for start in row_iter:
                end = min(start + batch_size, len(buffer_df))
                batch_df = buffer_df.iloc[start:end]
                sequences, prompt_lens, attention_masks, _, _ = encode_buffer_group(tokenizer, batch_df, device)
                logp_ref = completion_logprob_chunks(
                    ref_model,
                    sequences,
                    prompt_lens,
                    attention_masks,
                    tokenizer.eos_token_id,
                    max(1, int(chunk_size)),
                )
                sync_cuda_if_available()
                ref_values.extend(float(value) for value in logp_ref.detach().cpu().tolist())
                del sequences, attention_masks, logp_ref
    finally:
        del ref_model
        clear_cuda()

    buffer_df = buffer_df.copy()
    buffer_df["logp_ref"] = ref_values
    buffer_df.to_csv(buffer_path, index=False)
    debug_log(f"wrote reference logp column to {buffer_path}", rank=rank)
    return buffer_df


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
    resume_block_state=None,
    checkpoint_callback=None,
):
    default_order = list(range(len(dataset)))
    excluded_example_indices = {
        int(part.strip())
        for part in str(args.exclude_example_indices or "").split(",")
        if part.strip()
    }
    if excluded_example_indices:
        default_order = [idx for idx in default_order if idx not in excluded_example_indices]
        debug_log(
            f"[block {block_idx}] excluding example indices {sorted(excluded_example_indices)}; train_examples={len(default_order)}",
            rank=rank,
        )
    resume_rank_order = None
    start_epoch = 0
    resume_batch_start = 0
    if resume_block_state is not None and int(resume_block_state.get("current_block_idx", -1)) == block_idx:
        resume_rank_order = resume_block_state.get("resume_rank_order")
        start_epoch = int(resume_block_state.get("resume_epoch", 0))
        resume_batch_start = int(resume_block_state.get("resume_batch_start", 0) or 0)

    for epoch in range(start_epoch, args.epochs):
        if epoch == start_epoch and resume_rank_order is not None:
            rank_order = list(resume_rank_order)
        else:
            rank_order = list(default_order)
            if not args.disable_train_shuffle:
                order_rng = random.Random(args.seed + block_idx * 100000 + epoch)
                order_rng.shuffle(rank_order)
        if world_size > 1:
            remainder = len(rank_order) % world_size
            if remainder:
                rank_order.extend(rank_order[: world_size - remainder])
            rank_order = rank_order[rank::world_size]
        order_dir = Path(args.output_dir) / "debug_logs" / "rank_orders"
        order_dir.mkdir(parents=True, exist_ok=True)
        with (order_dir / f"block{block_idx}_epoch{epoch}_rank{rank}.json").open("w") as handle:
            json.dump(
                {
                    "block_idx": int(block_idx),
                    "epoch": int(epoch),
                    "rank": int(rank),
                    "disable_train_shuffle": bool(args.disable_train_shuffle),
                    "order_seed": None if args.disable_train_shuffle else int(args.seed + block_idx * 100000 + epoch),
                    "rank_order": [int(value) for value in rank_order],
                },
                handle,
                indent=2,
            )
        epoch_batch_start = resume_batch_start if epoch == start_epoch and resume_rank_order is not None else 0

        batch_iter = range(epoch_batch_start, len(rank_order), args.batch_size)
        if not args.disable_tqdm:
            batch_iter = tqdm(batch_iter, desc=f"block {block_idx} epoch {epoch}")
        for start in batch_iter:
            batch_indices = rank_order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            step_id = global_step + 1
            debug_log(
                f"[block {block_idx}] step {step_id} begin epoch={epoch} batch_start={start} batch_size={len(batch_indices)}",
                rank=rank,
            )

            loss_sum = 0.0
            reward_sum = 0.0
            logp_theta_sum = 0.0
            logp_ref_sum = 0.0
            total_sequences = len(batch_indices) * args.completions_per_prefix

            for micro_start in range(0, len(batch_indices), args.micro_batch_size):
                micro_indices = batch_indices[micro_start:micro_start + args.micro_batch_size]
                debug_log(
                    f"[block {block_idx}] step {step_id} micro begin micro_start={micro_start} micro_size={len(micro_indices)}",
                    rank=rank,
                )
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

                sequences, prompt_lens, attention_masks, rewards, precomputed_logp_ref = encode_buffer_group(
                    tokenizer,
                    micro_df,
                    next(unwrap_model(model).parameters()).device,
                )
                if not args.disable_micro_batch_debug_dump:
                    dump_micro_batch_debug(
                        args.output_dir,
                        block_idx,
                        step_id,
                        micro_start,
                        micro_indices,
                        micro_df,
                        prompt_lens,
                        sequences,
                    )
                debug_log(
                    f"[block {block_idx}] step {step_id} micro prepared rows={len(micro_df)} seq_shape={tuple(sequences.shape)} max_prompt_len={max(prompt_lens)} reward_mean={float(rewards.mean().detach().cpu()):.4f}",
                    rank=rank,
                )
                debug_log(f"[block {block_idx}] step {step_id} loss forward begin", rank=rank)
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
                    debug_callback=lambda message: debug_log(
                        f"[block {block_idx}] step {step_id} loss {message}",
                        rank=rank,
                    ),
                    precomputed_logp_ref=precomputed_logp_ref,
                    before_theta_chunk_callback=(
                        None
                        if args.disable_active_forward_debug_dump
                        else lambda chunk_start, chunk_end: dump_active_forward_debug(
                            args.output_dir,
                            block_idx,
                            step_id,
                            micro_start,
                            chunk_start,
                            chunk_end,
                            micro_df,
                            prompt_lens,
                            sequences,
                            attention_masks,
                        )
                    ),
                )
                debug_log(f"[block {block_idx}] step {step_id} loss forward end", rank=rank)
                micro_sequences = len(micro_df)
                debug_log(f"[block {block_idx}] step {step_id} backward begin", rank=rank)
                if not args.disable_active_forward_debug_dump:
                    dump_active_backward_debug(
                        args.output_dir,
                        block_idx,
                        step_id,
                        micro_start,
                        micro_df,
                        prompt_lens,
                        sequences,
                        attention_masks,
                    )
                (loss * (micro_sequences / total_sequences)).backward()
                debug_log(f"[block {block_idx}] step {step_id} backward end", rank=rank)

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

            debug_log(f"[block {block_idx}] step {step_id} optimizer step begin", rank=rank)
            optimizer.step()
            debug_log(f"[block {block_idx}] step {step_id} optimizer step end", rank=rank)
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
            if wandb_run is not None and should_log_wandb_step(args, global_step):
                debug_log(f"[block {block_idx}] wandb step log begin step={global_step}", rank=rank)
                wandb_run.log(record, step=global_step)
                debug_log(f"[block {block_idx}] wandb step log end step={global_step}", rank=rank)
            if checkpoint_callback is not None and args.save_every_steps and global_step % args.save_every_steps == 0:
                checkpoint_callback(
                    current_block_idx=block_idx,
                    global_step=global_step,
                    resume_epoch=epoch,
                    resume_batch_start=start + args.batch_size,
                    resume_rank_order=rank_order,
                )
            if args.max_train_steps and global_step >= args.max_train_steps:
                debug_log(
                    f"[block {block_idx}] reached max_train_steps={args.max_train_steps}; stopping training loop",
                    rank=rank,
                )
                return global_step
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
    parser.add_argument("--future_completions_per_partial", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_every_block", action="store_true")
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="one-step-post-correction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default="allow")
    parser.add_argument("--wandb_log_every", type=int, default=0)
    parser.add_argument("--eval_every_block", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_backend", type=str, default="hf", choices=["hf", "vllm"])
    parser.add_argument("--eval_examples", type=int, default=100)
    parser.add_argument("--eval_max_new_tokens", type=int, default=3072)
    parser.add_argument("--eval_temperature", type=float, default=0.25)
    parser.add_argument("--eval_do_sample", action="store_true")
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_batch_size", type=int, default=8)
    parser.add_argument("--vllm_visible_devices", type=str, default=None)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_custom_all_reduce", action="store_true")
    parser.add_argument("--skip_buffer_sampling", action="store_true")
    parser.add_argument("--debug_dump_timeout_seconds", type=int, default=600)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--disable_micro_batch_debug_dump", action="store_true")
    parser.add_argument("--disable_active_forward_debug_dump", action="store_true")
    parser.add_argument("--disable_train_shuffle", action="store_true")
    parser.add_argument("--exclude_example_indices", type=str, default="")
    parser.add_argument("--disable_precompute_ref_logp", action="store_true")
    parser.add_argument("--ref_logp_batch_size", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    debug_path = setup_debug_file(output_dir, f"trainer_preinit_pid{os.getpid()}.log", args.debug_dump_timeout_seconds)
    debug_log(f"trainer process start argv={' '.join(sys.argv)}")
    debug_log(f"pre-init env CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} LOCAL_RANK={os.environ.get('LOCAL_RANK')} RANK={os.environ.get('RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

    try:
        rank = 0
        world_size = 1
        distributed_device = None
        close_debug_file()
        debug_path = setup_debug_file(output_dir, "trainer.log", args.debug_dump_timeout_seconds)
        debug_log("single-process init complete", rank=rank)
        debug_log(
            f"post-init env CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} LOCAL_RANK=0 RANK=0 WORLD_SIZE=1",
            rank=rank,
        )
        seed_everything(args.seed + rank)

        output_dir.mkdir(parents=True, exist_ok=True)
        debug_log("output dir ready", rank=rank)

        debug_log("loading datasets", rank=rank)
        dataset = load_math_dataset(args.data_path)[: args.max_examples]
        eval_rows = load_math_dataset(args.eval_data_path)[: args.eval_examples] if args.eval_every_block else []
        debug_log(f"loaded datasets train={len(dataset)} eval={len(eval_rows)}", rank=rank)
        model_name = MODEL_NAME_BY_KEY[args.model]
        debug_log(f"loading tokenizer for {model_name}", rank=rank)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        debug_log("tokenizer ready", rank=rank)

        resume_state = None
        start_block_idx = 1
        global_step = 0
        adapter_path = None
        resume_block_state = None
        if args.resume_from_checkpoint:
            debug_log(f"reading resume state from {args.resume_from_checkpoint}", rank=rank)
            checkpoint_dir = Path(args.resume_from_checkpoint)
            adapter_path = checkpoint_dir / "adapter"
            resume_state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)["state"]
            resume_block_idx = resume_state.get("current_block_idx")
            resume_batch_start = int(resume_state.get("resume_batch_start", 0) or 0)
            if resume_block_idx is not None and resume_batch_start > 0:
                start_block_idx = int(resume_block_idx)
                resume_block_state = resume_state
            else:
                start_block_idx = int(resume_state.get("next_block_idx", 1))
            global_step = int(resume_state.get("global_step", 0))
            debug_log(
                f"resume state ready next_block_idx={start_block_idx} global_step={global_step}",
                rank=rank,
            )

        debug_log("initializing wandb state", rank=rank)
        wandb_run = maybe_init_wandb(args, rank, resume_state)
        if rank == 0 and wandb_run is not None and args.wandb_id is None:
            args.wandb_id = wandb_run.id
        if args.eval_only:
            debug_log("startup complete; entering eval-only path", rank=rank)
        else:
            debug_log("startup complete; entering block loop", rank=rank)

        metrics = []
        sample_records = []
        eval_records = []

        if args.eval_only:
            if not args.resume_from_checkpoint:
                raise ValueError("--eval_only requires --resume_from_checkpoint")
            debug_log("[eval-only] loading model from checkpoint adapter", rank=rank)
            if args.eval_backend == "vllm":
                eval_metrics = evaluate_model_with_vllm(
                    model_name,
                    tokenizer,
                    eval_rows,
                    args,
                    adapter_path=adapter_path,
                )
            else:
                model = load_lora_model(
                    model_name,
                    args.torch_dtype,
                    distributed_device,
                    adapter_path,
                    attn_implementation=args.attn_implementation,
                )
                eval_metrics = evaluate_model(model, tokenizer, eval_rows, args)
                del model
                clear_cuda()
            eval_metrics = {
                **eval_metrics,
                "block_idx": start_block_idx - 1,
                "step": global_step,
            }
            eval_records.append(eval_metrics)
            print(eval_metrics, flush=True)
            pd.DataFrame(eval_records).to_csv(output_dir / "eval_metrics.csv", index=False)
            if wandb_run is not None:
                debug_log("[eval-only] wandb eval log begin", rank=rank)
                wandb_run.log(eval_metrics, step=global_step)
                debug_log("[eval-only] wandb eval log end", rank=rank)
            log_point("[eval-only] eval outputs written", rank=rank)
            if wandb_run is not None:
                wandb_run.finish()
            return

        for block_idx in range(start_block_idx, args.num_blocks + 1):
            stage_adapter_path = adapter_path
            if not args.skip_buffer_sampling:
                generate_stage_buffer_subprocess(block_idx, args, stage_adapter_path)
            log_point(f"[block {block_idx}] sampler finished; entering training setup", rank=rank)

            buffer_path = output_dir / "buffers" / f"block_{block_idx}.csv"
            if not buffer_path.exists():
                raise FileNotFoundError(f"Missing sampled buffer for block {block_idx}: {buffer_path}")
            debug_log(f"[block {block_idx}] loading buffer from {buffer_path}", rank=rank)
            buffer_df = pd.read_csv(buffer_path)
            debug_log(f"[block {block_idx}] loaded {len(buffer_df)} buffered samples", rank=rank)
            buffer_df = precompute_buffer_logp_ref(
                model_name,
                tokenizer,
                buffer_df,
                buffer_path,
                args,
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                rank,
            )

            debug_log(f"[block {block_idx}] loading train model", rank=rank)
            model = load_lora_model(
                model_name,
                args.torch_dtype,
                distributed_device,
                stage_adapter_path,
                attn_implementation=args.attn_implementation,
            )
            if args.gradient_checkpointing:
                enable_gradient_checkpointing(model)
            model.train()
            optimizer = torch.optim.AdamW(
                [param for param in model.parameters() if param.requires_grad],
                lr=args.lr,
            )
            if args.resume_from_checkpoint and block_idx == start_block_idx:
                debug_log(
                    f"[block {block_idx}] restoring optimizer and RNG from {args.resume_from_checkpoint}",
                    rank=rank,
                )
                load_checkpoint_state(args.resume_from_checkpoint, optimizer, distributed_device)

            debug_log(f"[block {block_idx}] starting training loop", rank=rank)

            def save_mid_block_checkpoint(current_block_idx, global_step, resume_epoch, resume_batch_start, resume_rank_order):
                checkpoint_state = build_checkpoint_state(
                    args,
                    wandb_run,
                    global_step=global_step,
                    next_block_idx=current_block_idx,
                    current_block_idx=current_block_idx,
                    resume_epoch=resume_epoch,
                    resume_batch_start=resume_batch_start,
                    resume_rank_order=resume_rank_order,
                )
                save_checkpoint(output_dir, model, tokenizer, optimizer, checkpoint_state, distributed=False)
                log_point(
                    f"[block {current_block_idx}] checkpoint_latest updated at step {global_step}",
                    rank=rank,
                )

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
                resume_block_state=resume_block_state if block_idx == start_block_idx else None,
                checkpoint_callback=save_mid_block_checkpoint,
            )
            resume_block_state = None

            if args.save_every_block:
                block_dir = output_dir / f"block_{block_idx}"
                unwrap_model(model).save_pretrained(block_dir)
                tokenizer.save_pretrained(block_dir)
                log_point(f"[block {block_idx}] saved block checkpoint", rank=rank)

            if args.eval_every_block:
                debug_log(f"[block {block_idx}] starting eval on {len(eval_rows)} examples", rank=rank)
                if args.eval_backend == "vllm":
                    eval_metrics = evaluate_model_with_vllm(
                        model_name,
                        tokenizer,
                        eval_rows,
                        args,
                        adapter_path=stage_adapter_path,
                    )
                else:
                    eval_metrics = evaluate_model(model, tokenizer, eval_rows, args)
                eval_metrics = {**eval_metrics, "block_idx": block_idx, "step": global_step}
                eval_records.append(eval_metrics)
                print(eval_metrics, flush=True)
                if wandb_run is not None:
                    debug_log(f"[block {block_idx}] wandb eval log begin", rank=rank)
                    wandb_run.log(eval_metrics, step=global_step)
                    debug_log(f"[block {block_idx}] wandb eval log end", rank=rank)

            checkpoint_state = build_checkpoint_state(
                args,
                wandb_run,
                global_step=global_step,
                next_block_idx=block_idx + 1,
            )
            save_checkpoint(output_dir, model, tokenizer, optimizer, checkpoint_state, distributed=False)
            adapter_path = output_dir / "checkpoint_latest" / "adapter"
            log_point(f"[block {block_idx}] checkpoint_latest updated", rank=rank)

            del model
            del optimizer
            clear_cuda()

        pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
        if args.save_samples:
            pd.DataFrame(sample_records).to_csv(output_dir / "samples.csv", index=False)
        if eval_records:
            pd.DataFrame(eval_records).to_csv(output_dir / "eval_metrics.csv", index=False)

        log_point("[final] outputs written", rank=rank)

        if wandb_run is not None:
            wandb_run.finish()
    except Exception:
        debug_log("unhandled exception follows")
        debug_log(traceback.format_exc().rstrip())
        raise
    finally:
        close_debug_file()


if __name__ == "__main__":
    main()
