import argparse
import faulthandler
import gc
import os
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
    evaluate_model,
    load_lora_model,
    load_reference_model,
    load_math_dataset,
    maybe_init_wandb,
    resolve_model_name,
    resolve_prompt_model_key,
    score_completion,
    seed_everything,
)
from power_samp_utils import format_prompt


DEBUG_HANDLE = None
QUIET_DEBUG_LOGS = False
PRECOMPUTED_SCORE_COLUMNS = {"ref_policy", "logp_ref", "logp_theta_score", "log_z_hat", "tb_target"}
PRECOMPUTED_TOKEN_SCORE_COLUMNS = {"token_logp_ref", "token_logp_theta_score", "token_log_z_hat", "token_tb_target"}
PRECOMPUTED_PREFIX_FLOW_COLUMNS = {"ref_policy", "log_v0", "log_vk", "proposal_temperature", "token_logp_ref"}


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


def debug_log(message, rank=None, always=False):
    timestamp = datetime.now().isoformat(timespec="seconds")
    prefix = f"[debug {timestamp} pid={os.getpid()}"
    if rank is not None:
        prefix += f" rank={rank}"
    prefix += "]"
    line = f"{prefix} {message}"
    if always or not QUIET_DEBUG_LOGS:
        print(line, flush=True)
    if DEBUG_HANDLE is not None:
        print(line, file=DEBUG_HANDLE, flush=True)


def log_point(label, rank=0):
    debug_log(label, rank=rank, always=True)


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


def vllm_generate_texts(
    llm,
    sampling_params_cls,
    lora_request,
    prompts,
    args,
    max_tokens=None,
    n=1,
    temperature=None,
    desc="vLLM generate",
):
    max_tokens = args.max_completion_tokens if max_tokens is None else max_tokens
    temperature = args.temperature if temperature is None else temperature
    sampling_params = sampling_params_cls(
        n=n,
        temperature=temperature,
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


def decode_token_slice(tokenizer, token_ids):
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def stage_completion_slices(tokenizer, prompt_text, partial_completion, block_idx, args):
    stage_end = partial_completion_token_limit(block_idx, args)
    partial_ids = tokenizer.encode(partial_completion, add_special_tokens=False)
    if getattr(args, "block_train_mode", "cumulative") == "cumulative":
        return {
            "prefix_text": prompt_text,
            "prefix_token_len": len(tokenizer.encode(prompt_text)),
            "completion": partial_completion,
            "completion_token_len": len(partial_ids),
        }

    stage_start = min(max((block_idx - 1) * args.block_size, 0), stage_end)
    prefix_piece_ids = partial_ids[:stage_start]
    completion_ids = partial_ids[stage_start:stage_end]
    prefix_piece = tokenizer.decode(prefix_piece_ids, skip_special_tokens=False) if prefix_piece_ids else ""
    completion = tokenizer.decode(completion_ids, skip_special_tokens=False) if completion_ids else ""
    prefix_text = prompt_text + prefix_piece
    return {
        "prefix_text": prefix_text,
        "prefix_token_len": len(tokenizer.encode(prefix_text)),
        "completion": completion,
        "completion_token_len": len(completion_ids),
        "stage_start_token": stage_start,
        "stage_end_token": stage_start + len(completion_ids),
        "full_partial_completion": partial_completion,
        "full_partial_completion_token_len": len(partial_ids),
    }


def evaluate_model_with_vllm(model_name, tokenizer, eval_rows, args, adapter_path=None):
    if not eval_rows:
        return {}

    prompt_model = resolve_prompt_model_key(args.model, getattr(args, "prompt_model", None))
    prompts = [format_prompt(row["prompt"], prompt_model, tokenizer, cot=True) for row in eval_rows]
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
    prompt_model = resolve_prompt_model_key(args.model, getattr(args, "prompt_model", None))
    prompts = [format_prompt(row["prompt"], prompt_model, tokenizer, cot=True) for row in dataset]
    if getattr(args, "loss_level", "sequence") == "prefix_flow_token":
        return generate_prefix_flow_stage_buffer(
            model_name,
            tokenizer,
            dataset,
            prompts,
            block_idx,
            args,
            adapter_path,
            output_dir,
            example_idx_offset=example_idx_offset,
            buffer_path_override=buffer_path_override,
        )

    stage_token_limit = partial_completion_token_limit(block_idx, args)
    future_completions_per_partial = (
        args.future_completions_per_partial
        if args.future_completions_per_partial is not None
        else args.completions_per_prefix
    )
    future_temperature = (
        args.future_temperature
        if getattr(args, "future_temperature", None) is not None
        else args.temperature
    )
    if future_temperature == 0.0 and future_completions_per_partial != 1:
        raise ValueError("Greedy future sampling requires --future_completions_per_partial 1.")
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
                stage_slice = stage_completion_slices(
                    tokenizer,
                    prompt_text,
                    partial_completion,
                    block_idx,
                    args,
                )
                future_prompts.append(partial_prefix)
                future_metadata.append(
                    {
                        "example_idx": example_idx,
                        "sample_idx": sample_idx,
                        "question": row["prompt"],
                        "correct_answer": row["answer"],
                        "full_completion_for_reward": partial_completion,
                        **stage_slice,
                    }
                )

        if future_token_budget > 0:
            future_outputs = vllm_generate_texts(
                llm,
                sampling_params_cls,
                lora_request,
                future_prompts,
                args,
                max_tokens=future_token_budget,
                n=future_completions_per_partial,
                temperature=future_temperature,
                desc=f"block {block_idx} future reward generation",
            )
        else:
            future_outputs = [["" for _ in range(future_completions_per_partial)] for _ in future_prompts]
    finally:
        del llm
        clear_cuda()

    records = []
    for meta, futures in zip(future_metadata, future_outputs):
        future_rewards = []
        first_successful_parsed = None
        for future in futures:
            full_completion = meta["full_completion_for_reward"] + future
            future_reward, future_parsed = score_completion(full_completion, meta["correct_answer"])
            future_rewards.append(future_reward)
            if future_reward > 0 and first_successful_parsed is None:
                first_successful_parsed = future_parsed

        reward = 1.0 if any(r > 0 for r in future_rewards) else 0.0
        record = {
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
        for key in [
            "stage_start_token",
            "stage_end_token",
            "full_partial_completion_token_len",
            "full_partial_completion",
        ]:
            if key in meta:
                record[key] = meta[key]
        records.append(record)

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


def generate_prefix_flow_stage_buffer(
    model_name,
    tokenizer,
    dataset,
    prompts,
    block_idx,
    args,
    adapter_path,
    output_dir,
    *,
    example_idx_offset=0,
    buffer_path_override=None,
):
    prefix_token_limit = partial_completion_token_limit(block_idx, args)
    future_token_budget = max(args.max_completion_tokens - prefix_token_limit, 0)
    future_rollouts_per_prefix = (
        args.future_completions_per_partial
        if args.future_completions_per_partial is not None
        else args.completions_per_prefix
    )
    future_temperature = (
        args.future_temperature
        if getattr(args, "future_temperature", None) is not None
        else args.temperature
    )
    if future_temperature == 0.0:
        raise ValueError("prefix_flow_token requires a stochastic future proposal with positive temperature.")

    llm, sampling_params_cls, lora_request = load_vllm(model_name, args, adapter_path)
    try:
        prefix_outputs = vllm_generate_texts(
            llm,
            sampling_params_cls,
            lora_request,
            prompts,
            args,
            max_tokens=prefix_token_limit,
            n=args.completions_per_prefix,
            desc=f"block {block_idx} prefix generation",
        )

        future_prompts = []
        prefix_metadata = []
        for local_example_idx, (row, prompt_text, prefixes) in enumerate(zip(dataset, prompts, prefix_outputs)):
            example_idx = example_idx_offset + local_example_idx
            for sample_idx, prefix_completion in enumerate(prefixes):
                prefix_ids = tokenizer.encode(prefix_completion, add_special_tokens=False)
                if len(prefix_ids) > prefix_token_limit:
                    prefix_ids = prefix_ids[:prefix_token_limit]
                    prefix_completion = decode_token_slice(tokenizer, prefix_ids)
                future_prompts.append(prompt_text + prefix_completion)
                prefix_metadata.append(
                    {
                        "example_idx": example_idx,
                        "sample_idx": sample_idx,
                        "question": row["prompt"],
                        "correct_answer": row["answer"],
                        "prompt_text": prompt_text,
                        "prefix_text": prompt_text,
                        "prefix_token_len": len(tokenizer.encode(prompt_text)),
                        "completion": prefix_completion,
                        "completion_token_len": len(prefix_ids),
                    }
                )

        if future_token_budget > 0:
            future_outputs = vllm_generate_texts(
                llm,
                sampling_params_cls,
                lora_request,
                future_prompts,
                args,
                max_tokens=future_token_budget,
                n=future_rollouts_per_prefix,
                temperature=future_temperature,
                desc=f"block {block_idx} future value rollouts",
            )
        else:
            future_outputs = [["" for _ in range(future_rollouts_per_prefix)] for _ in future_prompts]
    finally:
        del llm
        clear_cuda()

    records = []
    for meta, futures in zip(prefix_metadata, future_outputs):
        for future_idx, future_text in enumerate(futures):
            future_ids = tokenizer.encode(future_text, add_special_tokens=False)
            full_completion = meta["completion"] + future_text
            reward, parsed = score_completion(full_completion, meta["correct_answer"])
            records.append(
                {
                    "block_idx": block_idx,
                    "example_idx": meta["example_idx"],
                    "sample_idx": meta["sample_idx"],
                    "future_idx": future_idx,
                    "question": meta["question"],
                    "correct_answer": meta["correct_answer"],
                    "prefix_token_len": meta["prefix_token_len"],
                    "prefix_text": meta["prefix_text"],
                    "completion_token_len": meta["completion_token_len"],
                    "completion": meta["completion"],
                    "future_text": future_text,
                    "future_token_len": len(future_ids),
                    "full_completion": full_completion,
                    "full_completion_token_len": meta["completion_token_len"] + len(future_ids),
                    "stage_end_token": meta["completion_token_len"],
                    "parsed_answer": parsed if reward > 0 else None,
                    "has_boxed_answer": parsed is not None,
                    "reward": float(reward),
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
    print(f"Saved prefix-flow raw buffer {buffer_path}: {len(records)} rollouts", flush=True)
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


def read_checkpoint_state(checkpoint_dir):
    state_path = Path(checkpoint_dir) / "training_state.pt"
    if not state_path.exists():
        return {}
    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state", {})


def main():
    parser = argparse.ArgumentParser(description="Eval-only helper plus shared buffer generation utilities.")
    parser.add_argument("--data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--eval_data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--output_dir", type=str, default="results/blockwise_tb_buffer")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--prompt_model", type=str, default=None, choices=sorted(MODEL_NAME_BY_KEY))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="one-step-post-correction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default="allow")
    parser.add_argument("--eval_every_block", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--full_model_eval", action="store_true")
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
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_custom_all_reduce", action="store_true")
    parser.add_argument("--debug_dump_timeout_seconds", type=int, default=600)
    parser.add_argument("--quiet_debug_logs", action="store_true")
    args = parser.parse_args()

    if not args.eval_only:
        raise ValueError(
            "This module no longer contains a training entrypoint. "
            "Use run_blockwise_buffer_pipeline.py for sample -> score -> Accelerate/DDP train."
        )
    if not args.resume_from_checkpoint:
        raise ValueError("--eval_only requires --resume_from_checkpoint")

    global QUIET_DEBUG_LOGS
    QUIET_DEBUG_LOGS = bool(args.quiet_debug_logs)

    output_dir = Path(args.output_dir)
    setup_debug_file(output_dir, f"eval_pid{os.getpid()}.log", args.debug_dump_timeout_seconds)
    try:
        seed_everything(args.seed)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = resolve_model_name(args.model)
        prompt_model = resolve_prompt_model_key(args.model, args.prompt_model)
        debug_log(f"eval-only start model_name={model_name} prompt_model={prompt_model}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        eval_rows = load_math_dataset(args.eval_data_path)[: args.eval_examples]
        checkpoint_dir = Path(args.resume_from_checkpoint)
        adapter_path = checkpoint_dir / "adapter"
        resume_state = read_checkpoint_state(checkpoint_dir)
        global_step = int(resume_state.get("global_step", 0) or 0)
        block_idx = int(resume_state.get("next_block_idx", 1) or 1) - 1

        wandb_run = maybe_init_wandb(args, 0, resume_state)
        if args.eval_backend == "vllm":
            eval_metrics = evaluate_model_with_vllm(
                model_name,
                tokenizer,
                eval_rows,
                args,
                adapter_path=None if args.full_model_eval else adapter_path,
            )
        else:
            if args.full_model_eval:
                model = load_reference_model(
                    model_name,
                    args.torch_dtype,
                    device,
                    attn_implementation=args.attn_implementation,
                )
            else:
                model = load_lora_model(
                    model_name,
                    args.torch_dtype,
                    device,
                    adapter_path,
                    attn_implementation=args.attn_implementation,
                )
            eval_metrics = evaluate_model(model, tokenizer, eval_rows, args)
            del model
            clear_cuda()

        eval_metrics = {
            **eval_metrics,
            "block_idx": block_idx,
            "step": global_step,
        }
        print(eval_metrics, flush=True)
        pd.DataFrame([eval_metrics]).to_csv(output_dir / "eval_metrics.csv", index=False)
        if wandb_run is not None:
            wandb_run.log(eval_metrics, step=global_step)
            wandb_run.finish()
        log_point("[eval-only] eval outputs written")
    except Exception:
        debug_log("unhandled exception follows", always=True)
        debug_log(traceback.format_exc().rstrip(), always=True)
        raise
    finally:
        close_debug_file()


if __name__ == "__main__":
    main()
