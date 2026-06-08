import argparse
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import transformers

from blockwise_power_tb_buffer_train import generate_stage_buffer
from blockwise_power_tb_train import MODEL_NAME_BY_KEY, load_math_dataset


def debug_log(output_dir, message):
    debug_dir = Path(output_dir) / "debug_logs"
    debug_dir.mkdir(parents=True, exist_ok=True)
    line = f"[debug {datetime.now().isoformat(timespec='seconds')} pid={os.getpid()}] {message}"
    print(line, flush=True)
    with (debug_dir / "vllm_sampler.log").open("a", buffering=1) as handle:
        print(line, file=handle, flush=True)


def seed_for_vllm(seed):
    random.seed(seed)
    np.random.seed(seed)


def parse_visible_devices():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def shard_ranges(total_examples, num_shards):
    shard_count = max(1, min(num_shards, total_examples))
    base = total_examples // shard_count
    remainder = total_examples % shard_count
    start = 0
    ranges = []
    for shard_idx in range(shard_count):
        shard_size = base + (1 if shard_idx < remainder else 0)
        end = start + shard_size
        if start < end:
            ranges.append((shard_idx, start, end))
        start = end
    return ranges


def build_worker_command(args, start_idx, end_idx, output_path):
    command = [
        sys.executable,
        Path(__file__).name,
        "--data_path",
        args.data_path,
        "--output_dir",
        args.output_dir,
        "--model",
        args.model,
        "--block_idx",
        str(args.block_idx),
        "--max_examples",
        str(args.max_examples),
        "--block_size",
        str(args.block_size),
        "--completions_per_prefix",
        str(args.completions_per_prefix),
        "--future_completions_per_partial",
        str(args.future_completions_per_partial if args.future_completions_per_partial is not None else args.completions_per_prefix),
        "--max_completion_tokens",
        str(args.max_completion_tokens),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--vllm_dtype",
        args.vllm_dtype,
        "--vllm_tensor_parallel_size",
        "1",
        "--vllm_gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm_max_model_len",
        str(args.vllm_max_model_len),
        "--vllm_batch_size",
        str(args.vllm_batch_size),
        "--example_start",
        str(start_idx),
        "--example_end",
        str(end_idx),
        "--buffer_output_path",
        str(output_path),
        "--worker_mode",
    ]
    if args.adapter_path:
        command.extend(["--adapter_path", args.adapter_path])
    if args.vllm_enforce_eager:
        command.append("--vllm_enforce_eager")
    if args.vllm_disable_custom_all_reduce:
        command.append("--vllm_disable_custom_all_reduce")
    return command


def merge_worker_outputs(output_paths, final_path):
    frames = [pd.read_csv(path) for path in output_paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["example_idx", "sample_idx"]).reset_index(drop=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(final_path, index=False)


def run_sharded_sampling(args):
    visible_devices = parse_visible_devices()
    ranges = shard_ranges(args.max_examples, len(visible_devices))
    debug_log(
        args.output_dir,
        f"launching sharded sampler workers block_idx={args.block_idx} visible_devices={visible_devices} shard_ranges={ranges}",
    )

    buffer_dir = Path(args.output_dir) / "buffers"
    temp_dir = buffer_dir / f".tmp_block_{args.block_idx}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    output_paths = []
    for shard_idx, (gpu_id, (_, start_idx, end_idx)) in enumerate(zip(visible_devices, ranges)):
        output_path = temp_dir / f"shard_{shard_idx}.csv"
        output_paths.append(output_path)
        command = build_worker_command(args, start_idx, end_idx, output_path)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        debug_log(
            args.output_dir,
            f"launching worker shard={shard_idx} gpu={gpu_id} example_range=[{start_idx},{end_idx}) output={output_path}",
        )
        processes.append(
            subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parent,
                env=env,
            )
        )

    failures = []
    for shard_idx, process in enumerate(processes):
        return_code = process.wait()
        if return_code != 0:
            failures.append((shard_idx, return_code))

    if failures:
        raise subprocess.CalledProcessError(
            failures[0][1],
            f"sharded sampler worker {failures[0][0]} failed; all failures={failures}",
        )

    final_path = Path(args.output_dir) / "buffers" / f"block_{args.block_idx}.csv"
    merge_worker_outputs(output_paths, final_path)
    for output_path in output_paths:
        output_path.unlink(missing_ok=True)
    temp_dir.rmdir()
    debug_log(args.output_dir, f"merged {len(output_paths)} worker shards into {final_path}")
    print(f"Saved buffer {final_path}: {args.max_examples * args.completions_per_prefix} samples", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Sample one block-wise TB buffer with vLLM.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=sorted(MODEL_NAME_BY_KEY))
    parser.add_argument("--block_idx", type=int, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--completions_per_prefix", type=int, default=4)
    parser.add_argument("--future_completions_per_partial", type=int, default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_batch_size", type=int, default=8)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_custom_all_reduce", action="store_true")
    parser.add_argument("--example_start", type=int, default=0)
    parser.add_argument("--example_end", type=int, default=None)
    parser.add_argument("--buffer_output_path", type=str, default=None)
    parser.add_argument("--worker_mode", action="store_true")
    args = parser.parse_args()

    debug_log(args.output_dir, f"sampler start block_idx={args.block_idx} adapter_path={args.adapter_path} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    seed_for_vllm(args.seed)
    visible_devices = parse_visible_devices()
    if not args.worker_mode and len(visible_devices) > 1:
        run_sharded_sampling(args)
        debug_log(args.output_dir, f"sampler finished block_idx={args.block_idx}")
        return

    model_name = MODEL_NAME_BY_KEY[args.model]
    debug_log(args.output_dir, f"loading dataset from {args.data_path}")
    dataset = load_math_dataset(args.data_path)[: args.max_examples]
    example_end = len(dataset) if args.example_end is None else min(args.example_end, len(dataset))
    example_start = min(args.example_start, example_end)
    dataset = dataset[example_start:example_end]
    debug_log(
        args.output_dir,
        f"dataset ready count={len(dataset)} model={model_name} example_range=[{example_start},{example_end}) worker_mode={args.worker_mode}",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    debug_log(args.output_dir, "tokenizer ready; starting stage buffer generation")

    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    generate_stage_buffer(
        model_name,
        tokenizer,
        dataset,
        args.block_idx,
        args,
        adapter_path,
        Path(args.output_dir),
        example_idx_offset=example_start,
        buffer_path_override=args.buffer_output_path,
    )
    debug_log(args.output_dir, f"sampler finished block_idx={args.block_idx}")


if __name__ == "__main__":
    main()
