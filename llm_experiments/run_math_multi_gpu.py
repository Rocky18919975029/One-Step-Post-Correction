import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


NUM_SHARDS = 5


def result_filename(model, mcmc_steps, temperature, batch_idx, seed):
    return (
        f"{model}_math_base_power_samp_results_"
        f"{mcmc_steps}_{temperature}_{batch_idx}_{seed}.csv"
    )


def run_shard(args, gpu, batch_idx, output_dir, log_dir):
    result_path = output_dir / result_filename(
        args.model,
        args.mcmc_steps,
        args.temperature,
        batch_idx,
        args.seed,
    )
    if result_path.exists() and not args.rerun:
        print(f"Shard {batch_idx}: existing result found, skipping", flush=True)
        return batch_idx, gpu, result_path, "skipped"

    log_path = log_dir / f"seed_{args.seed}_shard_{batch_idx}_gpu_{gpu}.log"
    command = [
        sys.executable,
        "power_samp_math.py",
        "--save_str",
        str(args.save_str),
        "--batch_idx",
        str(batch_idx),
        "--mcmc_steps",
        str(args.mcmc_steps),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--model",
        args.model,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"

    print(
        f"Shard {batch_idx}: starting on GPU {gpu}; log: {log_path}",
        flush=True,
    )
    started_at = time.monotonic()
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            time.sleep(args.status_interval)
            elapsed_minutes = (time.monotonic() - started_at) / 60
            print(
                f"Shard {batch_idx}: running on GPU {gpu} "
                f"({elapsed_minutes:.1f} minutes elapsed)",
                flush=True,
            )

    if process.returncode != 0:
        raise RuntimeError(
            f"Shard {batch_idx} failed on GPU {gpu}. See {log_path}"
        )
    if not result_path.exists():
        raise RuntimeError(
            f"Shard {batch_idx} exited successfully but did not create {result_path}"
        )
    return batch_idx, gpu, result_path, "completed"


def run_gpu_queue(args, gpu, batch_indices, output_dir, log_dir):
    results = []
    for batch_idx in batch_indices:
        results.append(run_shard(args, gpu, batch_idx, output_dir, log_dir))
    return results


def merge_results(result_paths, merged_path):
    frames = []
    for batch_idx, result_path in sorted(result_paths.items()):
        frame = pd.read_csv(result_path)
        if len(frame) != 100:
            raise RuntimeError(
                f"Shard {batch_idx} contains {len(frame)} rows instead of 100: {result_path}"
            )
        frame.insert(0, "batch_idx", batch_idx)
        frame.insert(1, "problem_idx", range(batch_idx * 100, batch_idx * 100 + len(frame)))
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != 500:
        raise RuntimeError(f"Merged result contains {len(merged)} rows instead of 500")
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False)
    return len(merged)


def main():
    parser = argparse.ArgumentParser(
        description="Run the five MATH500 shards across multiple GPUs and merge their CSV results."
    )
    parser.add_argument("--seed", type=int, required=True)
    gpu_group = parser.add_mutually_exclusive_group(required=True)
    gpu_group.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated physical GPU IDs, for example: 0,1,2,3",
    )
    gpu_group.add_argument(
        "--num_gpus",
        type=int,
        help="Use physical GPU IDs from 0 through num_gpus - 1.",
    )
    parser.add_argument("--model", type=str, default="qwen_math")
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--save_str", type=Path, default=Path("results"))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--status_interval",
        type=int,
        default=60,
        help="Seconds between progress messages.",
    )
    args = parser.parse_args()

    if args.gpus:
        gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    else:
        gpus = [str(gpu) for gpu in range(args.num_gpus)]
    if not gpus:
        raise ValueError("At least one GPU is required")

    script_dir = Path(__file__).resolve().parent
    if not args.save_str.is_absolute():
        args.save_str = (script_dir / args.save_str).resolve()

    output_dir = args.save_str / args.model
    log_dir = args.save_str / "logs" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    result_paths = {}
    gpu_queues = {
        gpu: list(range(gpu_index, NUM_SHARDS, len(gpus)))
        for gpu_index, gpu in enumerate(gpus)
    }
    print(f"GPU assignment: {gpu_queues}", flush=True)
    with ThreadPoolExecutor(max_workers=min(len(gpus), NUM_SHARDS)) as executor:
        futures = {
            executor.submit(
                run_gpu_queue,
                args,
                gpu,
                batch_indices,
                output_dir,
                log_dir,
            ): gpu
            for gpu, batch_indices in gpu_queues.items()
            if batch_indices
        }

        for future in as_completed(futures):
            for batch_idx, gpu, result_path, status in future.result():
                result_paths[batch_idx] = result_path
                print(f"Shard {batch_idx}: {status} on GPU {gpu} -> {result_path}")

    merged_path = output_dir / (
        f"{args.model}_math500_merged_"
        f"{args.mcmc_steps}_{args.temperature}_seed_{args.seed}.csv"
    )
    row_count = merge_results(result_paths, merged_path)
    print(f"Merged {row_count} rows -> {merged_path}")


if __name__ == "__main__":
    main()
