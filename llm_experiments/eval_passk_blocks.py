import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import transformers
from tqdm import tqdm

from blockwise_power_tb_train import (
    load_math_dataset,
    resolve_model_name,
    resolve_prompt_model_key,
    score_completion,
)
from power_samp_utils import format_prompt


def parse_blocks(value):
    blocks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        blocks.append(item if item == "base" else str(int(item)))
    if not blocks:
        raise ValueError("--blocks must include at least one block.")
    return blocks


def parse_visible_devices():
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [item.strip() for item in value.split(",") if item.strip()]


def shard_bounds(total, num_shards, shard_idx):
    base = total // num_shards
    remainder = total % num_shards
    start = shard_idx * base + min(shard_idx, remainder)
    end = start + base + (1 if shard_idx < remainder else 0)
    return start, end


def resolve_adapter_path(output_dir, block_idx):
    if block_idx == "base":
        return None
    adapter_path = Path(output_dir) / f"block_{int(block_idx)}"
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing LoRA adapter at {adapter_path}")
    return adapter_path


def pass_at_k(num_samples, num_correct, k):
    if k <= 0 or k > num_samples:
        raise ValueError(f"k must be in [1, {num_samples}], got {k}")
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    failure_probability = 1.0
    for offset in range(k):
        failure_probability *= (num_samples - num_correct - offset) / (num_samples - offset)
    return 1.0 - failure_probability


def run_worker(args):
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise ImportError("Install vLLM in the evaluation environment.") from exc

    model_name = resolve_model_name(args.model)
    prompt_model = resolve_prompt_model_key(args.model, args.prompt_model)
    rows = load_math_dataset(args.eval_data_path)[: args.eval_examples]
    start, end = shard_bounds(len(rows), args.num_shards, args.shard_idx)
    rows = rows[start:end]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [format_prompt(row["prompt"], prompt_model, tokenizer, cot=True) for row in rows]

    adapter_path = resolve_adapter_path(args.output_dir, args.block_idx)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype=args.vllm_dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_lora=adapter_path is not None,
        max_model_len=args.vllm_max_model_len,
        enforce_eager=args.vllm_enforce_eager,
        disable_custom_all_reduce=args.vllm_disable_custom_all_reduce,
        seed=args.seed,
    )
    lora_request = None if adapter_path is None else LoRARequest("adapter", 1, str(adapter_path))
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    output_path = Path(args.shard_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for batch_start in tqdm(
            range(0, len(prompts), args.prompt_batch_size),
            desc=f"{args.block_idx} shard {args.shard_idx}",
        ):
            batch_prompts = prompts[batch_start:batch_start + args.prompt_batch_size]
            batch_rows = rows[batch_start:batch_start + args.prompt_batch_size]
            outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
            for local_idx, (row, output) in enumerate(zip(batch_rows, outputs)):
                completions = [choice.text for choice in output.outputs]
                if len(completions) != args.num_samples:
                    raise RuntimeError(
                        f"Expected {args.num_samples} responses, got {len(completions)} "
                        f"for example {start + batch_start + local_idx}."
                    )
                correctness = []
                for completion in completions:
                    reward, _ = score_completion(completion, row["answer"])
                    correctness.append(1 if reward > 0 else 0)
                record = {
                    "example_idx": start + batch_start + local_idx,
                    "correct_count": int(sum(correctness)),
                    "num_samples": args.num_samples,
                    "correctness": "".join(str(value) for value in correctness),
                }
                if args.save_responses:
                    record["responses"] = completions
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()


def merge_block_results(args, block_idx, shard_paths, block_dir):
    records = []
    for shard_path in shard_paths:
        with shard_path.open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(key=lambda row: row["example_idx"])
    if len(records) != args.eval_examples:
        raise RuntimeError(f"Merged {len(records)} questions, expected {args.eval_examples}.")
    if [row["example_idx"] for row in records] != list(range(args.eval_examples)):
        raise RuntimeError("Merged shards do not cover every evaluation example exactly once.")

    per_question_path = block_dir / "per_question.jsonl"
    with per_question_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "block_idx": block_idx,
        "examples": args.eval_examples,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pass_at_k_estimator": "1-C(n-c,k)/C(n,k)",
        "sample_accuracy": sum(row["correct_count"] for row in records)
        / (args.eval_examples * args.num_samples),
        "any_correct_rate": sum(row["correct_count"] > 0 for row in records) / args.eval_examples,
    }
    for k in args.k_values:
        metrics[f"pass@{k}"] = sum(
            pass_at_k(args.num_samples, row["correct_count"], k) for row in records
        ) / args.eval_examples

    metrics_path = block_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    return metrics


def run_controller(args):
    gpu_ids = parse_visible_devices()
    if args.slurm_srun:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("--slurm_srun requires a running Slurm allocation.")
        if args.num_shards != args.slurm_nodes * args.shards_per_node:
            raise ValueError("--num_shards must equal --slurm_nodes * --shards_per_node.")
    elif len(gpu_ids) < args.num_shards:
        raise RuntimeError(f"Need {args.num_shards} visible GPUs, found {gpu_ids}.")
    if args.num_samples < max(args.k_values):
        raise ValueError("--num_samples must be at least the largest requested k.")

    root = Path(args.output_dir) / "passk_eval" / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for block_idx in args.blocks:
        block_dir = root / f"block_{block_idx}"
        metrics_path = block_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            metrics = json.loads(metrics_path.read_text())
            all_metrics.append(metrics)
            print(f"Reusing completed pass@k evaluation for block {block_idx}", flush=True)
            continue

        resolve_adapter_path(args.output_dir, block_idx)
        shard_dir = block_dir / "shards"
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = []
        print(f"Evaluating block {block_idx} on {args.num_shards} GPUs", flush=True)
        for shard_idx in range(args.num_shards):
            shard_path = shard_dir / f"shard_{shard_idx}.jsonl"
            shard_paths.append(shard_path)

        worker_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--output_dir", args.output_dir,
            "--block_idx", block_idx,
            "--eval_data_path", args.eval_data_path,
            "--model", args.model,
            "--prompt_model", args.prompt_model,
            "--eval_examples", str(args.eval_examples),
            "--num_samples", str(args.num_samples),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--max_new_tokens", str(args.max_new_tokens),
            "--prompt_batch_size", str(args.prompt_batch_size),
            "--vllm_dtype", args.vllm_dtype,
            "--vllm_gpu_memory_utilization", str(args.vllm_gpu_memory_utilization),
            "--vllm_max_model_len", str(args.vllm_max_model_len),
            "--seed", str(args.seed),
            "--num_shards", str(args.num_shards),
        ]
        if args.vllm_enforce_eager:
            worker_command.append("--vllm_enforce_eager")
        if args.vllm_disable_custom_all_reduce:
            worker_command.append("--vllm_disable_custom_all_reduce")
        if args.save_responses:
            worker_command.append("--save_responses")

        if args.slurm_srun:
            command = [
                "srun",
                "--nodes", str(args.slurm_nodes),
                "--ntasks", str(args.num_shards),
                "--ntasks-per-node", str(args.shards_per_node),
                "--gpus-per-task", "1",
                "--cpus-per-task", str(args.cpus_per_worker),
                "--gpu-bind", "single:1",
                "--kill-on-bad-exit=1",
                *worker_command,
                "--shard_output", str(shard_dir / "shard_{shard_idx}.jsonl"),
            ]
            subprocess.run(command, check=True)
        else:
            processes = []
            for shard_idx in range(args.num_shards):
                command = [
                    *worker_command,
                    "--shard_idx", str(shard_idx),
                    "--shard_output", str(shard_paths[shard_idx]),
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[shard_idx]
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
                processes.append(subprocess.Popen(command, env=env))

            failures = []
            for shard_idx, process in enumerate(processes):
                return_code = process.wait()
                if return_code != 0:
                    failures.append((shard_idx, return_code))
            if failures:
                raise RuntimeError(f"Pass@k worker failures for block {block_idx}: {failures}")

        metrics = merge_block_results(args, block_idx, shard_paths, block_dir)
        all_metrics.append(metrics)
        shutil.rmtree(shard_dir, ignore_errors=True)
        print(metrics, flush=True)

    json_path = root / "summary.json"
    csv_path = root / "summary.csv"
    json_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False) + "\n")
    import pandas as pd
    pd.DataFrame(all_metrics).to_csv(csv_path, index=False)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k for base and saved block adapters.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--blocks", default="base,1,2,3,4,5,6")
    parser.add_argument("--block_idx", default=None)
    parser.add_argument("--eval_data_path", default="data/MATH500.json")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--prompt_model", default="qwen")
    parser.add_argument("--eval_examples", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--k_values", default="1,4,16,32,64,128")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--prompt_batch_size", type=int, default=4)
    parser.add_argument("--vllm_dtype", default="bfloat16")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_custom_all_reduce", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--slurm_srun", action="store_true")
    parser.add_argument("--slurm_nodes", type=int, default=1)
    parser.add_argument("--shards_per_node", type=int, default=8)
    parser.add_argument("--cpus_per_worker", type=int, default=12)
    parser.add_argument("--run_name", default="temp0.6_top_p0.95_n128")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save_responses", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard_idx", type=int, default=None)
    parser.add_argument("--shard_output", default=None)
    args = parser.parse_args()
    args.blocks = parse_blocks(args.blocks)
    args.k_values = [int(item) for item in args.k_values.split(",") if item.strip()]
    if args.worker:
        if args.shard_idx is None:
            task_rank = os.environ.get("SLURM_PROCID")
            if task_rank is not None:
                args.shard_idx = int(task_rank)
        if args.shard_output is not None and args.shard_idx is not None:
            args.shard_output = args.shard_output.format(shard_idx=args.shard_idx)
        if args.block_idx is None or args.shard_idx is None or args.shard_output is None:
            raise ValueError("Worker mode requires a block, shard index, and shard output path.")
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
