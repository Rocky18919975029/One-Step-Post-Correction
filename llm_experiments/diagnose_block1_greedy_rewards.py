import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import transformers
from tqdm import tqdm

from blockwise_power_tb_train import resolve_model_name, score_completion


def parse_visible_devices():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def shard_bounds(total_rows, num_shards, shard_idx):
    base = total_rows // num_shards
    remainder = total_rows % num_shards
    start = shard_idx * base + min(shard_idx, remainder)
    end = start + base + (1 if shard_idx < remainder else 0)
    return start, end


def build_anchor_prefix(tokenizer, row, anchor_tokens):
    completion_ids = tokenizer.encode(str(row["completion"]), add_special_tokens=False)
    anchor_ids = completion_ids[: min(anchor_tokens, len(completion_ids))]
    anchor_completion = tokenizer.decode(anchor_ids, skip_special_tokens=False) if anchor_ids else ""
    return str(row["prefix_text"]) + anchor_completion, len(anchor_ids), anchor_completion


def run_worker(args):
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise ImportError("Install vLLM in this environment to run greedy reward diagnostics.") from exc

    model_name = resolve_model_name(args.model)
    df = pd.read_csv(args.buffer_path)
    start, end = shard_bounds(len(df), args.num_shards, args.shard_idx)
    shard_df = df.iloc[start:end].copy()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    anchor_token_lens = []
    anchor_completions = []
    for _, row in shard_df.iterrows():
        prompt, anchor_len, anchor_completion = build_anchor_prefix(tokenizer, row, args.anchor_tokens)
        prompts.append(prompt)
        anchor_token_lens.append(anchor_len)
        anchor_completions.append(anchor_completion)

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype=args.vllm_dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_lora=args.adapter_path is not None,
        max_model_len=args.vllm_max_model_len,
        enforce_eager=args.vllm_enforce_eager,
        disable_custom_all_reduce=args.vllm_disable_custom_all_reduce,
        seed=args.seed,
    )
    lora_request = None if args.adapter_path is None else LoRARequest("adapter", 1, str(args.adapter_path))
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=max(1, args.max_completion_tokens - args.anchor_tokens),
    )

    new_rewards = []
    parsed_answers = []
    greedy_completions = []
    for batch_start in tqdm(
        range(0, len(prompts), args.vllm_batch_size),
        desc=f"greedy reward shard {args.shard_idx}",
    ):
        batch_prompts = prompts[batch_start:batch_start + args.vllm_batch_size]
        outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
        for local_offset, output in enumerate(outputs):
            greedy_future = output.outputs[0].text
            row = shard_df.iloc[batch_start + local_offset]
            full_completion = anchor_completions[batch_start + local_offset] + greedy_future
            reward, parsed = score_completion(full_completion, row["correct_answer"])
            new_rewards.append(float(reward))
            parsed_answers.append(parsed)
            if args.save_greedy_completion:
                greedy_completions.append(greedy_future)

    result = pd.DataFrame(
        {
            "row_idx": list(range(start, end)),
            "source_row_idx": shard_df["source_row_idx"].tolist()
            if "source_row_idx" in shard_df.columns
            else list(range(start, end)),
            "block_idx": shard_df["block_idx"].tolist(),
            "example_idx": shard_df["example_idx"].tolist(),
            "sample_idx": shard_df["sample_idx"].tolist(),
            "original_reward": shard_df["reward"].astype(float).tolist(),
            "block1_greedy_reward": new_rewards,
            "block1_anchor_token_len": anchor_token_lens,
            "block1_greedy_parsed_answer": parsed_answers,
        }
    )
    if args.save_greedy_completion:
        result["block1_greedy_completion"] = greedy_completions

    output_path = Path(args.shard_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


def merge_shards(args, shard_paths):
    frames = [pd.read_csv(path) for path in shard_paths]
    diag = pd.concat(frames, ignore_index=True).sort_values("row_idx").reset_index(drop=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "block2_block1_greedy_rewards.csv"
    summary_path = output_dir / "summary.json"
    matrix_path = output_dir / "reward_2x2_matrix.csv"
    diag.to_csv(detail_path, index=False)

    original = (diag["original_reward"].astype(float) > 0).astype(int)
    greedy = (diag["block1_greedy_reward"].astype(float) > 0).astype(int)
    matrix = pd.crosstab(
        original,
        greedy,
        rownames=["original_reward"],
        colnames=["block1_greedy_reward"],
        dropna=False,
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    matrix.to_csv(matrix_path)

    summary = {
        "rows": int(len(diag)),
        "questions": int(diag["example_idx"].nunique()),
        "sample_num_questions": int(args.sample_num_questions),
        "original_reward_mean": float(original.mean()),
        "block1_greedy_reward_mean": float(greedy.mean()),
        "same_reward_rate": float((original == greedy).mean()),
        "matrix": {
            "original0_greedy0": int(matrix.loc[0, 0]),
            "original0_greedy1": int(matrix.loc[0, 1]),
            "original1_greedy0": int(matrix.loc[1, 0]),
            "original1_greedy1": int(matrix.loc[1, 1]),
        },
        "detail_csv": str(detail_path),
        "matrix_csv": str(matrix_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print("\n2x2 matrix: rows=original reward, cols=block1-greedy reward", flush=True)
    print(matrix.to_string(), flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {matrix_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


def run_controller(args):
    gpu_ids = parse_visible_devices()
    if len(gpu_ids) < args.num_shards:
        raise RuntimeError(f"Need {args.num_shards} visible GPUs, found {gpu_ids}.")

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    selected_buffer_path = args.buffer_path
    if args.sample_num_questions > 0:
        full_df = pd.read_csv(args.buffer_path)
        unique_examples = sorted(full_df["example_idx"].unique())
        if args.sample_num_questions > len(unique_examples):
            raise ValueError(
                f"Requested {args.sample_num_questions} questions, but buffer only has {len(unique_examples)}."
            )
        sampled_examples = (
            pd.Series(unique_examples)
            .sample(n=args.sample_num_questions, random_state=args.seed)
            .sort_values()
            .tolist()
        )
        selected_df = full_df[full_df["example_idx"].isin(sampled_examples)].copy()
        selected_df.insert(0, "source_row_idx", selected_df.index)
        selected_buffer_path = str(output_dir / f"sampled_{args.sample_num_questions}_questions_seed{args.seed}.csv")
        selected_df.to_csv(selected_buffer_path, index=False)
        print(
            f"Sampled {args.sample_num_questions} questions -> {len(selected_df)} rows: {selected_buffer_path}",
            flush=True,
        )

    processes = []
    shard_paths = []
    for shard_idx in range(args.num_shards):
        shard_path = shard_dir / f"shard_{shard_idx}.csv"
        shard_paths.append(shard_path)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--buffer_path", selected_buffer_path,
            "--output_dir", args.output_dir,
            "--model", args.model,
            "--anchor_tokens", str(args.anchor_tokens),
            "--max_completion_tokens", str(args.max_completion_tokens),
            "--vllm_dtype", args.vllm_dtype,
            "--vllm_gpu_memory_utilization", str(args.vllm_gpu_memory_utilization),
            "--vllm_max_model_len", str(args.vllm_max_model_len),
            "--vllm_batch_size", str(args.vllm_batch_size),
            "--seed", str(args.seed),
            "--num_shards", str(args.num_shards),
            "--sample_num_questions", "0",
            "--shard_idx", str(shard_idx),
            "--shard_output", str(shard_path),
        ]
        if args.adapter_path is not None:
            command.extend(["--adapter_path", args.adapter_path])
        if args.vllm_enforce_eager:
            command.append("--vllm_enforce_eager")
        if args.vllm_disable_custom_all_reduce:
            command.append("--vllm_disable_custom_all_reduce")
        if args.save_greedy_completion:
            command.append("--save_greedy_completion")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[shard_idx]
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        print(f"Launching shard {shard_idx} on GPU {gpu_ids[shard_idx]} -> {shard_path}", flush=True)
        processes.append(subprocess.Popen(command, env=env))

    failures = []
    for shard_idx, process in enumerate(processes):
        return_code = process.wait()
        if return_code != 0:
            failures.append((shard_idx, return_code))
    if failures:
        raise RuntimeError(f"Greedy reward diagnostic worker failures: {failures}")

    merge_shards(args, shard_paths)
    shutil.rmtree(shard_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For every block-2 sample, start from the block-1 token boundary, "
            "greedy decode a fresh continuation, and compare the resulting reward "
            "against the reward stored in the buffer."
        )
    )
    parser.add_argument("--buffer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--anchor_tokens", type=int, default=192)
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--vllm_dtype", default="bfloat16")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_batch_size", type=int, default=32)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_custom_all_reduce", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--sample_num_questions", type=int, default=0)
    parser.add_argument("--save_greedy_completion", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard_idx", type=int, default=None)
    parser.add_argument("--shard_output", default=None)
    args = parser.parse_args()

    if args.worker:
        if args.shard_idx is None or args.shard_output is None:
            raise ValueError("Worker mode requires --shard_idx and --shard_output.")
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
