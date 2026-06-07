import argparse
import os
from datetime import datetime
from pathlib import Path

import transformers

from blockwise_power_tb_buffer_train import generate_stage_buffer
from blockwise_power_tb_train import MODEL_NAME_BY_KEY, load_math_dataset, seed_everything


def debug_log(output_dir, message):
    debug_dir = Path(output_dir) / "debug_logs"
    debug_dir.mkdir(parents=True, exist_ok=True)
    line = f"[debug {datetime.now().isoformat(timespec='seconds')} pid={os.getpid()}] {message}"
    print(line, flush=True)
    with (debug_dir / "vllm_sampler.log").open("a", buffering=1) as handle:
        print(line, file=handle, flush=True)


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
    args = parser.parse_args()

    debug_log(args.output_dir, f"sampler start block_idx={args.block_idx} adapter_path={args.adapter_path} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    seed_everything(args.seed)
    model_name = MODEL_NAME_BY_KEY[args.model]
    debug_log(args.output_dir, f"loading dataset from {args.data_path}")
    dataset = load_math_dataset(args.data_path)[: args.max_examples]
    debug_log(args.output_dir, f"dataset ready count={len(dataset)} model={model_name}")
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
    )
    debug_log(args.output_dir, f"sampler finished block_idx={args.block_idx}")


if __name__ == "__main__":
    main()
