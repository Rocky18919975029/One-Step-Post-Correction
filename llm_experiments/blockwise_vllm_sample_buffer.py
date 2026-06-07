import argparse
from pathlib import Path

import transformers

from blockwise_power_tb_buffer_train import generate_stage_buffer
from blockwise_power_tb_train import MODEL_NAME_BY_KEY, load_math_dataset, seed_everything


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
    args = parser.parse_args()

    seed_everything(args.seed)
    model_name = MODEL_NAME_BY_KEY[args.model]
    dataset = load_math_dataset(args.data_path)[: args.max_examples]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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


if __name__ == "__main__":
    main()
