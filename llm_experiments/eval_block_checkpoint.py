import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import transformers

from blockwise_power_tb_buffer_train import evaluate_model_with_vllm
from blockwise_power_tb_train import evaluate_model, load_lora_model, load_math_dataset, resolve_model_name


def resolve_adapter_path(output_dir, block_idx):
    output_dir = Path(output_dir)
    if block_idx == "latest":
        adapter_path = output_dir / "checkpoint_latest" / "adapter"
    else:
        adapter_path = output_dir / f"block_{int(block_idx)}"
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing LoRA adapter at {adapter_path}")
    return adapter_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate one saved block adapter without touching the training job.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--block_idx", type=str, required=True, help="Block number such as 1, 2, 3, or latest.")
    parser.add_argument("--eval_data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--prompt_model", type=str, default=None, choices=["phi", "qwen", "qwen_math", "qwen_math_grpo", "tulu"])
    parser.add_argument("--eval_backend", type=str, default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--eval_examples", type=int, default=500)
    parser.add_argument("--eval_max_new_tokens", type=int, default=3072)
    parser.add_argument("--eval_do_sample", action="store_true")
    parser.add_argument("--eval_temperature", type=float, default=0.25)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_batch_size", type=int, default=32)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_custom_all_reduce", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    adapter_path = resolve_adapter_path(output_dir, args.block_idx)
    model_name = resolve_model_name(args.model)
    eval_rows = load_math_dataset(args.eval_data_path)[: args.eval_examples]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.eval_backend == "vllm":
        metrics = evaluate_model_with_vllm(model_name, tokenizer, eval_rows, args, adapter_path=adapter_path)
    else:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = load_lora_model(
            model_name,
            args.torch_dtype,
            device,
            adapter_path,
            attn_implementation=args.attn_implementation,
        )
        metrics = evaluate_model(model, tokenizer, eval_rows, args)
        del model

    metrics = {
        **metrics,
        "block_idx": args.block_idx,
        "adapter_path": str(adapter_path),
    }
    eval_dir = output_dir / "eval_runs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    safe_block = str(args.block_idx).replace("/", "_")
    json_path = eval_dir / f"eval_block_{safe_block}.json"
    csv_path = eval_dir / f"eval_block_{safe_block}.csv"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(metrics, flush=True)
    print(f"wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
