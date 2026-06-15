import argparse
import shutil
from pathlib import Path

import torch
import transformers

from blockwise_accelerate_train import build_accelerator, load_full_model
from blockwise_power_tb_train import resolve_model_name


def main():
    parser = argparse.ArgumentParser(description="Export an Accelerate FSDP checkpoint as a Hugging Face model.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()
    args.train_backend = "fsdp"
    args.full_finetune = True

    checkpoint_dir = Path(args.checkpoint_dir)
    accelerate_state = checkpoint_dir / "accelerate_state"
    if not accelerate_state.is_dir():
        raise FileNotFoundError(f"Missing FSDP accelerate state: {accelerate_state}")

    output_dir = Path(args.output_dir)
    accelerator = build_accelerator(args, gradient_accumulation_steps=1)
    model_name = resolve_model_name(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_full_model(model_name, args, accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer = accelerator.prepare(model, optimizer)

    accelerator.print(f"Loading FSDP checkpoint from {checkpoint_dir}")
    accelerator.load_state(accelerate_state)
    accelerator.wait_for_everyone()

    accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    full_state_dict = accelerator.get_state_dict(model)
    unwrapped = accelerator.unwrap_model(model)

    tmp_dir = output_dir.with_name(f"{output_dir.name}_tmp")
    if accelerator.is_main_process:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(
            tmp_dir,
            state_dict=full_state_dict,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer.save_pretrained(tmp_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_dir.rename(output_dir)
        print(f"Exported Hugging Face model to {output_dir}", flush=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
