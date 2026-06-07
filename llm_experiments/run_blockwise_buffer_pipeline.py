import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def checkpoint_dir(output_dir):
    return Path(output_dir) / "checkpoint_latest"


def checkpoint_adapter_dir(output_dir):
    return checkpoint_dir(output_dir) / "adapter"


def read_next_block_idx(output_dir):
    import torch

    state_path = checkpoint_dir(output_dir) / "training_state.pt"
    if not state_path.exists():
        return 1
    state = torch.load(state_path, map_location="cpu", weights_only=False)["state"]
    return int(state.get("next_block_idx", 1))


def run_step(step_idx, total_steps, title, command, env):
    banner = f"[{step_idx}/{total_steps}] {title}"
    print("\n" + "=" * len(banner), flush=True)
    print(banner, flush=True)
    print("=" * len(banner), flush=True)
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, env=env, check=True)


def build_sampler_command(args, block_idx, output_dir):
    command = [
        sys.executable,
        "blockwise_vllm_sample_buffer.py",
        "--data_path",
        args.data_path,
        "--output_dir",
        str(output_dir),
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
    adapter_dir = checkpoint_adapter_dir(output_dir)
    if block_idx > 1 and adapter_dir.exists():
        command.extend(["--adapter_path", str(adapter_dir)])
    if args.vllm_enforce_eager:
        command.append("--vllm_enforce_eager")
    if args.vllm_disable_custom_all_reduce:
        command.append("--vllm_disable_custom_all_reduce")
    return command


def build_train_command(args, block_idx, output_dir):
    command = [
        sys.executable,
        "blockwise_power_tb_buffer_train.py",
        "--data_path",
        args.data_path,
        "--eval_data_path",
        args.eval_data_path,
        "--model",
        args.model,
        "--max_examples",
        str(args.max_examples),
        "--batch_size",
        str(args.batch_size),
        "--micro_batch_size",
        str(args.micro_batch_size),
        "--epochs",
        str(args.epochs),
        "--num_blocks",
        str(block_idx),
        "--block_size",
        str(args.block_size),
        "--completions_per_prefix",
        str(args.completions_per_prefix),
        "--max_completion_tokens",
        str(args.max_completion_tokens),
        "--temperature",
        str(args.temperature),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(output_dir),
        "--skip_buffer_sampling",
        "--debug_dump_timeout_seconds",
        str(args.debug_dump_timeout_seconds),
    ]
    if args.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if args.save_samples:
        command.append("--save_samples")
    if args.save_every_block:
        command.append("--save_every_block")
    if args.score_micro_batch_size is not None:
        command.extend(["--score_micro_batch_size", str(args.score_micro_batch_size)])
    ckpt_dir = checkpoint_dir(output_dir)
    if ckpt_dir.exists():
        command.extend(["--resume_from_checkpoint", str(ckpt_dir)])
    return command


def build_eval_command(args, output_dir):
    command = [
        sys.executable,
        "blockwise_power_tb_buffer_train.py",
        "--data_path",
        args.data_path,
        "--eval_data_path",
        args.eval_data_path,
        "--model",
        args.model,
        "--eval_only",
        "--eval_backend",
        args.eval_backend,
        "--eval_every_block",
        "--eval_examples",
        str(args.eval_examples),
        "--eval_max_new_tokens",
        str(args.eval_max_new_tokens),
        "--vllm_batch_size",
        str(args.vllm_batch_size),
        "--output_dir",
        str(output_dir),
        "--resume_from_checkpoint",
        str(checkpoint_dir(output_dir)),
        "--debug_dump_timeout_seconds",
        str(args.debug_dump_timeout_seconds),
    ]
    if args.vllm_enforce_eager:
        command.append("--vllm_enforce_eager")
    if args.eval_do_sample:
        command.append("--eval_do_sample")
        command.extend(["--eval_temperature", str(args.eval_temperature)])
    return command


def apply_smoke_defaults(args):
    if not args.smoke:
        return
    if args.max_examples is None:
        args.max_examples = 8
    if args.num_blocks is None:
        args.num_blocks = 2
    if args.eval_examples is None:
        args.eval_examples = 10
    if args.vllm_batch_size is None:
        args.vllm_batch_size = 2
    if args.score_micro_batch_size is None:
        args.score_micro_batch_size = 1
    args.gradient_checkpointing = True
    args.save_samples = True
    args.save_every_block = True
    args.vllm_enforce_eager = True


def fill_defaults(args):
    if args.max_examples is None:
        args.max_examples = 32
    if args.num_blocks is None:
        args.num_blocks = 3
    if args.eval_examples is None:
        args.eval_examples = 100
    if args.vllm_batch_size is None:
        args.vllm_batch_size = 8


def main():
    parser = argparse.ArgumentParser(
        description="Run the maintained single-GPU blockwise buffer pipeline with visible per-step progress."
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/blockwise_buffer_pipeline")
    parser.add_argument("--data_path", type=str, default="../data/train.parquet")
    parser.add_argument("--eval_data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--score_micro_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--completions_per_prefix", type=int, default=4)
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--save_every_block", action="store_true")
    parser.add_argument("--eval_examples", type=int, default=None)
    parser.add_argument("--eval_max_new_tokens", type=int, default=3072)
    parser.add_argument("--eval_backend", type=str, default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--eval_do_sample", action="store_true")
    parser.add_argument("--eval_temperature", type=float, default=0.25)
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_batch_size", type=int, default=None)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_custom_all_reduce", action="store_true")
    parser.add_argument("--debug_dump_timeout_seconds", type=int, default=60)
    args = parser.parse_args()

    apply_smoke_defaults(args)
    fill_defaults(args)

    if args.num_blocks < 1:
        raise ValueError("--num_blocks must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start_block = read_next_block_idx(output_dir)
    if start_block > args.num_blocks:
        print(
            f"Checkpoint at {checkpoint_dir(output_dir)} is already past num_blocks={args.num_blocks}; running eval only.",
            flush=True,
        )
        total_steps = 1
        run_step(1, total_steps, "Evaluating latest checkpoint", build_eval_command(args, output_dir), env)
        return

    total_steps = (args.num_blocks - start_block + 1) * 2 + 1
    step_idx = 1

    for block_idx in range(start_block, args.num_blocks + 1):
        run_step(
            step_idx,
            total_steps,
            f"Sampling block {block_idx} with vLLM",
            build_sampler_command(args, block_idx, output_dir),
            env,
        )
        step_idx += 1

        run_step(
            step_idx,
            total_steps,
            f"Training through block {block_idx}",
            build_train_command(args, block_idx, output_dir),
            env,
        )
        step_idx += 1

    run_step(
        step_idx,
        total_steps,
        "Evaluating latest checkpoint",
        build_eval_command(args, output_dir),
        env,
    )


if __name__ == "__main__":
    main()
