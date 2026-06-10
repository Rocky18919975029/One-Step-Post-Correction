import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def checkpoint_dir(output_dir):
    return Path(output_dir) / "checkpoint_latest"


def checkpoint_adapter_dir(output_dir):
    return checkpoint_dir(output_dir) / "adapter"


def buffer_path(output_dir, block_idx):
    return Path(output_dir) / "buffers" / f"block_{block_idx}.csv"


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


def parse_gpu_list(gpu_spec):
    return [part.strip() for part in str(gpu_spec).split(",") if part.strip()]


def resolve_data_path(path):
    data_path = Path(path)
    if data_path.exists():
        return data_path
    return SCRIPT_DIR / data_path


def count_dataset_examples(path):
    data_path = resolve_data_path(path)
    if data_path.suffix == ".json":
        with data_path.open("r") as handle:
            return len(json.load(handle))
    if data_path.suffix == ".parquet":
        import pandas as pd

        return len(pd.read_parquet(data_path, columns=[]))
    raise ValueError(f"Unsupported dataset format for counting examples: {data_path}")


def is_complete_buffer(output_dir, block_idx, args):
    path = buffer_path(output_dir, block_idx)
    if not path.exists():
        return False

    required_columns = {
        "block_idx",
        "example_idx",
        "sample_idx",
        "question",
        "correct_answer",
        "prefix_token_len",
        "prefix_text",
        "completion_token_len",
        "completion",
        "reward",
    }
    expected_rows = args.max_examples * args.completions_per_prefix
    per_example_counts = {}

    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if not required_columns.issubset(fieldnames):
                return False

            row_count = 0
            for row in reader:
                row_count += 1
                try:
                    example_idx = int(row["example_idx"])
                    sample_idx = int(row["sample_idx"])
                    block_value = int(row["block_idx"])
                except (TypeError, ValueError):
                    return False

                if block_value != block_idx:
                    return False
                if not (0 <= example_idx < args.max_examples):
                    return False
                if not (0 <= sample_idx < args.completions_per_prefix):
                    return False
                per_example_counts.setdefault(example_idx, set()).add(sample_idx)

            if row_count != expected_rows:
                return False
    except Exception:
        return False

    if len(per_example_counts) != args.max_examples:
        return False
    return all(len(sample_ids) == args.completions_per_prefix for sample_ids in per_example_counts.values())


def is_scored_buffer(output_dir, block_idx, args):
    path = buffer_path(output_dir, block_idx)
    if not is_complete_buffer(output_dir, block_idx, args):
        return False
    required_score_columns = {"logp_ref", "logp_theta_score", "log_z_hat", "tb_target"}
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            return required_score_columns.issubset(set(reader.fieldnames or []))
    except Exception:
        return False


def build_sampler_command(args, block_idx, output_dir):
    sampler_tp_size = args.vllm_tensor_parallel_size
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
        str(sampler_tp_size),
        "--vllm_gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm_max_model_len",
        str(args.vllm_max_model_len),
        "--vllm_batch_size",
        str(args.vllm_batch_size),
    ]
    if args.prompt_model is not None:
        command.extend(["--prompt_model", args.prompt_model])
    adapter_dir = checkpoint_adapter_dir(output_dir)
    if block_idx > 1 and adapter_dir.exists():
        command.extend(["--adapter_path", str(adapter_dir)])
    if args.vllm_enforce_eager:
        command.append("--vllm_enforce_eager")
    if args.vllm_disable_custom_all_reduce:
        command.append("--vllm_disable_custom_all_reduce")
    return command


def build_score_command(args, block_idx, output_dir):
    command = [
        sys.executable,
        "blockwise_score_buffer.py",
        "--buffer_path",
        str(buffer_path(output_dir, block_idx)),
        "--model",
        args.model,
        "--torch_dtype",
        "bfloat16",
        "--attn_implementation",
        args.attn_implementation,
        "--score_batch_size",
        str(args.score_batch_size),
        "--completions_per_prefix",
        str(args.completions_per_prefix),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
    ]
    adapter_dir = checkpoint_adapter_dir(output_dir)
    if block_idx > 1 and adapter_dir.exists():
        command.extend(["--adapter_path", str(adapter_dir)])
    return command


def build_train_command(args, block_idx, output_dir):
    if args.accelerate_train and is_scored_buffer(output_dir, block_idx, args):
        adapter_dir = checkpoint_adapter_dir(output_dir)
        trainer_args = [
            sys.executable,
            "blockwise_accelerate_train.py",
            "--buffer_path",
            str(buffer_path(output_dir, block_idx)),
            "--output_dir",
            str(output_dir),
            "--model",
            args.model,
            "--block_idx",
            str(block_idx),
            "--max_examples",
            str(args.max_examples),
            "--batch_size",
            str(args.batch_size),
            "--micro_batch_size",
            str(args.micro_batch_size),
            "--completions_per_prefix",
            str(args.completions_per_prefix),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--attn_implementation",
            args.attn_implementation,
            "--log_every",
            str(args.wandb_log_every if args.wandb_log_every is not None else 1),
        ]
        if block_idx > 1 and adapter_dir.exists():
            trainer_args.extend(["--adapter_path", str(adapter_dir)])
        ckpt_dir = checkpoint_dir(output_dir)
        if ckpt_dir.exists():
            trainer_args.extend(["--resume_from_checkpoint", str(ckpt_dir)])
        if args.gradient_checkpointing:
            trainer_args.append("--gradient_checkpointing")
        if args.save_every_block:
            trainer_args.append("--save_every_block")
        if args.use_wandb:
            trainer_args.append("--use_wandb")
        if args.wandb_project is not None:
            trainer_args.extend(["--wandb_project", args.wandb_project])
        if args.wandb_entity is not None:
            trainer_args.extend(["--wandb_entity", args.wandb_entity])
        if args.wandb_run_name is not None:
            trainer_args.extend(["--wandb_run_name", args.wandb_run_name])
        if args.wandb_id is not None:
            trainer_args.extend(["--wandb_id", args.wandb_id])
        if args.wandb_resume is not None:
            trainer_args.extend(["--wandb_resume", args.wandb_resume])
        if args.wandb_log_every is not None:
            trainer_args.extend(["--wandb_log_every", str(args.wandb_log_every)])
        if not args.ddp_train:
            return trainer_args

        nproc = len(parse_gpu_list(args.train_gpus))
        if nproc < 2:
            raise ValueError("--ddp_train requires at least two --train_gpus entries.")
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(nproc),
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            str(args.train_master_port),
            *trainer_args[1:],
        ]

    trainer_args = [
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
        "--attn_implementation",
        args.attn_implementation,
        "--output_dir",
        str(output_dir),
        "--skip_buffer_sampling",
        "--debug_dump_timeout_seconds",
        str(args.debug_dump_timeout_seconds),
    ]
    if args.prompt_model is not None:
        trainer_args.extend(["--prompt_model", args.prompt_model])
    if args.gradient_checkpointing:
        trainer_args.append("--gradient_checkpointing")
    if args.quiet_debug_logs:
        trainer_args.append("--quiet_debug_logs")
    if args.save_samples:
        trainer_args.append("--save_samples")
    if args.save_every_block:
        trainer_args.append("--save_every_block")
    if args.score_micro_batch_size is not None:
        trainer_args.extend(["--score_micro_batch_size", str(args.score_micro_batch_size)])
    if args.score_chunk_backward:
        trainer_args.append("--score_chunk_backward")
    if args.save_every_steps:
        trainer_args.extend(["--save_every_steps", str(args.save_every_steps)])
    if args.use_wandb:
        trainer_args.append("--use_wandb")
    if args.wandb_project is not None:
        trainer_args.extend(["--wandb_project", args.wandb_project])
    if args.wandb_entity is not None:
        trainer_args.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_run_name is not None:
        trainer_args.extend(["--wandb_run_name", args.wandb_run_name])
    if args.wandb_id is not None:
        trainer_args.extend(["--wandb_id", args.wandb_id])
    if args.wandb_resume is not None:
        trainer_args.extend(["--wandb_resume", args.wandb_resume])
    if args.wandb_log_every is not None:
        trainer_args.extend(["--wandb_log_every", str(args.wandb_log_every)])
    ckpt_dir = checkpoint_dir(output_dir)
    if ckpt_dir.exists():
        trainer_args.extend(["--resume_from_checkpoint", str(ckpt_dir)])

    if not args.ddp_train:
        return trainer_args

    nproc = len(parse_gpu_list(args.train_gpus))
    if nproc < 2:
        raise ValueError("--ddp_train requires at least two --train_gpus entries.")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(nproc),
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(args.train_master_port),
        *trainer_args[1:],
    ]


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
    if args.prompt_model is not None:
        command.extend(["--prompt_model", args.prompt_model])
    if args.vllm_enforce_eager:
        command.append("--vllm_enforce_eager")
    if args.eval_do_sample:
        command.append("--eval_do_sample")
        command.extend(["--eval_temperature", str(args.eval_temperature)])
    if args.use_wandb:
        command.append("--use_wandb")
    if args.quiet_debug_logs:
        command.append("--quiet_debug_logs")
    if args.wandb_project is not None:
        command.extend(["--wandb_project", args.wandb_project])
    if args.wandb_entity is not None:
        command.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_run_name is not None:
        command.extend(["--wandb_run_name", args.wandb_run_name])
    if args.wandb_id is not None:
        command.extend(["--wandb_id", args.wandb_id])
    if args.wandb_resume is not None:
        command.extend(["--wandb_resume", args.wandb_resume])
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
    if args.num_blocks is None:
        args.num_blocks = 3
    if args.eval_examples is None:
        args.eval_examples = 100
    if args.vllm_batch_size is None:
        args.vllm_batch_size = 8


def main():
    parser = argparse.ArgumentParser(
        description="Run the maintained blockwise buffer pipeline with optional DDP training and visible per-step progress."
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--train_gpus", type=str, default=None)
    parser.add_argument("--ddp_train", action="store_true")
    parser.add_argument("--sampler_gpus", type=str, default=None)
    parser.add_argument("--train_master_port", type=int, default=29600)
    parser.add_argument("--sampler_master_port", type=int, default=29700)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/blockwise_buffer_pipeline")
    parser.add_argument("--data_path", type=str, default="../data/train.parquet")
    parser.add_argument("--eval_data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--prompt_model", type=str, default=None, choices=["phi", "qwen", "qwen_math", "qwen_math_grpo", "tulu"])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--score_micro_batch_size", type=int, default=None)
    parser.add_argument("--score_batch_size", type=int, default=1)
    parser.add_argument("--score_chunk_backward", action="store_true")
    parser.add_argument("--accelerate_train", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--completions_per_prefix", type=int, default=4)
    parser.add_argument("--future_completions_per_partial", type=int, default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--save_every_block", action="store_true")
    parser.add_argument("--save_every_steps", type=int, default=0)
    parser.add_argument("--eval_examples", type=int, default=None)
    parser.add_argument("--eval_max_new_tokens", type=int, default=3072)
    parser.add_argument("--eval_backend", type=str, default="vllm", choices=["none", "hf", "vllm"])
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
    parser.add_argument("--quiet_debug_logs", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="one-step-post-correction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default="allow")
    parser.add_argument("--wandb_log_every", type=int, default=0)
    args = parser.parse_args()

    apply_smoke_defaults(args)
    fill_defaults(args)

    if args.num_blocks < 1:
        raise ValueError("--num_blocks must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_examples is None:
        args.max_examples = count_dataset_examples(args.data_path)
        print(f"--max_examples not set; using full dataset count={args.max_examples}", flush=True)

    if args.sampler_gpus is None:
        args.sampler_gpus = args.gpu
    if args.train_gpus is None:
        args.train_gpus = args.gpu
    if args.ddp_train and len(parse_gpu_list(args.train_gpus)) < 2:
        raise ValueError("--ddp_train requires --train_gpus with at least two GPUs, e.g. --train_gpus 0,1.")
    if args.ddp_train and args.save_every_steps:
        raise ValueError("--save_every_steps is not supported with --ddp_train yet; use --save_every_block.")

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    sampler_env = base_env.copy()
    sampler_env["CUDA_VISIBLE_DEVICES"] = args.sampler_gpus
    sampler_env["MASTER_PORT"] = str(args.sampler_master_port)
    sampler_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if len(parse_gpu_list(args.sampler_gpus)) > 1:
        sampler_env.setdefault("NCCL_P2P_DISABLE", "1")
        sampler_env.setdefault("NCCL_IB_DISABLE", "1")

    train_env = base_env.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = args.train_gpus
    train_env["MASTER_PORT"] = str(args.train_master_port)
    if args.ddp_train and len(parse_gpu_list(args.train_gpus)) > 1:
        train_env.setdefault("NCCL_P2P_DISABLE", "1")
        train_env.setdefault("NCCL_IB_DISABLE", "1")

    score_env = base_env.copy()
    score_env["CUDA_VISIBLE_DEVICES"] = parse_gpu_list(args.train_gpus)[0]
    score_env["MASTER_PORT"] = str(args.train_master_port + 2)

    start_block = read_next_block_idx(output_dir)
    if start_block > args.num_blocks:
        if args.eval_backend == "none":
            print(
                f"Checkpoint at {checkpoint_dir(output_dir)} is already past num_blocks={args.num_blocks}; eval skipped.",
                flush=True,
            )
            return
        print(
            f"Checkpoint at {checkpoint_dir(output_dir)} is already past num_blocks={args.num_blocks}; running eval only.",
            flush=True,
        )
        total_steps = 1
        run_step(1, total_steps, "Evaluating latest checkpoint", build_eval_command(args, output_dir), train_env)
        return

    total_steps = (args.num_blocks - start_block + 1) * 3 + 1
    step_idx = 1

    for block_idx in range(start_block, args.num_blocks + 1):
        if is_complete_buffer(output_dir, block_idx, args):
            banner = f"[{step_idx}/{total_steps}] Reusing sampled buffer for block {block_idx}"
            print("\n" + "=" * len(banner), flush=True)
            print(banner, flush=True)
            print("=" * len(banner), flush=True)
            print(f"Found complete buffer at {buffer_path(output_dir, block_idx)}; skipping sampling.", flush=True)
        else:
            run_step(
                step_idx,
                total_steps,
                f"Sampling block {block_idx} with vLLM",
                build_sampler_command(args, block_idx, output_dir),
                sampler_env,
            )
        step_idx += 1

        if is_scored_buffer(output_dir, block_idx, args):
            banner = f"[{step_idx}/{total_steps}] Reusing scored buffer for block {block_idx}"
            print("\n" + "=" * len(banner), flush=True)
            print(banner, flush=True)
            print("=" * len(banner), flush=True)
            print(f"Found scored buffer at {buffer_path(output_dir, block_idx)}; skipping scoring.", flush=True)
        else:
            run_step(
                step_idx,
                total_steps,
                f"Scoring block {block_idx} buffer",
                build_score_command(args, block_idx, output_dir),
                score_env,
            )
        step_idx += 1

        run_step(
            step_idx,
            total_steps,
            f"Training through block {block_idx}",
            build_train_command(args, block_idx, output_dir),
            train_env,
        )
        step_idx += 1

    if args.eval_backend == "none":
        banner = f"[{step_idx}/{total_steps}] Skipping final eval"
        print("\n" + "=" * len(banner), flush=True)
        print(banner, flush=True)
        print("=" * len(banner), flush=True)
        return

    run_step(
        step_idx,
        total_steps,
        "Evaluating latest checkpoint",
        build_eval_command(args, output_dir),
        train_env,
    )


if __name__ == "__main__":
    main()
