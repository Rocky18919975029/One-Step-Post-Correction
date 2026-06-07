import argparse
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_training_args(args_list):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path", default="data/MATH500.json")
    parser.add_argument("--output_dir", default="results/blockwise_tb_buffer")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--max_examples", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--completions_per_prefix", type=int, default=4)
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--vllm_dtype", default="bfloat16")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_batch_size", type=int, default=8)
    parser.add_argument("--vllm_visible_devices", default=None)
    parser.add_argument("--vllm_enforce_eager", action="store_true")
    parser.add_argument("--vllm_disable_custom_all_reduce", action="store_true")
    known, _ = parser.parse_known_args(args_list)
    return known


def set_option(args_list, option, value):
    result = []
    idx = 0
    while idx < len(args_list):
        token = args_list[idx]
        if token == option:
            idx += 2
            continue
        if token.startswith(option + "="):
            idx += 1
            continue
        result.append(token)
        idx += 1
    result.extend([option, str(value)])
    return result


def add_flag(args_list, flag):
    return args_list if flag in args_list else [*args_list, flag]


def remove_option(args_list, option):
    result = []
    idx = 0
    while idx < len(args_list):
        token = args_list[idx]
        if token == option:
            idx += 2
            continue
        if token.startswith(option + "="):
            idx += 1
            continue
        result.append(token)
        idx += 1
    return result


def run(command, env):
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env, check=True)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def log_to_file(output_dir, message):
    debug_dir = Path(output_dir) / "debug_logs"
    debug_dir.mkdir(parents=True, exist_ok=True)
    line = f"[debug {datetime.now().isoformat(timespec='seconds')} pid={os.getpid()}] {message}"
    print(line, flush=True)
    with (debug_dir / "staged_launcher.log").open("a", buffering=1) as handle:
        print(line, file=handle, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run sync-buffer TB training as separate vLLM sampling and DDP training stages.")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--master_port", type=int, default=None)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.gpus:
        gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        num_gpus = len(gpus)
    elif args.num_gpus:
        gpus = [str(idx) for idx in range(args.num_gpus)]
        num_gpus = args.num_gpus
    else:
        raise ValueError("Provide either --gpus or --num_gpus.")

    training_args = args.training_args
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    known = parse_training_args(training_args)

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"

    ddp_env = base_env.copy()
    ddp_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    ddp_env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    ddp_env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    vllm_env = base_env.copy()
    if known.vllm_visible_devices:
        vllm_env["CUDA_VISIBLE_DEVICES"] = known.vllm_visible_devices
    elif known.vllm_tensor_parallel_size == 1:
        vllm_env["CUDA_VISIBLE_DEVICES"] = gpus[0]
    else:
        vllm_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:known.vllm_tensor_parallel_size])
    vllm_env["TORCHELASTIC_RESTART_COUNT"] = "0"

    checkpoint = known.resume_from_checkpoint
    start_block = 1
    if checkpoint:
        import torch

        state = torch.load(Path(checkpoint) / "training_state.pt", map_location="cpu", weights_only=False)["state"]
        start_block = int(state.get("next_block_idx", 1))
    log_to_file(known.output_dir, f"launcher start start_block={start_block} num_blocks={known.num_blocks} checkpoint={checkpoint}")

    for block_idx in range(start_block, known.num_blocks + 1):
        log_to_file(known.output_dir, f"stage {block_idx} preparing sampler adapter_path={Path(checkpoint) / 'adapter' if checkpoint else None}")
        print(f"[stage {block_idx}] preparing sampler", flush=True)
        adapter_path = Path(checkpoint) / "adapter" if checkpoint else None
        sampler_cmd = [
            sys.executable,
            "blockwise_vllm_sample_buffer.py",
            "--data_path",
            known.data_path,
            "--output_dir",
            known.output_dir,
            "--model",
            known.model,
            "--block_idx",
            str(block_idx),
            "--max_examples",
            str(known.max_examples),
            "--block_size",
            str(known.block_size),
            "--completions_per_prefix",
            str(known.completions_per_prefix),
            "--max_completion_tokens",
            str(known.max_completion_tokens),
            "--temperature",
            str(known.temperature),
            "--seed",
            str(known.seed),
            "--vllm_dtype",
            known.vllm_dtype,
            "--vllm_tensor_parallel_size",
            str(known.vllm_tensor_parallel_size),
            "--vllm_gpu_memory_utilization",
            str(known.vllm_gpu_memory_utilization),
            "--vllm_max_model_len",
            str(known.vllm_max_model_len),
            "--vllm_batch_size",
            str(known.vllm_batch_size),
        ]
        if adapter_path is not None:
            sampler_cmd.extend(["--adapter_path", str(adapter_path)])
        if known.vllm_enforce_eager:
            sampler_cmd.append("--vllm_enforce_eager")
        if known.vllm_disable_custom_all_reduce:
            sampler_cmd.append("--vllm_disable_custom_all_reduce")
        print("vLLM CUDA_VISIBLE_DEVICES:", vllm_env.get("CUDA_VISIBLE_DEVICES"), flush=True)
        log_to_file(known.output_dir, f"stage {block_idx} sampler env CUDA_VISIBLE_DEVICES={vllm_env.get('CUDA_VISIBLE_DEVICES')}")
        run(sampler_cmd, vllm_env)
        log_to_file(known.output_dir, f"stage {block_idx} sampler complete")
        print(f"[stage {block_idx}] sampler complete; launching DDP training", flush=True)

        block_args = list(training_args)
        block_args = set_option(block_args, "--num_blocks", block_idx)
        block_args = add_flag(block_args, "--skip_buffer_sampling")
        block_args = remove_option(block_args, "--vllm_visible_devices")
        if checkpoint:
            block_args = set_option(block_args, "--resume_from_checkpoint", checkpoint)
        else:
            block_args = remove_option(block_args, "--resume_from_checkpoint")

        master_port = args.master_port or find_free_port()

        train_cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(num_gpus),
            "--master_port",
            str(master_port),
            "blockwise_power_tb_buffer_train.py",
            *block_args,
        ]
        print("DDP CUDA_VISIBLE_DEVICES:", ddp_env.get("CUDA_VISIBLE_DEVICES"), flush=True)
        log_to_file(
            known.output_dir,
            f"stage {block_idx} launching DDP CUDA_VISIBLE_DEVICES={ddp_env.get('CUDA_VISIBLE_DEVICES')} master_port={master_port}",
        )
        run(train_cmd, ddp_env)
        log_to_file(known.output_dir, f"stage {block_idx} DDP training complete")
        print(f"[stage {block_idx}] training complete", flush=True)
        checkpoint = str(Path(known.output_dir) / "checkpoint_latest")


if __name__ == "__main__":
    main()
