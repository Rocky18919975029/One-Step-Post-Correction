import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch block-wise TB training with torchrun/DDP.")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated physical GPU IDs, e.g. 0,1,2,3.")
    parser.add_argument("--master_port", type=int, default=29501)
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(num_gpus),
        "--master_port",
        str(args.master_port),
        "blockwise_power_tb_train.py",
        *training_args,
    ]
    print("Launching:", " ".join(command), flush=True)
    print("CUDA_VISIBLE_DEVICES:", env["CUDA_VISIBLE_DEVICES"], flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env, check=True)


if __name__ == "__main__":
    main()
