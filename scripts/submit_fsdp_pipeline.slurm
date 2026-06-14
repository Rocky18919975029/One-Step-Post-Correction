#!/usr/bin/env bash
#SBATCH --job-name=ospc-fsdp
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=640G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/user/zhongal/One-Step-Post-Correction}"

export OUTPUT_DIR="${OUTPUT_DIR:-results/fsdp_full_finetune}"
export TRAIN_BACKEND=fsdp
export FULL_FINETUNE=1
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export LR="${LR:-1e-6}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

exec bash "$REPO_DIR/scripts/submit_ddp_pipeline.slurm"
