#!/usr/bin/env bash
#SBATCH --job-name=ospc-export-fsdp
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=640G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/user/zhongal/One-Step-Post-Correction}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/psamp}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PATH/bin/python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to an FSDP checkpoint directory.}"
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:?Set OUTPUT_MODEL_DIR for the exported Hugging Face model.}"
MODEL="${MODEL:-/data/user/zhongal/.cache/qwen2.5-7b-local}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
LR="${LR:-1e-6}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_PORT="${MASTER_PORT:-$((24000 + SLURM_JOB_ID % 10000))}"
else
  MASTER_PORT="${MASTER_PORT:-29600}"
fi

module purge
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

cd "$REPO_DIR/llm_experiments"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/data/user/zhongal/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TOKENIZERS_PARALLELISM=false

cmd=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nnodes 1
  --nproc_per_node 8
  --master_addr 127.0.0.1
  --master_port "$MASTER_PORT"
  export_fsdp_checkpoint.py
  --checkpoint_dir "$CHECKPOINT_DIR"
  --output_dir "$OUTPUT_MODEL_DIR"
  --model "$MODEL"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --lr "$LR"
)

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
