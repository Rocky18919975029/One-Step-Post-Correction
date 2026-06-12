#!/usr/bin/env bash
#SBATCH --job-name=ospc-train-block
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=640G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Train one already-scored block without running sample or score again.
# Useful for diagnostics such as continuing block 2 for extra epochs.

REPO_DIR="${REPO_DIR:-/data/user/zhongal/One-Step-Post-Correction}"
CONDA_ENV="${CONDA_ENV:-psamp}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/psamp}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PATH/bin/python}"
CONDA_SH="${CONDA_SH:-}"
HF_HOME="${HF_HOME:-/data/user/zhongal/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DISABLED="${WANDB_DISABLED:-false}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TRAIN_GPUS="${TRAIN_GPUS:-$GPU_IDS}"

SOURCE_OUTPUT_DIR="${SOURCE_OUTPUT_DIR:?Set SOURCE_OUTPUT_DIR to the scored run to continue from.}"
BLOCK_IDX="${BLOCK_IDX:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOURCE_OUTPUT_DIR}_block${BLOCK_IDX}_extra}"
BUFFER_PATH="${BUFFER_PATH:-$SOURCE_OUTPUT_DIR/buffers/block_${BLOCK_IDX}.csv}"
ADAPTER_PATH="${ADAPTER_PATH:-$SOURCE_OUTPUT_DIR/checkpoint_latest/adapter}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-$SOURCE_OUTPUT_DIR/checkpoint_latest}"

MODEL="${MODEL:-qwen}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
COMPLETIONS_PER_PREFIX="${COMPLETIONS_PER_PREFIX:-4}"
LOSS_LEVEL="${LOSS_LEVEL:-token}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-5}"
ALPHA="${ALPHA:-4.0}"
RATIO_CLIP_EPSILON="${RATIO_CLIP_EPSILON:-0.2}"
SEED="${SEED:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-1}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-one-step-post-correction}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-never}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-0}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
else
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29600}"
fi

module purge
module load miniconda3

if [[ -z "$CONDA_SH" ]]; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
if [[ -d "$CONDA_ENV_PATH" ]]; then
  conda activate "$CONDA_ENV_PATH"
else
  conda activate "$CONDA_ENV"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
hash -r

cd "$REPO_DIR/llm_experiments"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
export CUDA_LAUNCH_BLOCKING
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME
export TRANSFORMERS_CACHE
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
export WANDB_MODE
export WANDB_DISABLED

if [[ ! -f "$BUFFER_PATH" ]]; then
  echo "Missing BUFFER_PATH: $BUFFER_PATH" >&2
  exit 1
fi
if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "Missing ADAPTER_PATH: $ADAPTER_PATH" >&2
  exit 1
fi

IFS=',' read -r -a gpu_array <<< "$TRAIN_GPUS"
nproc="${#gpu_array[@]}"
if [[ "$nproc" -lt 1 ]]; then
  echo "TRAIN_GPUS must contain at least one GPU id." >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN" -m torch.distributed.run
  --nnodes 1
  --nproc_per_node "$nproc"
  --master_addr 127.0.0.1
  --master_port "$TRAIN_MASTER_PORT"
  blockwise_accelerate_train.py
  --buffer_path "$BUFFER_PATH"
  --output_dir "$OUTPUT_DIR"
  --model "$MODEL"
  --adapter_path "$ADAPTER_PATH"
  --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
  --block_idx "$BLOCK_IDX"
  --batch_size "$BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --completions_per_prefix "$COMPLETIONS_PER_PREFIX"
  --epochs "$EPOCHS"
  --lr "$LR"
  --alpha "$ALPHA"
  --ratio_clip_epsilon "$RATIO_CLIP_EPSILON"
  --seed "$SEED"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --loss_level "$LOSS_LEVEL"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --log_every "$LOG_EVERY"
  --save_every_block
  --wandb_project "$WANDB_PROJECT"
  --wandb_resume "$WANDB_RESUME"
  --wandb_log_every "$WANDB_LOG_EVERY"
)

if [[ -n "$MAX_EXAMPLES" ]]; then
  cmd+=(--max_examples "$MAX_EXAMPLES")
fi
case "${GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|on)
    cmd+=(--gradient_checkpointing)
    ;;
esac
case "${USE_WANDB,,}" in
  1|true|yes|on)
    cmd+=(--use_wandb)
    ;;
esac
if [[ -n "$WANDB_ENTITY" ]]; then
  cmd+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_RUN_NAME" ]]; then
  cmd+=(--wandb_run_name "$WANDB_RUN_NAME")
fi
if [[ -n "$WANDB_ID" ]]; then
  cmd+=(--wandb_id "$WANDB_ID")
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
