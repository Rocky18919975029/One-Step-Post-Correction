#!/usr/bin/env bash
#SBATCH --job-name=ospc-ddp
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

# Usage:
#   sbatch scripts/submit_ddp_pipeline.slurm
#
# Common overrides:
#   sbatch --export=ALL,NUM_BLOCKS=16,MAX_EXAMPLES=500,OUTPUT_DIR=results/full_ddp scripts/submit_ddp_pipeline.slurm
#
# Tuned for the chess Slurm cluster, acd_u partition, 1 node with 8 NVIDIA
# H100 80GB GPUs. The account is intentionally omitted so Slurm can use the
# user's default enabled account.
# GPU compute nodes are assumed to have no internet access. Make sure the model
# and datasets are already present in the local Hugging Face/data caches.
# Gradient checkpointing defaults on for full H100 runs because
# completions_per_prefix=8 can exceed 80GB during training. It should be paired
# with eager attention on this cluster.
# Attention defaults to eager because SDPA has triggered rank-local SIGSEGVs on
# this cluster under DDP training.

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
SAMPLER_GPUS="${SAMPLER_GPUS:-$GPU_IDS}"

OUTPUT_DIR="${OUTPUT_DIR:-results/ddp_full_run}"
DATA_PATH="${DATA_PATH:-../data/train.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/MATH500.json}"
MODEL="${MODEL:-qwen}"
PROMPT_MODEL="${PROMPT_MODEL:-qwen}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-one-step-post-correction}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_ID="${WANDB_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-0}"

MAX_EXAMPLES="${MAX_EXAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SCORE_MICRO_BATCH_SIZE="${SCORE_MICRO_BATCH_SIZE:-1}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1}"
SCORE_NUM_WORKERS="${SCORE_NUM_WORKERS:-}"
SCORE_CHUNK_BACKWARD="${SCORE_CHUNK_BACKWARD:-0}"
ACCELERATE_TRAIN="${ACCELERATE_TRAIN:-1}"
LOSS_LEVEL="${LOSS_LEVEL:-sequence}"
EPOCHS="${EPOCHS:-1}"
NUM_BLOCKS="${NUM_BLOCKS:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-192}"
COMPLETIONS_PER_PREFIX="${COMPLETIONS_PER_PREFIX:-4}"
FUTURE_COMPLETIONS_PER_PARTIAL="${FUTURE_COMPLETIONS_PER_PARTIAL:-4}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-3072}"
TEMPERATURE="${TEMPERATURE:-0.25}"
ALPHA="${ALPHA:-4.0}"
BETA="${BETA:-1.0}"
LR="${LR:-1e-5}"
SEED="${SEED:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-500}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-3072}"

VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-8}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
else
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29600}"
fi
SAMPLER_MASTER_PORT="${SAMPLER_MASTER_PORT:-$((TRAIN_MASTER_PORT + 1))}"

mkdir -p logs

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

echo "Python executable: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
import peft
import torch

print(f"sys.executable={sys.executable}", flush=True)
print(f"peft={peft.__version__}", flush=True)
print(f"torch={torch.__version__}", flush=True)
PY
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"

cmd=(
  "$PYTHON_BIN" run_blockwise_buffer_pipeline.py
  --gpu 0
  --train_gpus "$TRAIN_GPUS"
  --sampler_gpus "$SAMPLER_GPUS"
  --train_master_port "$TRAIN_MASTER_PORT"
  --sampler_master_port "$SAMPLER_MASTER_PORT"
  --ddp_train
  --output_dir "$OUTPUT_DIR"
  --data_path "$DATA_PATH"
  --eval_data_path "$EVAL_DATA_PATH"
  --model "$MODEL"
  --prompt_model "$PROMPT_MODEL"
  --batch_size "$BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --score_micro_batch_size "$SCORE_MICRO_BATCH_SIZE"
  --score_batch_size "$SCORE_BATCH_SIZE"
  --loss_level "$LOSS_LEVEL"
  --epochs "$EPOCHS"
  --num_blocks "$NUM_BLOCKS"
  --block_size "$BLOCK_SIZE"
  --completions_per_prefix "$COMPLETIONS_PER_PREFIX"
  --future_completions_per_partial "$FUTURE_COMPLETIONS_PER_PARTIAL"
  --max_completion_tokens "$MAX_COMPLETION_TOKENS"
  --temperature "$TEMPERATURE"
  --alpha "$ALPHA"
  --beta "$BETA"
  --lr "$LR"
  --seed "$SEED"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --save_samples
  --save_every_block
  --eval_backend "$EVAL_BACKEND"
  --eval_examples "$EVAL_EXAMPLES"
  --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS"
  --vllm_dtype "$VLLM_DTYPE"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
  --vllm_batch_size "$VLLM_BATCH_SIZE"
  --quiet_debug_logs
)

case "${VLLM_ENFORCE_EAGER,,}" in
  1|true|yes|on)
    cmd+=(--vllm_enforce_eager)
    ;;
esac

case "${GRADIENT_CHECKPOINTING,,}" in
  1|true|yes|on)
    cmd+=(--gradient_checkpointing)
    ;;
esac

case "${SCORE_CHUNK_BACKWARD,,}" in
  1|true|yes|on)
    cmd+=(--score_chunk_backward)
    ;;
esac

case "${ACCELERATE_TRAIN,,}" in
  1|true|yes|on)
    cmd+=(--accelerate_train)
    ;;
esac

case "${USE_WANDB,,}" in
  1|true|yes|on)
    cmd+=(--use_wandb)
    ;;
esac

cmd+=(--wandb_project "$WANDB_PROJECT")
cmd+=(--wandb_resume "$WANDB_RESUME")
cmd+=(--wandb_log_every "$WANDB_LOG_EVERY")
if [[ -n "$WANDB_ENTITY" ]]; then
  cmd+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_RUN_NAME" ]]; then
  cmd+=(--wandb_run_name "$WANDB_RUN_NAME")
fi
if [[ -n "$WANDB_ID" ]]; then
  cmd+=(--wandb_id "$WANDB_ID")
fi

if [[ -n "$MAX_EXAMPLES" ]]; then
  cmd+=(--max_examples "$MAX_EXAMPLES")
fi

if [[ -n "$SCORE_NUM_WORKERS" ]]; then
  cmd+=(--score_num_workers "$SCORE_NUM_WORKERS")
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
