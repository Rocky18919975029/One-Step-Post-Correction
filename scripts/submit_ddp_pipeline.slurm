#!/usr/bin/env bash
#SBATCH --job-name=ospc-ddp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Usage:
#   sbatch scripts/submit_ddp_pipeline.slurm
#
# Common overrides:
#   sbatch --export=ALL,NUM_BLOCKS=16,MAX_EXAMPLES=500,OUTPUT_DIR=results/full_ddp scripts/submit_ddp_pipeline.slurm
#
# Edit the SBATCH partition/account lines above if your HPC requires them.
# GPU compute nodes are assumed to have no internet access. Make sure the model
# and datasets are already present in the local Hugging Face/data caches.

REPO_DIR="${REPO_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-psamp}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-$GPU_IDS}"
SAMPLER_GPUS="${SAMPLER_GPUS:-$GPU_IDS}"

OUTPUT_DIR="${OUTPUT_DIR:-results/ddp_full_run}"
DATA_PATH="${DATA_PATH:-../data/train.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/MATH500.json}"
MODEL="${MODEL:-qwen}"

MAX_EXAMPLES="${MAX_EXAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SCORE_MICRO_BATCH_SIZE="${SCORE_MICRO_BATCH_SIZE:-1}"
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

EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-500}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-3072}"

VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-8}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
else
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29600}"
fi
SAMPLER_MASTER_PORT="${SAMPLER_MASTER_PORT:-$((TRAIN_MASTER_PORT + 1))}"

mkdir -p logs

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

cd "$REPO_DIR/llm_experiments"

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME
export TRANSFORMERS_CACHE
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
export WANDB_MODE
export WANDB_DISABLED

cmd=(
  python run_blockwise_buffer_pipeline.py
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
  --batch_size "$BATCH_SIZE"
  --micro_batch_size "$MICRO_BATCH_SIZE"
  --score_micro_batch_size "$SCORE_MICRO_BATCH_SIZE"
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
  --gradient_checkpointing
  --save_samples
  --save_every_block
  --eval_backend "$EVAL_BACKEND"
  --eval_examples "$EVAL_EXAMPLES"
  --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS"
  --vllm_dtype "$VLLM_DTYPE"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
  --vllm_batch_size "$VLLM_BATCH_SIZE"
  --vllm_enforce_eager
  --quiet_debug_logs
)

if [[ -n "$MAX_EXAMPLES" ]]; then
  cmd+=(--max_examples "$MAX_EXAMPLES")
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
