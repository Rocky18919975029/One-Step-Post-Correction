#!/usr/bin/env bash
#SBATCH --job-name=ospc-eval-block
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Usage:
#   sbatch --export=ALL,BLOCK_IDX=1 scripts/submit_eval_block.slurm
#   sbatch --export=ALL,BLOCK_IDX=latest scripts/submit_eval_block.slurm

REPO_DIR="${REPO_DIR:-/data/user/zhongal/One-Step-Post-Correction}"
CONDA_ENV="${CONDA_ENV:-psamp}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/psamp}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PATH/bin/python}"
CONDA_SH="${CONDA_SH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-results/ddp_full_c4_f4_flash_accel_3072_b192}"
BLOCK_IDX="${BLOCK_IDX:-latest}"
MODEL="${MODEL:-/data/user/zhongal/.cache/qwen2.5-7b-local}"
PROMPT_MODEL="${PROMPT_MODEL:-qwen}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/MATH500.json}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-500}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-3072}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

HF_HOME="${HF_HOME:-/data/user/zhongal/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"

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

cd "$REPO_DIR/llm_experiments"

export PYTHONUNBUFFERED=1
export HF_HOME
export TRANSFORMERS_CACHE
export HF_HUB_OFFLINE
export TRANSFORMERS_OFFLINE
export WANDB_MODE
export WANDB_DISABLED
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cmd=(
  "$PYTHON_BIN" eval_block_checkpoint.py
  --output_dir "$OUTPUT_DIR"
  --block_idx "$BLOCK_IDX"
  --eval_data_path "$EVAL_DATA_PATH"
  --model "$MODEL"
  --prompt_model "$PROMPT_MODEL"
  --eval_backend "$EVAL_BACKEND"
  --eval_examples "$EVAL_EXAMPLES"
  --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --vllm_dtype "$VLLM_DTYPE"
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
  --vllm_batch_size "$VLLM_BATCH_SIZE"
)

case "${VLLM_ENFORCE_EAGER,,}" in
  1|true|yes|on)
    cmd+=(--vllm_enforce_eager)
    ;;
esac

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
