#!/usr/bin/env bash
#SBATCH --job-name=ospc-b1-greedy
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=640G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/user/zhongal/One-Step-Post-Correction}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/psamp}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PATH/bin/python}"
CONDA_SH="${CONDA_SH:-}"

RUN_DIR="${RUN_DIR:-results/c8_m1_greedy_alpha05_fsdp_fullft_hpc}"
BUFFER_PATH="${BUFFER_PATH:-$RUN_DIR/buffers/block_2.csv}"
MODEL="${MODEL:-$RUN_DIR/block_1}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/diagnostics/block1_greedy_from_block2}"
ANCHOR_TOKENS="${ANCHOR_TOKENS:-192}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-3072}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SAMPLE_NUM_QUESTIONS="${SAMPLE_NUM_QUESTIONS:-50}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
SEED="${SEED:-0}"
SAVE_GREEDY_COMPLETION="${SAVE_GREEDY_COMPLETION:-0}"

HF_HOME="${HF_HOME:-/data/user/zhongal/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

module purge
module load miniconda3
if [[ -z "$CONDA_SH" ]]; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV_PATH"

cd "$REPO_DIR/llm_experiments"

export PYTHONUNBUFFERED=1
export HF_HOME TRANSFORMERS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cmd=(
  "$PYTHON_BIN" diagnose_block1_greedy_rewards.py
  --buffer_path "$BUFFER_PATH"
  --output_dir "$OUTPUT_DIR"
  --model "$MODEL"
  --anchor_tokens "$ANCHOR_TOKENS"
  --max_completion_tokens "$MAX_COMPLETION_TOKENS"
  --vllm_dtype "$VLLM_DTYPE"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
  --vllm_batch_size "$VLLM_BATCH_SIZE"
  --seed "$SEED"
  --num_shards "$NUM_SHARDS"
  --sample_num_questions "$SAMPLE_NUM_QUESTIONS"
)

if [[ -n "$ADAPTER_PATH" ]]; then
  cmd+=(--adapter_path "$ADAPTER_PATH")
fi
case "${VLLM_ENFORCE_EAGER,,}" in
  1|true|yes|on) cmd+=(--vllm_enforce_eager) ;;
esac
case "${SAVE_GREEDY_COMPLETION,,}" in
  1|true|yes|on) cmd+=(--save_greedy_completion) ;;
esac

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
