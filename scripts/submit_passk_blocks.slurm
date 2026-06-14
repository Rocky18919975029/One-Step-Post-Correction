#!/usr/bin/env bash
#SBATCH --job-name=ospc-passk
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
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/psamp}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PATH/bin/python}"
CONDA_SH="${CONDA_SH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-results/c8_f4_token_alpha05_stochastic_m4_6blocks}"
BLOCKS="${BLOCKS:-base,1,2,3,4,5,6}"
MODEL="${MODEL:-/data/user/zhongal/.cache/qwen2.5-7b-local}"
PROMPT_MODEL="${PROMPT_MODEL:-qwen}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/MATH500.json}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
K_VALUES="${K_VALUES:-1,4,16,32,64,128}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-3072}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-4}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-temp0.6_top_p0.95_n128}"
SAVE_RESPONSES="${SAVE_RESPONSES:-0}"
FORCE="${FORCE:-0}"

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
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

cmd=(
  "$PYTHON_BIN" eval_passk_blocks.py
  --output_dir "$OUTPUT_DIR"
  --blocks "$BLOCKS"
  --eval_data_path "$EVAL_DATA_PATH"
  --model "$MODEL"
  --prompt_model "$PROMPT_MODEL"
  --eval_examples "$EVAL_EXAMPLES"
  --num_samples "$NUM_SAMPLES"
  --k_values "$K_VALUES"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --prompt_batch_size "$PROMPT_BATCH_SIZE"
  --vllm_dtype "$VLLM_DTYPE"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
  --seed "$SEED"
  --num_shards 8
  --run_name "$RUN_NAME"
)

case "${VLLM_ENFORCE_EAGER,,}" in
  1|true|yes|on) cmd+=(--vllm_enforce_eager) ;;
esac
case "${SAVE_RESPONSES,,}" in
  1|true|yes|on) cmd+=(--save_responses) ;;
esac
case "${FORCE,,}" in
  1|true|yes|on) cmd+=(--force) ;;
esac

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
