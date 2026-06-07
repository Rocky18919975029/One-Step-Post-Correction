#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/blockwise_buffer_single_smoke}"
DATA_PATH="${DATA_PATH:-../data/train.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-data/MATH500.json}"
MODEL="${MODEL:-qwen}"

# Keep the main training/sampling hyperparameters aligned with the full run.
# Only the dataset scale / number of stages / eval count are reduced for a fast smoke test.
MAX_EXAMPLES="${MAX_EXAMPLES:-8}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-10}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-2}"
SEED="${SEED:-0}"

if [[ "${NUM_BLOCKS}" -lt 2 ]]; then
  echo "NUM_BLOCKS must be at least 2 for this smoke test because it validates resume-on-next-block."
  exit 1
fi

cd "$(dirname "$0")"

echo "[1/5] Sampling block 1 with vLLM"
CUDA_VISIBLE_DEVICES="${GPU}" python blockwise_vllm_sample_buffer.py \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --block_idx 1 \
  --max_examples "${MAX_EXAMPLES}" \
  --block_size 192 \
  --completions_per_prefix 4 \
  --max_completion_tokens 3072 \
  --temperature 0.25 \
  --seed "${SEED}" \
  --vllm_batch_size "${VLLM_BATCH_SIZE}" \
  --vllm_enforce_eager

echo "[2/5] Training block 1 from the sampled buffer"
CUDA_VISIBLE_DEVICES="${GPU}" python blockwise_power_tb_buffer_train.py \
  --data_path "${DATA_PATH}" \
  --eval_data_path "${EVAL_DATA_PATH}" \
  --model "${MODEL}" \
  --max_examples "${MAX_EXAMPLES}" \
  --batch_size 4 \
  --micro_batch_size 1 \
  --gradient_checkpointing \
  --epochs 1 \
  --num_blocks 1 \
  --block_size 192 \
  --completions_per_prefix 4 \
  --max_completion_tokens 3072 \
  --temperature 0.25 \
  --alpha 4.0 \
  --beta 1.0 \
  --lr 1e-5 \
  --seed "${SEED}" \
  --save_samples \
  --save_every_block \
  --output_dir "${OUTPUT_DIR}" \
  --skip_buffer_sampling \
  --debug_dump_timeout_seconds 60

echo "[3/5] Sampling block 2 with the latest adapter"
CUDA_VISIBLE_DEVICES="${GPU}" python blockwise_vllm_sample_buffer.py \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --block_idx 2 \
  --max_examples "${MAX_EXAMPLES}" \
  --block_size 192 \
  --completions_per_prefix 4 \
  --max_completion_tokens 3072 \
  --temperature 0.25 \
  --seed "${SEED}" \
  --vllm_batch_size "${VLLM_BATCH_SIZE}" \
  --adapter_path "${OUTPUT_DIR}/checkpoint_latest/adapter" \
  --vllm_enforce_eager

echo "[4/5] Resuming training for block 2"
CUDA_VISIBLE_DEVICES="${GPU}" python blockwise_power_tb_buffer_train.py \
  --data_path "${DATA_PATH}" \
  --eval_data_path "${EVAL_DATA_PATH}" \
  --model "${MODEL}" \
  --max_examples "${MAX_EXAMPLES}" \
  --batch_size 4 \
  --micro_batch_size 1 \
  --gradient_checkpointing \
  --epochs 1 \
  --num_blocks "${NUM_BLOCKS}" \
  --block_size 192 \
  --completions_per_prefix 4 \
  --max_completion_tokens 3072 \
  --temperature 0.25 \
  --alpha 4.0 \
  --beta 1.0 \
  --lr 1e-5 \
  --seed "${SEED}" \
  --save_samples \
  --save_every_block \
  --output_dir "${OUTPUT_DIR}" \
  --skip_buffer_sampling \
  --resume_from_checkpoint "${OUTPUT_DIR}/checkpoint_latest" \
  --debug_dump_timeout_seconds 60

echo "[5/5] Evaluating the latest checkpoint with the stable vLLM backend"
CUDA_VISIBLE_DEVICES="${GPU}" python blockwise_power_tb_buffer_train.py \
  --data_path "${DATA_PATH}" \
  --eval_data_path "${EVAL_DATA_PATH}" \
  --model "${MODEL}" \
  --eval_only \
  --eval_backend vllm \
  --eval_every_block \
  --eval_examples "${EVAL_EXAMPLES}" \
  --eval_max_new_tokens 3072 \
  --vllm_batch_size "${VLLM_BATCH_SIZE}" \
  --vllm_enforce_eager \
  --output_dir "${OUTPUT_DIR}" \
  --resume_from_checkpoint "${OUTPUT_DIR}/checkpoint_latest" \
  --debug_dump_timeout_seconds 60

echo
echo "Smoke test complete."
echo "Outputs are under: ${OUTPUT_DIR}"
