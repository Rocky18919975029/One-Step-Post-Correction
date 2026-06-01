#!/bin/bash
#SBATCH --job-name=psamp_parquet_math
#SBATCH -t 0-23:59
#SBATCH --mem=200000
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --array=0-687

# 86 shards x 8 seeds = 688 tasks for 8,523 examples with shard_size=100.
NUM_SHARDS=86
NUM_SEEDS=8
SEED=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

export HF_HOME={HUGGING_FACE_HOME}
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"

export PYTHONPATH="$PYTHONPATH:{/path/to/One-Step-Post-Correction/llm_experiments}"
export HF_TOKEN={HF_TOKEN}

source activate psamp
cd /path/to/One-Step-Post-Correction/llm_experiments

echo "Running shard BATCH_IDX=${BATCH_IDX} with SEED=${SEED} (task ${SLURM_ARRAY_TASK_ID})"
python power_samp_parquet_math.py \
  --data_path=../data/train.parquet \
  --batch_idx="${BATCH_IDX}" \
  --shard_size=100 \
  --mcmc_steps=10 \
  --temperature=0.25 \
  --seed="${SEED}" \
  --model=qwen
