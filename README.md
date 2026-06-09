# One-Step-Post-Correction

This repository implements the current block-wise training algorithm for math reasoning with offline buffers, LoRA fine-tuning, and vLLM-based sampling/evaluation.

The README is intentionally written for the *current* manuscript and code only. It does not preserve older training routes, deprecated launchers, or historical variants.

## What This Repository Does

We train a model to improve math reasoning by optimizing **partial completions** stage by stage.

For each prompt `x`, we do not train directly on full sampled completions. Instead, we split the reasoning process into `num_blocks` stages. At stage `k`, we define a token budget

```text
l_k = min(k * block_size, max_completion_tokens)
```

and train on partial completions `y_{<= l_k}` only.

The key idea is:

1. Sample several partial completions up to stage `k`.
2. For each partial completion, sample several future continuations to the end.
3. Give the partial completion reward `1` if **any** sampled future reaches a correct final answer, otherwise `0`.
4. Train the model so that the distribution over partial completions better matches this stage-wise target.

This is an **offline-buffered** implementation:

- sampling and reward estimation happen first
- stage buffers are written to disk as CSV files
- training then consumes those saved buffers

The implementation defaults to single-GPU training, can optionally use DDP for multi-GPU training, and can shard sampling across multiple GPUs.

## Current Algorithm

### Stage semantics

For stage `k`:

- training object: partial completion `y_{<= l_k}`
- reward object: whether this partial completion can be extended to a correct final answer
- reward aggregation: **OR / any-hit**, not mean reward

That means the code optimizes:

- `log pi_theta(y_{<= l_k} | x)`
- `log pi_ref(y_{<= l_k} | x)`
- a binary stage reward estimated from future rollouts

It does **not** optimize log-probability on the full completion at each stage.

### Buffer semantics

Each row in `buffers/block_k.csv` represents one stage-`k` training sample:

- a prompt `x`
- one sampled partial completion `y_{<= l_k}`
- a binary reward for that partial completion

The reward is computed by sampling `future_completions_per_partial` futures from `x + y_{<= l_k}` and setting:

```text
reward = 1  if any future ends correct
reward = 0  otherwise
```

The buffer also stores diagnostic fields such as:

- `future_reward_mean`
- `future_any_correct`
- `parsed_answer`
- `has_boxed_answer`

Only `reward` is used for training.

## Code Structure

The active implementation lives in [`llm_experiments`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments).

The most important files are:

- [`llm_experiments/run_blockwise_buffer_pipeline.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/run_blockwise_buffer_pipeline.py)
  Unified scheduler for the maintained workflow. It handles stage-by-stage sampling, training, resume, buffer reuse, and final evaluation.

- [`llm_experiments/blockwise_vllm_sample_buffer.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/blockwise_vllm_sample_buffer.py)
  vLLM sampling entrypoint. When multiple sampler GPUs are visible, it shards the dataset into multiple single-GPU workers and merges the resulting CSV shards automatically.

- [`llm_experiments/blockwise_power_tb_buffer_train.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/blockwise_power_tb_buffer_train.py)
  Main training script for offline stage buffers. It also contains:
  - stage buffer generation logic
  - vLLM eval-only path
  - resume / checkpoint logic

- [`llm_experiments/blockwise_power_tb_train.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/blockwise_power_tb_train.py)
  Shared lower-level utility module, not a runnable training entrypoint:
  - LoRA model loading
  - checkpoint save/load
  - trajectory-balance-style loss
  - reward parsing and grading helpers

- [`llm_experiments/make_mini_train_parquet.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/make_mini_train_parquet.py)
  Helper for constructing a reproducible mini training set from a larger parquet file.

- [`llm_experiments/inspect_blockwise_samples.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/inspect_blockwise_samples.py)
  Small inspection helper for looking through generated samples.

Historical training and evaluation scripts that are not part of the maintained
workflow have been removed from this repository.

## End-to-End Workflow

The maintained workflow is:

1. Build or load a training dataset.
2. For block `k`, sample a partial-completion buffer with vLLM.
3. Train from that buffer on a single GPU.
4. Save `checkpoint_latest`.
5. Repeat for the next block.
6. Run final evaluation from the latest checkpoint with vLLM.

The unified pipeline script does this automatically.

### Sampling

Sampling for block `k` works as follows:

1. Build prompts with the repo's existing math prompt formatter.
2. Sample `completions_per_prefix` partial completions up to `l_k`.
3. For each partial completion, sample `future_completions_per_partial` futures.
4. Score each full completion against the ground-truth answer.
5. Write one CSV row per partial completion.

When multiple sampler GPUs are provided, the dataset is split by example index and processed by independent single-GPU workers. The workers write temporary shard CSVs, which are then merged into one final `block_k.csv`.

### Training

Training from a buffer works as follows:

1. Read `buffers/block_k.csv`.
2. Group rows by `example_idx`.
3. For each training example, take `completions_per_prefix` partial completions.
4. Encode `prompt + partial_completion`.
5. Compute the loss on the partial completion tokens only.
6. Update the LoRA adapter.

The current reference model is the base model with the LoRA adapter disabled.

### Evaluation

Evaluation uses the latest adapter checkpoint and runs full completion generation with vLLM on the evaluation set. The current maintained eval path is the `--eval_only --eval_backend vllm` route implemented in [`llm_experiments/blockwise_power_tb_buffer_train.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/blockwise_power_tb_buffer_train.py).

## Data And Outputs

### Training data

The training parquet is expected to contain a math question and ground-truth answer. The loader currently reads:

- `question`
- `gt_answer` or `answer`

and converts them into the internal format:

```text
{"prompt": ..., "answer": ...}
```

### Main output directory layout

A typical run directory looks like this:

```text
results/<run_name>/
  buffers/
    block_1.csv
    block_2.csv
    ...
  checkpoint_latest/
    adapter/
    optimizer.pt
    training_state.pt
    ...
  metrics.csv
  samples.csv
  eval_metrics.csv
  final/
  debug_logs/
```

### Buffer columns

The most important columns in `block_k.csv` are:

- `block_idx`
- `example_idx`
- `sample_idx`
- `prefix_text`
- `completion`
- `completion_token_len`
- `reward`
- `future_reward_mean`
- `future_any_correct`

Interpretation:

- `prefix_text` is the prompt
- `completion` is the stage-limited partial completion
- `reward` is the binary OR reward used for training

## Setup

```bash
git clone https://github.com/Rocky18919975029/One-Step-Post-Correction.git
cd One-Step-Post-Correction
conda env create -f environment.yml
conda activate psamp
```

Some workflows also require:

```bash
pip install peft safetensors
```

If you want wandb logging:

```bash
pip install wandb
```

## Recommended Commands

### Create a 500-example mini training set

```bash
cd llm_experiments
python make_mini_train_parquet.py \
  --input ../data/train.parquet \
  --output ../data/mini_train.parquet \
  --count 500 \
  --seed 0
```

### Full maintained pipeline

This is the main entrypoint.

```bash
cd llm_experiments

python run_blockwise_buffer_pipeline.py \
  --gpu 0 \
  --sampler_gpus 0,1,2,3 \
  --output_dir results/blockwise_buffer_mini500_qwen_seed0 \
  --data_path ../data/mini_train.parquet \
  --eval_data_path data/MATH500.json \
  --model qwen \
  --max_examples 500 \
  --batch_size 4 \
  --micro_batch_size 2 \
  --score_micro_batch_size 1 \
  --epochs 1 \
  --num_blocks 16 \
  --block_size 192 \
  --completions_per_prefix 4 \
  --future_completions_per_partial 4 \
  --max_completion_tokens 3072 \
  --temperature 0.25 \
  --alpha 4.0 \
  --beta 1.0 \
  --lr 1e-5 \
  --seed 0 \
  --gradient_checkpointing \
  --save_samples \
  --save_every_block \
  --save_every_steps 5 \
  --eval_backend vllm \
  --eval_examples 100 \
  --eval_max_new_tokens 3072 \
  --vllm_dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 4096 \
  --vllm_batch_size 32 \
  --vllm_enforce_eager \
  --debug_dump_timeout_seconds 60 \
  --use_wandb \
  --wandb_project one-step-post-correction \
  --wandb_run_name mini500-qwen-seed0
```

This configuration means:

- sampling uses up to 4 GPUs, sharded by example
- training stays on a single GPU
- stage buffers are reused on resume if they are complete
- final evaluation runs from the latest checkpoint

To train with DDP on multiple GPUs, add `--ddp_train` and set `--train_gpus`:

```bash
cd llm_experiments

python run_blockwise_buffer_pipeline.py \
  --gpu 0 \
  --train_gpus 0,1,2,3 \
  --sampler_gpus 0,1,2,3 \
  --ddp_train \
  --output_dir results/blockwise_buffer_mini500_qwen_seed0_ddp \
  --data_path ../data/mini_train.parquet \
  --eval_data_path data/MATH500.json \
  --model qwen \
  --max_examples 500 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --score_micro_batch_size 1 \
  --epochs 1 \
  --num_blocks 4 \
  --block_size 192 \
  --completions_per_prefix 4 \
  --future_completions_per_partial 4 \
  --max_completion_tokens 3072 \
  --temperature 0.25 \
  --alpha 4.0 \
  --beta 1.0 \
  --lr 1e-5 \
  --seed 0 \
  --gradient_checkpointing \
  --save_samples \
  --save_every_block \
  --eval_backend vllm \
  --eval_examples 100 \
  --eval_max_new_tokens 3072 \
  --vllm_dtype bfloat16 \
  --vllm_gpu_memory_utilization 0.9 \
  --vllm_max_model_len 4096 \
  --vllm_batch_size 32 \
  --vllm_enforce_eager \
  --use_wandb \
  --wandb_project one-step-post-correction \
  --wandb_run_name mini500-qwen-seed0-ddp \
  --quiet_debug_logs
```

### Small smoke test

```bash
cd llm_experiments
python run_blockwise_buffer_pipeline.py --smoke --output_dir results/blockwise_buffer_single_smoke
```

## Resume Semantics

Resume is built around `checkpoint_latest`.

The pipeline checks `checkpoint_latest/training_state.pt` to decide the next block. It also checks whether the current block buffer is already complete. If the buffer exists and has the expected number of rows and samples, the pipeline skips that block's sampling step and goes straight to training.

Mid-block resume is also supported. When `save_every_steps > 0`, the trainer writes:

- current block
- current epoch
- current batch start
- shuffled example order

so training can restart from inside a block rather than repeating the whole block.

## Key Parameters

### Stage structure

- `num_blocks`: number of stages
- `block_size`: tokens added per stage
- `max_completion_tokens`: full completion budget

### Sampling

- `completions_per_prefix`: number of partial completions sampled per prompt per stage
- `future_completions_per_partial`: number of future rollouts used to estimate the binary stage reward
- `vllm_batch_size`: vLLM prompt batch size during sampling/eval
- `sampler_gpus`: GPUs used for sharded vLLM sampling
- `train_gpus`: GPUs used by the trainer; defaults to `gpu`
- `ddp_train`: launch the trainer with `torchrun` over all `train_gpus`

### Training

- `batch_size`: optimizer batch size measured in prompts
- `micro_batch_size`: number of prompts per gradient-accumulation micro-step
- `score_micro_batch_size`: number of sampled partial completions scored together inside the loss

Practical interpretation:

- `micro_batch_size` mainly affects training memory and throughput
- `score_micro_batch_size` mainly affects inner loss-scoring memory and throughput
- under DDP, `batch_size` is per rank, so global effective prompt batch size is `batch_size * number_of_train_gpus`

### Logging

- `save_every_block`: save a full checkpoint at block boundaries
- `save_every_steps`: save resumable checkpoints inside a block
- `use_wandb`: enable experiment tracking
- `wandb_log_every`: step-level wandb logging interval

Current default behavior is conservative: step-level wandb logging is disabled by default in the maintained pipeline, because local logs and CSVs are more stable than very frequent remote logging.

## Current Design Choices

### Training

The maintained trainer defaults to single-GPU for reliability and can be launched in DDP mode when multiple training GPUs are available:

- single-GPU training is the default path
- DDP training is enabled with `--ddp_train --train_gpus 0,1,...`
- rank 0 writes checkpoints, metrics, eval outputs, and sample dumps
- each DDP rank writes its own debug log under `debug_logs/trainer_rank*.log`
- mid-block checkpointing with `save_every_steps` is not supported in DDP yet; use `save_every_block`

### Multi-GPU sampling is sharded, not tensor-parallel

When you pass multiple `sampler_gpus`, the current code does **not** build one tensor-parallel vLLM engine. Instead it launches multiple single-GPU sampling workers and merges their buffers afterward.

That choice is important for throughput on 7B-class models.

### Offline buffers are first-class

This repository is not trying to hide the offline-buffer compromise. The implementation explicitly chooses:

- sampled CSV buffers on disk
- resumable training from those buffers
- deterministic inspection of stage data

This makes the system easier to debug and easier to recover after interruptions.

## Reading The Code

If you want to understand the implementation from top to bottom, this is the most useful reading order:

1. [`llm_experiments/run_blockwise_buffer_pipeline.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/run_blockwise_buffer_pipeline.py)
2. [`llm_experiments/blockwise_vllm_sample_buffer.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/blockwise_vllm_sample_buffer.py)
3. [`llm_experiments/blockwise_power_tb_buffer_train.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/blockwise_power_tb_buffer_train.py)
4. [`llm_experiments/blockwise_power_tb_train.py`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/blockwise_power_tb_train.py)

That order mirrors the actual runtime:

- scheduler
- sampling
- offline buffer training
- shared model/loss/checkpoint utilities

## What Is Not Described Here

This README does not document:

- historical experimental scripts removed from the maintained tree
- removed distributed training launchers
- historical manuscript variants
- older full-completion buffer semantics

Everything here is intended to describe the current maintained algorithm and the code that implements it.
