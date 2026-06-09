# Training Crash Handoff

This document preserves the historical investigation notes for the blockwise
offline-buffer training crash.

The crash is now resolved. The final root cause and fix are documented in
[`TRAINING_SEGFAULT_ROOT_CAUSE.md`](TRAINING_SEGFAULT_ROOT_CAUSE.md). In short:
gradient-checkpointed micro-batch backward work was not synchronized before the
next micro-batch forward, which could trigger intermittent native CUDA
segfaults. The fix is to synchronize CUDA immediately after each micro-batch
`backward()`.

## Current Algorithm State

The current code implements the revised manuscript semantics:

- The buffer stores **partial completions**, not full completions.
- For stage `k`, each training object is `prompt + y_{\le l_k}`, where
  `l_k = k * block_size`.
- Future continuations are used only to estimate the partial completion reward.
- The reward is an **OR reward**:
  - sample `future_completions_per_partial` future continuations;
  - if any future reaches the correct final answer, the partial completion gets
    `reward = 1`;
  - otherwise it gets `reward = 0`.
- Training computes `log pi_theta` and `log pi_ref` only over the partial
  completion, i.e. from `prompt_len` to the end of the partial sequence.
- Sampling is offline-buffered. Training reads `buffers/block_k.csv`.
- Multi-GPU sampling is implemented as sharded single-GPU workers, not tensor
  parallelism.
- Training is intentionally single-GPU.

Important files:

- `llm_experiments/run_blockwise_buffer_pipeline.py`
- `llm_experiments/blockwise_vllm_sample_buffer.py`
- `llm_experiments/blockwise_power_tb_buffer_train.py`
- `llm_experiments/blockwise_power_tb_train.py`
- `llm_experiments/replay_blockwise_micro_batch.py`
- `llm_experiments/run_training_crash_diagnostics.py`

## Working Evidence

### Partial Buffer And OR Reward Were Verified

A smoke buffer showed:

- `completion_token_len = 192` for block 1;
- `prefix_text` is the original prompt;
- `completion` is the partial completion;
- reward fields exist:
  - `reward`
  - `future_reward_mean`
  - `future_any_correct`

OR reward was directly observed. Some rows had:

```text
future_reward_mean = 0.166667
reward = 1.0
future_any_correct = True
```

This verifies that reward is not the average future reward. It is the OR over
future correctness.

### Four-GPU Sampling Was Changed

The first multi-GPU sampling attempt used vLLM tensor parallelism. That did not
speed up generation for the 7B model.

The sampler was then changed to sharded sampling:

- one scheduler process;
- one independent single-GPU vLLM worker per sampler GPU;
- each worker processes a shard of examples;
- shard CSVs are merged into `buffers/block_k.csv`.

This is expected to improve sampling throughput for 7B models that fit on one
GPU.

### Buffer Reuse On Resume Was Added

The pipeline now checks whether the current block buffer is complete before
sampling. If complete, it skips sampling:

```text
Reusing sampled buffer for block 1
Found complete buffer at .../buffers/block_1.csv; skipping sampling.
```

## Resolved Problem

Training crashes during block 1 with a hard crash:

```text
Segmentation fault (core dumped)
```

The crash happens inside the training loss forward path. The Python-level debug
logs consistently show the last message as:

```text
[block 1] step N loss forward begin
```

Then the process segfaults or hangs inside model forward.

Observed examples:

- crash around step 8, `micro_start=2`;
- crash around step 16 or step 17 in other runs;
- when faulthandler fired during one hang, the main thread was inside
  `torch.nn.modules.linear.py` `forward`.

The crash was not a normal Python exception. It was a native/runtime-level
failure. The successful fix was to call `sync_cuda_if_available()` immediately
after each micro-batch `backward()` before entering the next micro-batch
forward.

## Important Reproduction Commands

### Direct Trainer Reproduction

This uses the existing block 1 buffer and does not involve the full pipeline:

```bash
cd ~/One-Step-Post-Correction/llm_experiments

CUDA_VISIBLE_DEVICES=0 python blockwise_power_tb_buffer_train.py \
  --data_path ../data/mini_train.parquet \
  --eval_data_path data/MATH500.json \
  --model qwen \
  --max_examples 500 \
  --batch_size 4 \
  --micro_batch_size 2 \
  --score_micro_batch_size 1 \
  --epochs 1 \
  --num_blocks 1 \
  --block_size 192 \
  --completions_per_prefix 4 \
  --max_completion_tokens 3072 \
  --temperature 0.25 \
  --alpha 4.0 \
  --beta 1.0 \
  --lr 1e-5 \
  --seed 0 \
  --gradient_checkpointing \
  --skip_buffer_sampling \
  --output_dir results/blockwise_buffer_mini500_qwen_seed0 \
  --debug_dump_timeout_seconds 60 \
  --attn_implementation sdpa
```

This has crashed even without:

- the pipeline;
- vLLM;
- xformers;
- wandb;
- step-level wandb logging;
- `--save_samples`;
- `--save_every_steps`.

### One-Shot Diagnostic Matrix

A bounded diagnostic script was added so the next investigator can run a matrix
instead of trying one option at a time:

```bash
cd ~/One-Step-Post-Correction/llm_experiments

CUDA_VISIBLE_DEVICES=0 python run_training_crash_diagnostics.py \
  --source_output_dir results/blockwise_buffer_mini500_qwen_seed0 \
  --max_train_steps 30 \
  --max_replay_micro_batches 40 \
  --attn_implementation sdpa
```

It writes logs and a summary under:

```text
results/training_crash_diagnostics_YYYYMMDD_HHMMSS/
```

The cases include:

- `replay_fresh_per_step`
- `trainer_minimal_gc`
- `trainer_minimal_no_gc`
- `trainer_micro1_gc`
- `trainer_with_dump_tqdm_gc`
- `trainer_save_steps_gc`

The goal is to identify whether the failure follows:

- gradient checkpointing;
- micro-batch size;
- tqdm / debug dumping;
- checkpoint saving;
- or the trainer loop itself.

## What Has Been Ruled Out

The following explanations are unlikely based on direct tests:

### Not Caused By Sampling

The crash occurs after reusing a complete `block_1.csv` and skipping sampling.

### Not Caused By The Pipeline

The crash also occurs when directly running `blockwise_power_tb_buffer_train.py`.

### Not Caused By Step-Level `wandb.log`

The crash persisted with:

```text
--wandb_log_every 0
```

and also when `--use_wandb` was omitted entirely.

### Not Caused By One Specific Micro-Batch Alone

Dumped micro-batches were replayed with:

```bash
CUDA_VISIBLE_DEVICES=0 python replay_blockwise_micro_batch.py \
  --checkpoint_dir results/blockwise_buffer_mini500_qwen_seed0/checkpoint_latest \
  --micro_batch_csv results/blockwise_buffer_mini500_qwen_seed0/debug_logs/micro_batches/block1_step16_micro2.csv \
  --run_backward
```

and also:

```bash
CUDA_VISIBLE_DEVICES=0 python replay_blockwise_micro_batch.py \
  --checkpoint_dir results/blockwise_buffer_mini500_qwen_seed0/checkpoint_latest \
  --micro_batch_csv results/blockwise_buffer_mini500_qwen_seed0/debug_logs/micro_batches/block1_step16_micro2.csv \
  --run_backward \
  --run_optimizer_step
```

Both succeeded.

### Not Caused By Short Sequential Replay

Sequential replay of multiple dumped micro-batches also succeeded, including
replay with per-step optimizer cadence:

```bash
CUDA_VISIBLE_DEVICES=0 python replay_blockwise_micro_batch.py \
  --checkpoint_dir results/blockwise_buffer_mini500_qwen_seed0/checkpoint_latest \
  --micro_batch_glob "results/blockwise_buffer_mini500_qwen_seed0/debug_logs/micro_batches/block1_step*_micro*.csv" \
  --max_micro_batches 40 \
  --run_backward \
  --run_optimizer_step \
  --step_every_micro_batches 2
```

Important caveat: earlier replay runs used `checkpoint_latest` state, whereas
fresh trainer crashes often start from fresh LoRA. The replay tool now supports:

```text
--fresh_model
```

to test this difference.

### Not Fixed By Python 3.11

A Python 3.11 minimal training environment was created. The crash still occurred:

```text
[block 1] step 8 loss forward begin
Segmentation fault (core dumped)
```

This makes a Python 3.12-only explanation unlikely.

### Not Fixed By `attn_implementation=sdpa`

The direct trainer still crashed with:

```text
--attn_implementation sdpa
```

So the crash is not simply fixed by switching away from the default attention
backend.

## Current Suspicion

The strongest current hypothesis is that the failure is caused by an interaction
inside the live trainer loop during repeated model forward/backward/update,
possibly involving:

- PEFT LoRA + Qwen2.5-7B;
- gradient checkpointing;
- bfloat16;
- PyTorch/CUDA runtime behavior;
- repeated reference-model forward with `disable_adapter()`;
- or some trainer-specific state that the replay script still does not exactly
mirror.

This is not yet proven.

The fact pattern is:

- standalone micro-batch replay succeeds;
- sequential replay succeeds;
- direct trainer still crashes;
- crash occurs inside model forward, not in Python bookkeeping.

## Recommended Next Steps

1. Run `run_training_crash_diagnostics.py` and inspect `summary.tsv`.
2. If `trainer_minimal_no_gc` passes while `trainer_minimal_gc` fails, focus on
   gradient checkpointing.
3. If `trainer_micro1_gc` passes while `trainer_minimal_gc` fails, use
   `micro_batch_size=1` as the immediate workaround.
4. If all trainer cases fail but replay passes, diff trainer vs replay further.
   The likely remaining difference is how `train_stage_from_buffer` constructs
   batches and records metrics/sample outputs around the live loop.
5. Try an even more conservative runtime stack:
   - Python 3.10 or 3.11;
   - PyTorch 2.5.1;
   - a compatible Transformers/PEFT stack;
   - no xformers;
   - no vLLM in the training environment.
6. As a practical workaround, try:
   - `--micro_batch_size 1`;
   - no `--gradient_checkpointing`;
   - no `--save_samples`;
   - no `--save_every_steps`;
   - no wandb;
   - `--attn_implementation sdpa`;
   and then add features back only after the base trainer survives block 1.

## Relevant Recent Commits

The debugging tools and related changes were introduced in these recent commits:

```text
8cd54e5 Add one-shot training crash diagnostics
3b5a366 Allow fresh-model replay from dumped batches
904be09 Mirror per-step optimizer cadence in replay
281e56d Allow glob-only micro-batch replay
e42f22a Add glob-based sequential micro-batch replay
38e4f8e Extend micro-batch replay for sequential repro
ea95a4c Add micro-batch replay debugging tools
43bb7c8 Add safer attention option and batch dumps
```

## Notes For The Next Engineer

Please avoid assuming this is caused by wandb, sampling, or a bad micro-batch.
Those were tested and did not explain the crash.

The fastest way to continue is to run the diagnostic matrix, read
`summary.tsv`, and then reduce the failing case further.
