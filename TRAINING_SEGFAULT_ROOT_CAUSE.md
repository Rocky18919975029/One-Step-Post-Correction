# Training Segfault Root Cause

This note records the resolved native crash observed during block-wise buffer
training, so future debugging does not repeat the same investigation.

## Symptom

Training intermittently crashed with:

```text
Segmentation fault (core dumped)
```

The crash happened during block 1 training, often after a successful
micro-batch backward and either:

- at the start of the next micro-batch forward, or
- immediately after logging `backward begin`.

The failure was a native/runtime crash, not a Python exception.

## What It Was Not

The investigation ruled out several likely suspects:

- It was not caused by vLLM sampling. The crash reproduced with
  `--skip_buffer_sampling` while reusing an existing `buffers/block_1.csv`.
- It was not one fixed bad training row. Skipping rows that appeared near one
  crash only moved the crash later.
- It was not specific to SDPA attention. Eager attention also crashed.
- It was not solved by removing gradient checkpointing, because that path ran
  out of memory for the current Qwen 7B setup.

## Root Cause

The stable reproduction pattern was:

1. A batch is split into multiple micro-batches.
2. `optimizer.zero_grad(set_to_none=True)` runs once at the start of the batch.
3. Each micro-batch runs forward and backward.
4. Gradients accumulate across micro-batches before `optimizer.step()`.

With gradient checkpointing enabled, backward recomputes parts of the forward
graph. CUDA kernels and allocator work are asynchronous, so Python can return
from `.backward()` before all GPU work and memory reuse are fully synchronized.

The next micro-batch forward could then begin while the previous micro-batch
backward/recompute was still settling, causing intermittent CUDA memory/state
corruption and a native segfault.

## Fix

Synchronize CUDA immediately after each micro-batch backward:

```python
(loss * (micro_sequences / total_sequences)).backward()
sync_cuda_if_available()
```

This forces the previous micro-batch backward kernels and memory operations to
finish before the next micro-batch forward starts.

The same fix was applied to both maintained buffer training and the older direct
training path.

## Validation

After adding the post-backward CUDA synchronization and running with
`--debug_dump_timeout_seconds 0`, block 1 training completed successfully:

```text
block 1 epoch 0: 100%|...| 125/125
[block 1] checkpoint_latest updated
[final] outputs written
```

## Recommendation

For this Qwen 7B block-wise buffer setup:

- Keep `--gradient_checkpointing` enabled to avoid OOM.
- Keep the post-backward CUDA synchronization in the training loop.
- Prefer `--debug_dump_timeout_seconds 0` for full training runs unless a
  traceback dump is specifically needed for debugging.

