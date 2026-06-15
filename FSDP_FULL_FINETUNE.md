# FSDP Full-Finetune Backend

The maintained Accelerate trainer has two explicit backends:

- `ddp` (default): the existing LoRA/DDP path and checkpoint format.
- `fsdp`: full-parameter BF16 fine-tuning with PyTorch FSDP `FULL_SHARD`.

The FSDP path is opt-in and does not change existing DDP commands. Use the
dedicated submission script on one 8xH100 node:

```bash
sbatch --export=ALL,\
MODEL=/data/user/zhongal/.cache/qwen2.5-7b-local,\
OUTPUT_DIR=results/fsdp_smoke,\
MAX_EXAMPLES=32,NUM_BLOCKS=2,EVAL_BACKEND=none \
scripts/submit_fsdp_pipeline.slurm
```

For a formal run, omit `MAX_EXAMPLES` and set the experiment parameters in the
same way as the DDP script. A conservative full-finetune learning-rate default
of `1e-6` is used by the FSDP wrapper.

FSDP checkpoints contain:

- `checkpoint_latest/model/`: merged Hugging Face model used by the next
  sample/score stage.
- `checkpoint_latest/accelerate_state/`: sharded model/optimizer state used to
  resume training.
- `checkpoint_latest/training_state.pt`: pipeline metadata such as the next
  block index.
- `block_<k>/`: standalone merged model checkpoint for evaluation.

Runtime state is sharded to keep optimizer checkpointing scalable. A full
Hugging Face checkpoint is also materialized because vLLM must load the updated
policy between blocks.

For long-running training on a remote server, periodic restartable checkpoints
can be enabled with `SAVE_EVERY_STEPS=20`. Intermediate checkpoints contain the
sharded FSDP model/optimizer state and the exact epoch/dataloader position.
Block-end checkpoints additionally contain the merged Hugging Face model.

## FSDP performance regression checklist

FSDP depends on fast GPU-to-GPU collectives for parameter all-gathers. On the
8xH100 nodes, do not inherit the legacy DDP debugging defaults
`NCCL_P2P_DISABLE=1` or `NCCL_IB_DISABLE=1`. Disabling both transports caused a
confirmed regression from roughly 13 seconds per optimizer step to roughly 393
seconds per optimizer step while GPU utilization remained high. This can look
like a hang, but it is FSDP repeatedly communicating through a slow fallback.

The maintained FSDP entry points therefore default to:

```bash
NCCL_P2P_DISABLE=0
NCCL_IB_DISABLE=0
CUDA_LAUNCH_BLOCKING=0
```

When a standalone block run is unexpectedly slow, compare its effective launch
command and environment with `scripts/submit_fsdp_pipeline.slurm` before
changing model, batch, checkpoint, or optimizer settings. Also remember that
the tqdm bar advances only after a complete gradient-accumulation cycle and an
optimizer step; a temporarily unchanged bar is not by itself evidence of a
deadlock.

Periodic sharded checkpoints can be converted to standalone Hugging Face models
with `scripts/submit_export_fsdp_checkpoint.slurm`. The exporter reconstructs
the same eight-rank FSDP layout, restores the Accelerate state, and writes only
the merged model; it does not read a training buffer or perform an optimizer
step.
