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
