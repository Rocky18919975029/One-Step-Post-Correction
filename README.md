# One-Step-Post-Correction


### [Paper](https://arxiv.org/abs/2510.14901) | [Project Page](https://aakaran.github.io/reasoning_with_sampling/)

[![rws](teaser.png)](teaser.png)


This repo contains the PyTorch implementation for One-Step-Post-Correction, adapted from Reasoning with Sampling.
> [**Reasoning with Sampling: Your Base Model is Smarter Than You Think**](https://arxiv.org/abs/2510.14901)<br>
> [Aayush Karan](https://aakaran.github.io/), [Yilun Du](https://yilundu.github.io/)
> <br>Harvard<br>



## Setup

Run the following script to setup environment.

```bash
git clone https://github.com/Rocky18919975029/One-Step-Post-Correction.git
cd One-Step-Post-Correction
conda env create -f environment.yml
conda activate psamp
```


## Sampling
The maintained path in this repo is the block-wise single-process workflow documented below. Older one-shot sampling, evaluation, pass@k, and Slurm helper scripts have been moved to [`llm_experiments/legacy/`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/legacy) so they do not clutter the main training surface.

To run power sampling on MATH500 with 8 seeds and the eval set split across 5 shards:
```bash
sbatch llm_experiments/legacy/scripts/power_samp_math.sh
```
The output is several .csv files (based on the shard and seed number) that store the response outputs, correct answers, original prompts, etc. 

The maintained path in this repo is now single-process execution. If you want to shard or queue jobs across several GPUs, launch separate single-GPU processes externally rather than using the removed in-repo DDP launchers.

## Block-wise Power Distribution Matching

To train the block-wise reward-augmented power distribution objective from the manuscript, first install LoRA support if it is not already present:

```bash
pip install peft
```

Then run a small smoke test:

```bash
cd llm_experiments
CUDA_VISIBLE_DEVICES=0 python blockwise_power_tb_train.py \
  --model qwen_math \
  --max_examples 1 \
  --num_blocks 1 \
  --completions_per_prefix 2 \
  --max_completion_tokens 64 \
  --save_samples
```

The script uses the existing MATH prompt format, boxed-answer parser, and math grader as the answer-correctness reward. It implements the manuscript's arithmetic-mean VarGrad estimate:

```text
log Z_hat = mean_m(alpha log pi_ref - log pi_theta + reward / beta)
```

and the trajectory balance loss:

```text
(stopgrad(log Z_hat) + log pi_theta - alpha log pi_ref - reward / beta)^2
```

The default block hyperparameters follow the sampling code: ```max_new_tokens=3072```, ```num_blocks=16```, and ```block_size=192```.

Training proceeds stage-by-stage: block ```k``` is trained for all requested epochs before moving to block ```k+1```, and the updated model is used to generate prefixes for the next stage.

When ```--save_samples``` is enabled, sampled completions are written to ```samples.csv``` with block index, example index, prefix text, completion text, parsed answer, reward, and reference/trainable log probabilities.

For a small multi-block debug run:

```bash
CUDA_VISIBLE_DEVICES=0 python blockwise_power_tb_train.py \
  --model qwen_math \
  --max_examples 2 \
  --num_blocks 3 \
  --completions_per_prefix 2 \
  --max_completion_tokens 512 \
  --save_samples \
  --output_dir results/blockwise_tb_debug
```

Then inspect every block's inputs, outputs, rewards, and log probabilities:

```bash
python inspect_blockwise_samples.py results/blockwise_tb_debug --show_examples 2
```

Optional experiment tracking, checkpoint resume, and block-end evaluation all work in the maintained single-process path:

```bash
CUDA_VISIBLE_DEVICES=0 python blockwise_power_tb_train.py \
  --model qwen_math \
  --eval_data_path data/MATH500.json \
  --max_examples 32 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --score_micro_batch_size 1 \
  --gradient_checkpointing \
  --num_blocks 16 \
  --completions_per_prefix 2 \
  --max_completion_tokens 3072 \
  --save_samples \
  --save_every_block \
  --eval_every_block \
  --eval_examples 32 \
  --eval_max_new_tokens 3072 \
  --use_wandb \
  --wandb_project one-step-post-correction \
  --wandb_run_name blockwise-32x16 \
  --output_dir results/blockwise_tb_single_32x16
```

The latest stage-boundary checkpoint is stored in ```checkpoint_latest```. Resume with:

```bash
CUDA_VISIBLE_DEVICES=0 python blockwise_power_tb_train.py \
  --resume_from_checkpoint results/blockwise_tb_single_32x16/checkpoint_latest \
  --model qwen_math \
  --eval_data_path data/MATH500.json \
  --max_examples 32 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --score_micro_batch_size 1 \
  --gradient_checkpointing \
  --num_blocks 16 \
  --completions_per_prefix 2 \
  --max_completion_tokens 3072 \
  --save_samples \
  --save_every_block \
  --eval_every_block \
  --eval_examples 32 \
  --eval_max_new_tokens 3072 \
  --use_wandb \
  --wandb_resume allow \
  --output_dir results/blockwise_tb_single_32x16
```

If the checkpoint has a wandb run id, the resumed run continues logging to the same wandb run. Checkpoints are uploaded as wandb artifacts only when ```--wandb_log_checkpoints``` is set.

Synchronous buffer training now follows a single-GPU, stage-by-stage workflow:

1. Sample a block buffer with vLLM.
2. Train from that saved buffer with ```blockwise_power_tb_buffer_train.py```.
3. Repeat for the next block using ```checkpoint_latest/adapter```.

For a maintained end-to-end smoke test with reduced scale but production-aligned core hyperparameters:

```bash
cd llm_experiments
./run_blockwise_buffer_single_smoke.sh
```

Sample block 1:

```bash
CUDA_VISIBLE_DEVICES=0 python blockwise_vllm_sample_buffer.py \
  --data_path ../data/train.parquet \
  --output_dir results/blockwise_buffer_small_qwen_seed0 \
  --model qwen \
  --block_idx 1 \
  --max_examples 32 \
  --block_size 192 \
  --completions_per_prefix 4 \
  --max_completion_tokens 3072 \
  --temperature 0.25 \
  --seed 0 \
  --vllm_batch_size 8 \
  --vllm_enforce_eager
```

Train from the sampled buffer:

```bash
CUDA_VISIBLE_DEVICES=0 python blockwise_power_tb_buffer_train.py \
  --data_path ../data/train.parquet \
  --eval_data_path data/MATH500.json \
  --model qwen \
  --max_examples 32 \
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
  --seed 0 \
  --save_samples \
  --save_every_block \
  --output_dir results/blockwise_buffer_small_qwen_seed0 \
  --skip_buffer_sampling
```

Resume the next block from ```checkpoint_latest``` after sampling ```block_2.csv```:

```bash
CUDA_VISIBLE_DEVICES=0 python blockwise_power_tb_buffer_train.py \
  --data_path ../data/train.parquet \
  --eval_data_path data/MATH500.json \
  --model qwen \
  --max_examples 32 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --gradient_checkpointing \
  --epochs 1 \
  --num_blocks 2 \
  --block_size 192 \
  --completions_per_prefix 4 \
  --max_completion_tokens 3072 \
  --temperature 0.25 \
  --alpha 4.0 \
  --beta 1.0 \
  --lr 1e-5 \
  --seed 0 \
  --save_samples \
  --save_every_block \
  --output_dir results/blockwise_buffer_small_qwen_seed0 \
  --skip_buffer_sampling \
  --resume_from_checkpoint results/blockwise_buffer_small_qwen_seed0/checkpoint_latest
```

For stable checkpoint evaluation, use the eval-only vLLM backend:

```bash
CUDA_VISIBLE_DEVICES=0 python blockwise_power_tb_buffer_train.py \
  --data_path ../data/train.parquet \
  --eval_data_path data/MATH500.json \
  --model qwen \
  --eval_only \
  --eval_backend vllm \
  --eval_every_block \
  --eval_examples 100 \
  --eval_max_new_tokens 3072 \
  --vllm_batch_size 2 \
  --vllm_enforce_eager \
  --output_dir results/blockwise_buffer_small_qwen_seed0 \
  --resume_from_checkpoint results/blockwise_buffer_small_qwen_seed0/checkpoint_latest
```

By default, the trainer scores multiple completions for each prompt in parallel. If this exceeds
memory, add ```--score_micro_batch_size 1``` to score completions one at a time.

## Evaluation
The maintained evaluation path for block-wise buffer checkpoints is the `--eval_only --eval_backend vllm` flow shown above.

Older single-shot grading and pass@k utilities are archived under [`llm_experiments/legacy/`](/Users/zeshenghong/Documents/Codex/2026-06-01/clone-aakaran-reasoning-with-sampling-git/One-Step-Post-Correction/llm_experiments/legacy).

**Single-shot Reasoning**

To grade the responses for single-shot reasoning, collect the .csv files for a given seed run in a folder (e.g. ```results/qwen_math/MATH```) and pass it into ```legacy/eval_math.py```:

```bash
cd llm_experiments
PYTHONPATH=. python legacy/eval_math.py results/qwen_math
```

```legacy/eval_gpqa.py``` is similar, and for ```legacy/eval_he.py```, an additional ```--output_fname``` argument is required, as HumanEval collects all responses in a jsonl file (e.g. ```--output_fname=qwen_math_he```).

For AlpacaEval 2.0, ```legacy/eval_alpaca.py``` collects a ```--folder``` into one json file ```--output_fname```. For evaluating the json file, follow the instructions in the official repo: https://github.com/tatsu-lab/alpaca_eval


**Pass@k Performance**

For pass@k performance, collect the .csv files across seeds in a folder again (e.g. ```results/qwen_math/MATH```) and pass into ```legacy/passk_math.py```:
```bash
python llm_experiments/legacy/passk_math.py --folder=results/qwen_math/MATH
```
The output is a plot of the pass@k performance. As with single-shot reasoning, ```legacy/eval_gpqa.py``` and ```legacy/eval_he.py``` are similar, but for the latter an additional ```--output_fname``` argument is required.
