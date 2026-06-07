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
The llm_experiments folder contains slurm scripts to run power sampling for MATH500 (```power_samp_math.py```), whose .json is included in llm_experiments/data, as well as HumanEval (```power_samp_he.py```), GPQA Diamond (```power_samp_gpqa.py```), and AlpacaEval 2.0 (```power_samp_alpaca.py```), whose corresponding eval sets can be downloaded from their official repos. 

To run power sampling on MATH500 with 8 seeds and the eval set split across 5 shards:
```bash
sbatch llm_experiments/scripts/power_samp_math.sh
```
The output is several .csv files (based on the shard and seed number) that store the response outputs, correct answers, original prompts, etc. 

To run all five MATH500 shards across multiple GPUs and merge the results:

```bash
cd llm_experiments
python run_math_multi_gpu.py --seed 0 --num_gpus 5
```

To select specific physical GPU IDs:

```bash
python run_math_multi_gpu.py --seed 0 --gpus 0,2,4,6
```

Each GPU runs at most one shard at a time. Per-shard logs and CSV files are stored under ```results/```, followed by one merged 500-row CSV.

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

For multi-GPU block-wise training, launch with DDP:

```bash
python run_blockwise_multi_gpu.py --gpus 0,1,2,3 -- \
  --model qwen_math \
  --max_examples 32 \
  --num_blocks 16 \
  --completions_per_prefix 2 \
  --max_completion_tokens 512 \
  --save_samples \
  --save_every_block \
  --output_dir results/blockwise_tb_ddp_32x16
```

Each GPU handles different prompts while synchronizing LoRA gradients. Rank-specific files are written as ```metrics_rank*.csv``` and ```samples_rank*.csv```, then rank 0 merges them into ```metrics.csv``` and ```samples.csv```.

Optional experiment tracking, checkpoint resume, and block-end evaluation:

```bash
python run_blockwise_multi_gpu.py --gpus 0,1,2,3 -- \
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
  --output_dir results/blockwise_tb_ddp_32x16
```

The latest stage-boundary checkpoint is stored in ```checkpoint_latest```. Resume with:

```bash
python run_blockwise_multi_gpu.py --gpus 0,1,2,3 -- \
  --resume_from_checkpoint results/blockwise_tb_ddp_32x16/checkpoint_latest \
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
  --output_dir results/blockwise_tb_ddp_32x16
```

If the checkpoint has a wandb run id, the resumed run continues logging to the same wandb run. Checkpoints are uploaded as wandb artifacts only when ```--wandb_log_checkpoints``` is set.

Synchronous buffer training decouples stage sampling from training. Rank 0 first samples a stage
buffer with vLLM, then all DDP ranks train from the saved completions:

```bash
python run_blockwise_buffer_staged.py --gpus 0,1 -- \
  --data_path ../data/train.parquet \
  --eval_data_path data/MATH500.json \
  --model qwen \
  --max_examples 32 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --gradient_checkpointing \
  --epochs 1 \
  --num_blocks 3 \
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
  --eval_every_block \
  --eval_examples 100 \
  --eval_max_new_tokens 3072 \
  --vllm_batch_size 8 \
  --vllm_visible_devices 0 \
  --vllm_enforce_eager \
  --ddp_timeout_minutes 120 \
  --use_wandb \
  --wandb_run_name buffer-small-qwen-seed0 \
  --output_dir results/blockwise_buffer_small_qwen_seed0
```

By default, the trainer scores multiple completions for each prompt in parallel. If this exceeds
memory, add ```--score_micro_batch_size 1``` to score completions one at a time.

## Evaluation
**Single-shot Reasoning**

To grade the responses for single-shot reasoning, collect the .csv files for a given seed run in a folder (e.g. ```results/qwen_math/MATH```) and pass it into ```eval_math.py```:

```bash
cd llm_experiments
PYTHONPATH=. python eval_math.py results/qwen_math
```

```eval_gpqa.py``` is similar, and for ```eval_he.py```, an additional ```--output_fname``` argument is required, as HumanEval collects all responses in a jsonl file (e.g. ```--output_fname=qwen_math_he```).

For AlpacaEval 2.0, ```eval_alpaca.py``` collects a ```--folder``` into one json file ```--output_fname```. For evaluating the json file, follow the instructions in the official repo: https://github.com/tatsu-lab/alpaca_eval


**Pass@k Performance**

For pass@k performance, collect the .csv files across seeds in a folder again (e.g. ```results/qwen_math/MATH```) and pass into ```passk_math.py```:
```bash
python llm_experiments/passk_math.py --folder=results/qwen_math/MATH
```
The output is a plot of the pass@k performance. As with single-shot reasoning, ```eval_gpqa.py``` and ```eval_he.py``` are similar, but for the latter an additional ```--output_fname``` argument is required.
