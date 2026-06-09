import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from tqdm import tqdm

from constants import *
from grader_utils.math_grader import grade_answer
from grader_utils.parse_utils import parse_answer
from power_samp_utils import format_prompt


MODEL_NAME_BY_KEY = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen_math_grpo": "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "tulu": "allenai/Llama-3.1-Tulu-3-8B-DPO",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def load_math_dataset(path):
    path = Path(path)
    if not path.exists():
        path = Path(__file__).resolve().parent / path
    if path.suffix == ".json":
        with path.open("r") as f:
            return json.load(f)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return [
            {
                "prompt": str(row["question"]),
                "answer": str(row.get("gt_answer", row.get("answer", ""))),
            }
            for _, row in df.iterrows()
        ]
    raise ValueError(f"Unsupported dataset format: {path}")


def parse_torch_dtype(dtype):
    if dtype == "auto":
        return "auto"
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def load_lora_model(model_name, torch_dtype, device=None, adapter_path=None, attn_implementation=None):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "blockwise_power_tb_train.py requires peft for LoRA training. "
            "Install it in psamp with: pip install peft"
        ) from exc

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    model_kwargs = {
        "torch_dtype": parse_torch_dtype(torch_dtype),
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    ).to(device)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, config)
    if adapter_path is not None:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Resume requires safetensors. Install it with: pip install safetensors") from exc

        adapter_path = Path(adapter_path)
        adapter_weights = load_file(adapter_path / "adapter_model.safetensors")
        model_keys = set(model.state_dict().keys())
        mapped_weights = {}
        unmatched_keys = []
        for key, value in adapter_weights.items():
            candidates = [
                key,
                key.replace(".lora_A.weight", ".lora_A.default.weight"),
                key.replace(".lora_B.weight", ".lora_B.default.weight"),
            ]
            matched_key = next((candidate for candidate in candidates if candidate in model_keys), None)
            if matched_key is None:
                unmatched_keys.append(key)
            else:
                mapped_weights[matched_key] = value

        if not mapped_weights:
            raise RuntimeError(f"No LoRA adapter weights were loaded from {adapter_path}")
        if unmatched_keys:
            print(
                f"Warning: ignored {len(unmatched_keys)} unmatched adapter keys from {adapter_path}",
                flush=True,
            )
        model.load_state_dict(mapped_weights, strict=False)
    model.print_trainable_parameters()
    return model


def first_model_device(model):
    model = unwrap_model(model)
    if hasattr(model, "hf_device_map"):
        for device in model.hf_device_map.values():
            if device not in ("cpu", "disk"):
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                return torch.device(device)
    return next(model.parameters()).device


def enable_gradient_checkpointing(model):
    raw_model = unwrap_model(model)
    if hasattr(raw_model, "config"):
        raw_model.config.use_cache = False
    if hasattr(raw_model, "enable_input_require_grads"):
        raw_model.enable_input_require_grads()
    if hasattr(raw_model, "gradient_checkpointing_enable"):
        try:
            raw_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            raw_model.gradient_checkpointing_enable()


def completion_end(seq, prompt_len, eos_token_id):
    eos_positions = (seq[prompt_len:] == eos_token_id).nonzero(as_tuple=False)
    if len(eos_positions) == 0:
        return len(seq)
    return prompt_len + int(eos_positions[0].item()) + 1


def completion_logprob(model, sequences, prompt_lens, attention_masks, eos_token_id):
    output = model(sequences, attention_mask=attention_masks)
    logits = output.logits[:, :-1, :]
    labels = sequences[:, 1:]
    token_logprobs = F.log_softmax(logits.float(), dim=-1)
    gathered = token_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    losses = []
    for row_idx, prompt_len in enumerate(prompt_lens):
        start = max(prompt_len - 1, 0)
        end = completion_end(sequences[row_idx], prompt_len, eos_token_id)
        losses.append(gathered[row_idx, start:max(end - 1, start)].sum())
    return torch.stack(losses)


def completion_logprob_chunks(model, sequences, prompt_lens, attention_masks, eos_token_id, chunk_size):
    chunks = []
    for start in range(0, sequences.shape[0], chunk_size):
        end = min(start + chunk_size, sequences.shape[0])
        chunks.append(
            completion_logprob(
                model,
                sequences[start:end],
                prompt_lens[start:end],
                attention_masks[start:end],
                eos_token_id,
            )
        )
    return torch.cat(chunks, dim=0)


def sync_cuda_if_available():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sample_continuations(model, tokenizer, prefixes, max_new_tokens, temperature, num_return_sequences):
    model = unwrap_model(model)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    encoded = tokenizer.pad(
        {"input_ids": prefixes},
        padding=True,
        return_tensors="pt",
    )
    input_width = encoded["input_ids"].shape[1]
    left_pad_counts = [input_width - len(prefix) for prefix in prefixes]
    input_device = first_model_device(model)
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    with torch.no_grad():
        original_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
        if hasattr(model, "config"):
            model.config.use_cache = True
        try:
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )
        finally:
            if original_use_cache is not None and hasattr(model, "config"):
                model.config.use_cache = original_use_cache
    prompt_lens = []
    expanded_left_pad_counts = []
    for left_pad_count in left_pad_counts:
        prompt_lens.extend([input_width] * num_return_sequences)
        expanded_left_pad_counts.extend([left_pad_count] * num_return_sequences)

    attention_masks = torch.ones_like(output)
    for row_idx, left_pad_count in enumerate(expanded_left_pad_counts):
        if left_pad_count > 0:
            attention_masks[row_idx, :left_pad_count] = 0
    return output, prompt_lens, attention_masks


def correctness_rewards(tokenizer, sequences, prompt_lens, answers, num_return_sequences):
    rewards = []
    completions = []
    parsed_answers = []
    expanded_answers = []
    for answer in answers:
        expanded_answers.extend([answer] * num_return_sequences)

    for seq, prompt_len, answer in zip(sequences, prompt_lens, expanded_answers):
        completion = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        parsed = parse_answer(completion)
        try:
            reward = float(grade_answer(str(parsed), str(answer)))
        except Exception:
            reward = 0.0
        rewards.append(reward)
        completions.append(completion)
        parsed_answers.append(parsed)
    return (
        torch.tensor(rewards, dtype=torch.float32, device=sequences.device),
        completions,
        parsed_answers,
    )


def score_completion(completion, answer):
    parsed = parse_answer(completion)
    try:
        reward = float(grade_answer(str(parsed), str(answer)))
    except Exception:
        reward = 0.0
    return reward, parsed


def build_prefixes_for_block(model, tokenizer, input_ids_list, block_idx, block_size, temperature):
    if block_idx == 1:
        return input_ids_list

    prefix_new_tokens = (block_idx - 1) * block_size
    sequences, prompt_lens, _ = sample_continuations(
        model,
        tokenizer,
        input_ids_list,
        max_new_tokens=prefix_new_tokens,
        temperature=temperature,
        num_return_sequences=1,
    )
    prefixes = []
    for seq, prompt_len in zip(sequences, prompt_lens):
        end = min(prompt_len + prefix_new_tokens, len(seq))
        prefix = seq[:end].detach().cpu().tolist()
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        while prefix and prefix[0] == pad_token_id:
            prefix = prefix[1:]
        prefixes.append(prefix)
    return prefixes


def vargrad_tb_loss(
    model,
    tokenizer,
    sequences,
    prompt_lens,
    attention_masks,
    rewards,
    alpha,
    beta,
    num_return_sequences,
    score_micro_batch_size=None,
    debug_callback=None,
    precomputed_logp_ref=None,
    before_theta_chunk_callback=None,
):
    def log_phase(message):
        if debug_callback is not None:
            debug_callback(message)

    raw_model = unwrap_model(model)
    if precomputed_logp_ref is None and not hasattr(raw_model, "disable_adapter"):
        raise RuntimeError("Reference logprob requires a PEFT LoRA model with disable_adapter().")

    if score_micro_batch_size is None or score_micro_batch_size >= sequences.shape[0]:
        log_phase("theta forward begin full")
        logp_theta = completion_logprob(model, sequences, prompt_lens, attention_masks, tokenizer.eos_token_id)
        sync_cuda_if_available()
        log_phase("theta forward end full")
        if precomputed_logp_ref is not None:
            logp_ref = precomputed_logp_ref.to(device=logp_theta.device, dtype=torch.float32).detach()
        else:
            with torch.no_grad():
                with raw_model.disable_adapter():
                    log_phase("ref forward begin full")
                    logp_ref = completion_logprob(raw_model, sequences, prompt_lens, attention_masks, tokenizer.eos_token_id)
                    sync_cuda_if_available()
                    log_phase("ref forward end full")

        log_phase("loss assembly begin full")
        log_reward_augmented_target = alpha * logp_ref + rewards / beta
        log_z_terms = alpha * logp_ref - logp_theta.detach() + rewards / beta
        log_z_hat = log_z_terms.view(-1, num_return_sequences).mean(dim=1)
        expanded_log_z_hat = log_z_hat.repeat_interleave(num_return_sequences)

        loss = (
            expanded_log_z_hat.detach()
            + logp_theta
            - log_reward_augmented_target.detach()
        ).pow(2).mean()
        log_phase("loss assembly end full")
        return loss, logp_theta.detach(), logp_ref.detach()

    score_micro_batch_size = max(1, int(score_micro_batch_size))
    if precomputed_logp_ref is not None:
        score_micro_batch_size = max(score_micro_batch_size, int(num_return_sequences))
    if precomputed_logp_ref is not None:
        logp_ref = precomputed_logp_ref.to(device=sequences.device, dtype=torch.float32).detach()
    else:
        with torch.no_grad():
            with raw_model.disable_adapter():
                log_phase(f"ref forward begin chunk_size={score_micro_batch_size}")
                logp_ref = completion_logprob_chunks(
                    raw_model,
                    sequences,
                    prompt_lens,
                    attention_masks,
                    tokenizer.eos_token_id,
                    score_micro_batch_size,
                ).detach()
                sync_cuda_if_available()
                log_phase("ref forward end")

    logp_theta_chunks = []
    for start in range(0, sequences.shape[0], score_micro_batch_size):
        end = min(start + score_micro_batch_size, sequences.shape[0])
        log_phase(f"theta forward begin rows={start}:{end}")
        if before_theta_chunk_callback is not None:
            before_theta_chunk_callback(start, end)
        logp_theta = completion_logprob(
            model,
            sequences[start:end],
            prompt_lens[start:end],
            attention_masks[start:end],
            tokenizer.eos_token_id,
        )
        sync_cuda_if_available()
        log_phase(f"theta forward end rows={start}:{end}")
        logp_theta_chunks.append(logp_theta)

    log_phase("loss assembly begin")
    logp_theta_all = torch.cat(logp_theta_chunks, dim=0)
    log_z_terms = alpha * logp_ref - logp_theta_all.detach() + rewards / beta
    log_z_hat = log_z_terms.view(-1, num_return_sequences).mean(dim=1)
    expanded_log_z_hat = log_z_hat.repeat_interleave(num_return_sequences)

    target = (alpha * logp_ref + rewards / beta).detach()
    residual = expanded_log_z_hat.detach() + logp_theta_all - target
    loss = residual.pow(2).mean()
    log_phase("loss assembly end")
    return loss, logp_theta_all.detach(), logp_ref.detach()


def save_checkpoint(output_dir, model, tokenizer, optimizer, state, distributed):
    if distributed:
        dist.barrier()

    checkpoint_dir = Path(output_dir) / "checkpoint_latest"
    tmp_dir = Path(output_dir) / "checkpoint_latest_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    unwrap_model(model).save_pretrained(tmp_dir / "adapter")
    tokenizer.save_pretrained(tmp_dir / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "state": state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        tmp_dir / "training_state.pt",
    )

    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    tmp_dir.rename(checkpoint_dir)


def move_optimizer_state_to_device(optimizer, device):
    if device is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint_state(checkpoint_dir, optimizer=None, optimizer_device=None):
    checkpoint_dir = Path(checkpoint_dir)
    training_state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if optimizer is not None:
        optimizer.load_state_dict(training_state["optimizer"])
        move_optimizer_state_to_device(optimizer, optimizer_device)

    torch_rng_state = training_state["torch_rng_state"]
    if torch_rng_state.device.type != "cpu":
        torch_rng_state = torch_rng_state.cpu()
    torch.set_rng_state(torch_rng_state)
    if torch.cuda.is_available() and training_state["cuda_rng_state_all"] is not None:
        cuda_rng_state_all = [
            state.cpu() if torch.is_tensor(state) and state.device.type != "cpu" else state
            for state in training_state["cuda_rng_state_all"]
        ]
        visible_device_count = torch.cuda.device_count()
        saved_device_count = len(cuda_rng_state_all)
        if visible_device_count == saved_device_count:
            torch.cuda.set_rng_state_all(cuda_rng_state_all)
        else:
            restore_count = min(visible_device_count, saved_device_count)
            if restore_count == 0:
                print(
                    "Warning: checkpoint contains CUDA RNG state, but no visible CUDA devices are available for restore.",
                    flush=True,
                )
            else:
                print(
                    "Warning: checkpoint CUDA RNG state count "
                    f"({saved_device_count}) does not match visible CUDA device count "
                    f"({visible_device_count}); restoring the first {restore_count} state(s).",
                    flush=True,
                )
                for device_idx in range(restore_count):
                    torch.cuda.set_rng_state(cuda_rng_state_all[device_idx], device_idx)
    np.random.set_state(training_state["numpy_rng_state"])
    random.setstate(training_state["python_rng_state"])
    return training_state["state"]
def maybe_init_wandb(args, rank, resume_state):
    if not args.use_wandb or rank != 0:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install wandb first: pip install wandb") from exc

    run_id = args.wandb_id or (resume_state or {}).get("wandb_id")
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        id=run_id,
        resume=args.wandb_resume,
        config=vars(args),
        settings=wandb.Settings(start_method="thread"),
    )
    return run


def wandb_log_checkpoint(run, checkpoint_dir):
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(f"{run.name or run.id}-checkpoint", type="checkpoint")
    artifact.add_dir(str(checkpoint_dir))
    run.log_artifact(artifact)


def evaluate_model(model, tokenizer, eval_rows, args):
    if not eval_rows:
        return {}

    raw_model = unwrap_model(model)
    had_gradient_checkpointing = bool(
        getattr(raw_model, "is_gradient_checkpointing", False)
        or getattr(getattr(raw_model, "model", None), "gradient_checkpointing", False)
    )
    raw_model.eval()
    if had_gradient_checkpointing and hasattr(raw_model, "gradient_checkpointing_disable"):
        raw_model.gradient_checkpointing_disable()
    rewards = []
    boxed = []
    original_use_cache = getattr(getattr(raw_model, "config", None), "use_cache", None)
    if hasattr(raw_model, "config"):
        raw_model.config.use_cache = True
    try:
        for row in tqdm(eval_rows, desc="eval", leave=False):
            prompt = format_prompt(row["prompt"], args.model, tokenizer, cot=True)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(first_model_device(raw_model))
            attention_mask = torch.ones_like(input_ids)
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": args.eval_max_new_tokens,
                "do_sample": args.eval_do_sample,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if args.eval_do_sample:
                generate_kwargs["temperature"] = args.eval_temperature

            with torch.no_grad():
                output = raw_model.generate(**generate_kwargs)

            completion = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            reward, parsed = score_completion(completion, row["answer"])
            rewards.append(reward)
            boxed.append(parsed is not None)

            del output
            del input_ids
            del attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if original_use_cache is not None and hasattr(raw_model, "config"):
            raw_model.config.use_cache = original_use_cache
        if had_gradient_checkpointing and hasattr(raw_model, "gradient_checkpointing_enable"):
            try:
                raw_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                raw_model.gradient_checkpointing_enable()
        raw_model.train()
    return {
        "eval/accuracy": float(np.mean(rewards)),
        "eval/boxed_rate": float(np.mean(boxed)),
        "eval/examples": len(eval_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--eval_data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--output_dir", type=str, default="results/blockwise_tb")
    parser.add_argument("--model", type=str, default="qwen_math", choices=sorted(MODEL_NAME_BY_KEY))
    parser.add_argument("--max_examples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--score_micro_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument("--completions_per_prefix", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--max_completion_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_every_block", action="store_true")
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="one-step-post-correction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume", type=str, default="allow")
    parser.add_argument("--wandb_log_checkpoints", action="store_true")
    parser.add_argument("--eval_every_block", action="store_true")
    parser.add_argument("--eval_examples", type=int, default=32)
    parser.add_argument("--eval_max_new_tokens", type=int, default=3072)
    parser.add_argument("--eval_temperature", type=float, default=0.25)
    parser.add_argument("--eval_do_sample", action="store_true")
    args = parser.parse_args()
    distributed = False
    rank = 0
    local_rank = 0
    world_size = 1
    distributed_device = None

    if args.completions_per_prefix < 2:
        print(
            "Warning: completions_per_prefix < 2 makes the arithmetic-mean "
            "VarGrad TB residual zero for each prefix. Use at least 2 for training.",
            flush=True,
        )
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    if args.micro_batch_size < 1 or args.micro_batch_size > args.batch_size:
        raise ValueError("--micro_batch_size must be between 1 and --batch_size.")
    if args.score_micro_batch_size is not None and args.score_micro_batch_size < 1:
        raise ValueError("--score_micro_batch_size must be at least 1.")

    seed_everything(args.seed + rank)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_state = None
    adapter_path = None
    if args.resume_from_checkpoint:
        checkpoint_dir = Path(args.resume_from_checkpoint)
        adapter_path = checkpoint_dir / "adapter"
        resume_state = torch.load(
            checkpoint_dir / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )["state"]

    wandb_run = maybe_init_wandb(args, rank, resume_state)
    if rank == 0 and wandb_run is not None and args.wandb_id is None:
        args.wandb_id = wandb_run.id

    model_name = MODEL_NAME_BY_KEY[args.model]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_lora_model(model_name, args.torch_dtype, distributed_device, adapter_path)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    model.train()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )
    start_block_idx = 1
    global_step = 0
    if args.resume_from_checkpoint:
        resume_state = load_checkpoint_state(
            args.resume_from_checkpoint,
            optimizer,
            distributed_device,
        )
        start_block_idx = int(resume_state.get("next_block_idx", 1))
        global_step = int(resume_state.get("global_step", 0))

    dataset = load_math_dataset(args.data_path)[: args.max_examples]
    eval_rows = load_math_dataset(args.eval_data_path)[: args.eval_examples] if args.eval_every_block else []
    prompts = [format_prompt(row["prompt"], args.model, tokenizer, cot=True) for row in dataset]
    answers = [str(row["answer"]) for row in dataset]
    input_ids_list = [tokenizer.encode(prompt) for prompt in prompts]

    metrics = []
    sample_records = []
    eval_records = []
    for block_idx in range(start_block_idx, args.num_blocks + 1):
        for epoch in range(args.epochs):
            order = list(range(len(dataset)))
            random.shuffle(order)
            if distributed:
                remainder = len(order) % world_size
                if remainder:
                    order.extend(order[: world_size - remainder])
                order = order[rank::world_size]
            for start in tqdm(range(0, len(order), args.batch_size), desc=f"block {block_idx} epoch {epoch}"):
                batch_indices = order[start:start + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                step_id = global_step + 1
                total_sequences = len(batch_indices) * args.completions_per_prefix
                loss_sum = 0.0
                reward_sum = 0.0
                logp_theta_sum = 0.0
                logp_ref_sum = 0.0

                for micro_start in range(0, len(batch_indices), args.micro_batch_size):
                    micro_indices = batch_indices[micro_start:micro_start + args.micro_batch_size]
                    batch_inputs = [input_ids_list[idx] for idx in micro_indices]
                    batch_answers = [answers[idx] for idx in micro_indices]

                    prefixes = build_prefixes_for_block(
                        model,
                        tokenizer,
                        batch_inputs,
                        block_idx,
                        args.block_size,
                        args.temperature,
                    )
                    sequences, prompt_lens, attention_masks = sample_continuations(
                        model,
                        tokenizer,
                        prefixes,
                        max_new_tokens=(
                            args.max_completion_tokens
                            if args.max_completion_tokens is not None
                            else max(args.max_new_tokens - (block_idx - 1) * args.block_size, 1)
                        ),
                        temperature=args.temperature,
                        num_return_sequences=args.completions_per_prefix,
                    )
                    rewards, completions, parsed_answers = correctness_rewards(
                        tokenizer,
                        sequences,
                        prompt_lens,
                        batch_answers,
                        args.completions_per_prefix,
                    )

                    loss, logp_theta, logp_ref = vargrad_tb_loss(
                        model,
                        tokenizer,
                        sequences,
                        prompt_lens,
                        attention_masks,
                        rewards,
                        args.alpha,
                        args.beta,
                        args.completions_per_prefix,
                        args.score_micro_batch_size,
                    )
                    micro_sequences = len(micro_indices) * args.completions_per_prefix
                    (loss * (micro_sequences / total_sequences)).backward()
                    loss_sum += float(loss.detach().cpu()) * micro_sequences
                    reward_sum += float(rewards.sum().detach().cpu())
                    logp_theta_sum += float(logp_theta.sum().cpu())
                    logp_ref_sum += float(logp_ref.sum().cpu())

                    if args.save_samples:
                        rewards_cpu = rewards.detach().cpu().tolist()
                        logp_theta_cpu = logp_theta.detach().cpu().tolist()
                        logp_ref_cpu = logp_ref.detach().cpu().tolist()
                        for local_idx, example_idx in enumerate(micro_indices):
                            prefix_text = tokenizer.decode(prefixes[local_idx], skip_special_tokens=True)
                            for sample_idx in range(args.completions_per_prefix):
                                flat_idx = local_idx * args.completions_per_prefix + sample_idx
                                sample_records.append({
                                    "step": step_id,
                                    "epoch": epoch,
                                    "block_idx": block_idx,
                                    "rank": rank,
                                    "example_idx": example_idx,
                                    "sample_idx": sample_idx,
                                    "question": dataset[example_idx]["prompt"],
                                    "correct_answer": answers[example_idx],
                                    "prefix_token_len": len(prefixes[local_idx]),
                                    "prefix_text": prefix_text,
                                    "completion_token_len": int(
                                        completion_end(
                                            sequences[flat_idx],
                                            prompt_lens[flat_idx],
                                            tokenizer.eos_token_id,
                                        ) - prompt_lens[flat_idx]
                                    ),
                                    "completion": completions[flat_idx],
                                    "parsed_answer": parsed_answers[flat_idx],
                                    "has_boxed_answer": parsed_answers[flat_idx] is not None,
                                    "reward": rewards_cpu[flat_idx],
                                    "logp_theta": logp_theta_cpu[flat_idx],
                                    "logp_ref": logp_ref_cpu[flat_idx],
                                })

                optimizer.step()

                global_step = step_id
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "block_idx": block_idx,
                    "rank": rank,
                    "loss": loss_sum / total_sequences,
                    "reward_mean": reward_sum / total_sequences,
                    "logp_theta_mean": logp_theta_sum / total_sequences,
                    "logp_ref_mean": logp_ref_sum / total_sequences,
                }
                metrics.append(record)
                print(record, flush=True)
                if wandb_run is not None:
                    wandb_run.log(record, step=global_step)

        if args.save_every_block:
            block_dir = output_dir / f"block_{block_idx}"
            unwrap_model(model).save_pretrained(block_dir)
            tokenizer.save_pretrained(block_dir)

        if args.eval_every_block:
            eval_metrics = evaluate_model(model, tokenizer, eval_rows, args)
            eval_metrics = {**eval_metrics, "block_idx": block_idx, "step": global_step}
            eval_records.append(eval_metrics)
            print(eval_metrics, flush=True)
            if wandb_run is not None:
                wandb_run.log(eval_metrics, step=global_step)

        checkpoint_state = {
            "next_block_idx": block_idx + 1,
            "global_step": global_step,
            "wandb_id": wandb_run.id if wandb_run is not None else args.wandb_id,
            "args": vars(args),
        }
        save_checkpoint(output_dir, model, tokenizer, optimizer, checkpoint_state, distributed=False)
        if args.wandb_log_checkpoints:
            wandb_log_checkpoint(wandb_run, output_dir / "checkpoint_latest")

    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    if args.save_samples:
        pd.DataFrame(sample_records).to_csv(output_dir / "samples.csv", index=False)
    if eval_records:
        pd.DataFrame(eval_records).to_csv(output_dir / "eval_metrics.csv", index=False)

    unwrap_model(model).save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
