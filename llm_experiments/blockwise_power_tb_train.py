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


def resolve_model_name(model):
    return MODEL_NAME_BY_KEY.get(model, model)


def resolve_prompt_model_key(model, prompt_model=None):
    if prompt_model is not None:
        return prompt_model
    if model in MODEL_NAME_BY_KEY:
        return model
    return "qwen"


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


def capture_rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    device = torch.device(device)
    if torch.cuda.is_available() and device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    device = torch.device(device)
    if "cuda" in state and torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def sync_cuda_if_available():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def score_completion(completion, answer):
    parsed = parse_answer(completion)
    try:
        reward = float(grade_answer(str(parsed), str(answer)))
    except Exception:
        reward = 0.0
    return reward, parsed


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
):
    raw_model = unwrap_model(model)
    if not hasattr(raw_model, "disable_adapter"):
        raise RuntimeError("Reference logprob requires a PEFT LoRA model with disable_adapter().")

    if score_micro_batch_size is None or score_micro_batch_size >= sequences.shape[0]:
        logp_theta = completion_logprob(model, sequences, prompt_lens, attention_masks, tokenizer.eos_token_id)
        with torch.no_grad():
            with raw_model.disable_adapter():
                logp_ref = completion_logprob(raw_model, sequences, prompt_lens, attention_masks, tokenizer.eos_token_id)

        log_reward_augmented_target = alpha * logp_ref + rewards / beta
        log_z_terms = alpha * logp_ref - logp_theta.detach() + rewards / beta
        log_z_hat = log_z_terms.view(-1, num_return_sequences).mean(dim=1)
        expanded_log_z_hat = log_z_hat.repeat_interleave(num_return_sequences)

        loss = (
            expanded_log_z_hat.detach()
            + logp_theta
            - log_reward_augmented_target.detach()
        ).pow(2).mean()
        return loss, logp_theta.detach(), logp_ref.detach()

    score_micro_batch_size = max(1, int(score_micro_batch_size))
    with torch.no_grad():
        with raw_model.disable_adapter():
            logp_ref = completion_logprob_chunks(
                raw_model,
                sequences,
                prompt_lens,
                attention_masks,
                tokenizer.eos_token_id,
                score_micro_batch_size,
            ).detach()

    logp_theta_chunks = []
    for start in range(0, sequences.shape[0], score_micro_batch_size):
        end = min(start + score_micro_batch_size, sequences.shape[0])
        logp_theta = completion_logprob(
            model,
            sequences[start:end],
            prompt_lens[start:end],
            attention_masks[start:end],
            tokenizer.eos_token_id,
        )
        logp_theta_chunks.append(logp_theta)

    logp_theta_all = torch.cat(logp_theta_chunks, dim=0)
    log_z_terms = alpha * logp_ref - logp_theta_all.detach() + rewards / beta
    log_z_hat = log_z_terms.view(-1, num_return_sequences).mean(dim=1)
    expanded_log_z_hat = log_z_hat.repeat_interleave(num_return_sequences)

    target = (alpha * logp_ref + rewards / beta).detach()
    residual = expanded_log_z_hat.detach() + logp_theta_all - target
    loss = residual.pow(2).mean()
    return loss, logp_theta_all.detach(), logp_ref.detach()


def vargrad_tb_loss_with_score_chunk_backward(
    model,
    tokenizer,
    sequences,
    prompt_lens,
    attention_masks,
    rewards,
    alpha,
    beta,
    num_return_sequences,
    score_micro_batch_size,
    loss_scale,
    active_callback=None,
):
    raw_model = unwrap_model(model)
    if not hasattr(raw_model, "disable_adapter"):
        raise RuntimeError("Reference logprob requires a PEFT LoRA model with disable_adapter().")

    row_count = int(sequences.shape[0])
    score_micro_batch_size = max(1, int(score_micro_batch_size or row_count))
    chunk_ranges = [
        (start, min(start + score_micro_batch_size, row_count))
        for start in range(0, row_count, score_micro_batch_size)
    ]

    if active_callback is not None:
        active_callback("ref_forward", 0, row_count)
    with torch.no_grad():
        with raw_model.disable_adapter():
            logp_ref = completion_logprob_chunks(
                raw_model,
                sequences,
                prompt_lens,
                attention_masks,
                tokenizer.eos_token_id,
                score_micro_batch_size,
            ).detach()

    theta_for_z_chunks = []
    chunk_rng_states = []
    for start, end in chunk_ranges:
        if active_callback is not None:
            active_callback("theta_for_z_forward", start, end)
        chunk_rng_states.append(capture_rng_state(sequences.device))
        with torch.no_grad():
            theta_for_z_chunks.append(
                completion_logprob(
                    model,
                    sequences[start:end],
                    prompt_lens[start:end],
                    attention_masks[start:end],
                    tokenizer.eos_token_id,
                ).detach()
            )

    logp_theta_for_z = torch.cat(theta_for_z_chunks, dim=0)
    log_z_terms = alpha * logp_ref - logp_theta_for_z + rewards / beta
    log_z_hat = log_z_terms.view(-1, num_return_sequences).mean(dim=1)
    expanded_log_z_hat = log_z_hat.repeat_interleave(num_return_sequences).detach()
    target = (alpha * logp_ref + rewards / beta).detach()
    detached_residual = expanded_log_z_hat + logp_theta_for_z - target
    detached_loss = detached_residual.pow(2).mean()

    for (start, end), rng_state in zip(chunk_ranges, chunk_rng_states):
        if active_callback is not None:
            active_callback("theta_grad_forward", start, end)
        restore_rng_state(rng_state, sequences.device)
        logp_theta = completion_logprob(
            model,
            sequences[start:end],
            prompt_lens[start:end],
            attention_masks[start:end],
            tokenizer.eos_token_id,
        )
        residual = expanded_log_z_hat[start:end] + logp_theta - target[start:end]
        chunk_loss = residual.pow(2).sum() / row_count
        if active_callback is not None:
            active_callback("theta_backward", start, end)
        (chunk_loss * loss_scale).backward()
        sync_cuda_if_available()

    if active_callback is not None:
        active_callback("complete", 0, row_count)
    return detached_loss.detach(), logp_theta_for_z.detach(), logp_ref.detach()


def save_checkpoint(output_dir, model, tokenizer, optimizer, state, distributed):
    if distributed:
        dist.barrier()

    checkpoint_dir = Path(output_dir) / "checkpoint_latest"
    tmp_dir = Path(output_dir) / "checkpoint_latest_tmp"
    is_rank0 = not distributed or dist.get_rank() == 0
    if not is_rank0:
        if distributed:
            dist.barrier()
        return

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
    if distributed:
        dist.barrier()


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
            prompt_model = resolve_prompt_model_key(args.model, getattr(args, "prompt_model", None))
            prompt = format_prompt(row["prompt"], prompt_model, tokenizer, cot=True)
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
