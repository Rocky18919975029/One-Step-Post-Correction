import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from constants import *  # noqa: F401,F403
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


def cached_snapshot_for_repo(repo_id):
    cache_roots = []
    for env_name in ["HF_HUB_CACHE", "TRANSFORMERS_CACHE"]:
        value = os.environ.get(env_name)
        if value:
            cache_roots.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home).expanduser() / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    cache_dir_name = "models--" + repo_id.replace("/", "--")
    for cache_root in cache_roots:
        snapshots_dir = cache_root / cache_dir_name / "snapshots"
        if not snapshots_dir.exists():
            continue
        candidates = [
            path
            for path in snapshots_dir.iterdir()
            if path.is_dir() and (path / "config.json").exists()
        ]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return str(candidates[0])
    return None


def infer_repo_id_from_missing_path(model):
    name = Path(model).name.lower()
    compact = name.replace("_", "-")
    if "qwen2.5-math-7b" in compact:
        return "Qwen/Qwen2.5-Math-7B"
    if "qwen2.5-7b" in compact:
        return "Qwen/Qwen2.5-7B"
    if "qwen2.5-0.5b" in compact:
        return "Qwen/Qwen2.5-0.5B"
    return None


def resolve_model_name(model):
    if Path(str(model)).expanduser().exists():
        return str(Path(str(model)).expanduser())

    repo_id = MODEL_NAME_BY_KEY.get(model)
    if repo_id is None:
        repo_id = infer_repo_id_from_missing_path(str(model))
    if repo_id is None and "/" in str(model) and not str(model).startswith("/"):
        repo_id = str(model)

    if repo_id is not None:
        cached_snapshot = cached_snapshot_for_repo(repo_id)
        if cached_snapshot is not None:
            return cached_snapshot
        return repo_id

    return model


def describe_model_resolution(model):
    resolved = resolve_model_name(model)
    if resolved != model:
        return f"{model} -> {resolved}"
    return str(model)


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
    return getattr(model, "module", model)


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
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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


def load_reference_model(model_name, torch_dtype, device=None, attn_implementation=None):
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model_kwargs = {
        "torch_dtype": parse_torch_dtype(torch_dtype),
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if hasattr(model, "config"):
        model.config.use_cache = False
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

    losses = []
    for row_idx, prompt_len in enumerate(prompt_lens):
        start = max(prompt_len - 1, 0)
        end = completion_end(sequences[row_idx], prompt_len, eos_token_id)
        slice_end = max(end - 1, start)
        row_logits = logits[row_idx, start:slice_end]
        if row_logits.numel() == 0:
            losses.append(row_logits.sum())
            continue
        row_labels = labels[row_idx, start:slice_end]
        row_logprobs = F.log_softmax(row_logits.float(), dim=-1)
        losses.append(row_logprobs.gather(-1, row_labels.unsqueeze(-1)).squeeze(-1).sum())
    return torch.stack(losses)


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


def maybe_init_wandb(args, rank, resume_state):
    if not args.use_wandb or rank != 0:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install wandb first: pip install wandb") from exc

    resume_mode = args.wandb_resume or "allow"
    run_id = args.wandb_id
    if run_id is None and resume_mode != "never":
        run_id = (resume_state or {}).get("wandb_id")
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        id=run_id,
        resume=resume_mode,
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
    finally:
        if hasattr(raw_model, "config"):
            raw_model.config.use_cache = original_use_cache
        if had_gradient_checkpointing and hasattr(raw_model, "gradient_checkpointing_enable"):
            try:
                raw_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                raw_model.gradient_checkpointing_enable()

    return {
        "eval/accuracy": float(np.mean(rewards)),
        "eval/boxed_rate": float(np.mean(boxed)),
        "eval/examples": len(eval_rows),
    }
