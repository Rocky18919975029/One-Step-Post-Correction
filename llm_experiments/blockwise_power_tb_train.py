import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


def load_lora_model(model_name, torch_dtype):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "blockwise_power_tb_train.py requires peft for LoRA training. "
            "Install it in psamp with: pip install peft"
        ) from exc

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=parse_torch_dtype(torch_dtype),
        device_map="auto",
        trust_remote_code=True,
    )
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
    model.print_trainable_parameters()
    return model


def first_model_device(model):
    if hasattr(model, "hf_device_map"):
        for device in model.hf_device_map.values():
            if device not in ("cpu", "disk"):
                if isinstance(device, int):
                    return torch.device(f"cuda:{device}")
                return torch.device(device)
    return next(model.parameters()).device


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


def sample_continuations(model, tokenizer, prefixes, max_new_tokens, temperature, num_return_sequences):
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
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
        )
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


def vargrad_tb_loss(model, tokenizer, sequences, prompt_lens, attention_masks, rewards, alpha, beta, num_return_sequences):
    logp_theta = completion_logprob(model, sequences, prompt_lens, attention_masks, tokenizer.eos_token_id)

    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("Reference logprob requires a PEFT LoRA model with disable_adapter().")
    with torch.no_grad():
        with model.disable_adapter():
            logp_ref = completion_logprob(model, sequences, prompt_lens, attention_masks, tokenizer.eos_token_id)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--output_dir", type=str, default="results/blockwise_tb")
    parser.add_argument("--model", type=str, default="qwen_math", choices=sorted(MODEL_NAME_BY_KEY))
    parser.add_argument("--max_examples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
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
    parser.add_argument("--save_every_block", action="store_true")
    parser.add_argument("--save_samples", action="store_true")
    args = parser.parse_args()

    if args.completions_per_prefix < 2:
        print(
            "Warning: completions_per_prefix < 2 makes the arithmetic-mean "
            "VarGrad TB residual zero for each prefix. Use at least 2 for training.",
            flush=True,
        )

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_NAME_BY_KEY[args.model]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_lora_model(model_name, args.torch_dtype)
    model.train()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
    )

    dataset = load_math_dataset(args.data_path)[: args.max_examples]
    prompts = [format_prompt(row["prompt"], args.model, tokenizer, cot=True) for row in dataset]
    answers = [str(row["answer"]) for row in dataset]
    input_ids_list = [tokenizer.encode(prompt) for prompt in prompts]

    metrics = []
    sample_records = []
    global_step = 0
    for epoch in range(args.epochs):
        order = list(range(len(dataset)))
        random.shuffle(order)
        for block_idx in range(1, args.num_blocks + 1):
            for start in tqdm(range(0, len(order), args.batch_size), desc=f"epoch {epoch} block {block_idx}"):
                batch_indices = order[start:start + args.batch_size]
                batch_inputs = [input_ids_list[idx] for idx in batch_indices]
                batch_answers = [answers[idx] for idx in batch_indices]

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
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                global_step += 1
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "block_idx": block_idx,
                    "loss": float(loss.detach().cpu()),
                    "reward_mean": float(rewards.mean().detach().cpu()),
                    "logp_theta_mean": float(logp_theta.mean().cpu()),
                    "logp_ref_mean": float(logp_ref.mean().cpu()),
                }
                metrics.append(record)
                print(record, flush=True)

                if args.save_samples:
                    rewards_cpu = rewards.detach().cpu().tolist()
                    logp_theta_cpu = logp_theta.detach().cpu().tolist()
                    logp_ref_cpu = logp_ref.detach().cpu().tolist()
                    for local_idx, example_idx in enumerate(batch_indices):
                        prefix_text = tokenizer.decode(prefixes[local_idx], skip_special_tokens=True)
                        for sample_idx in range(args.completions_per_prefix):
                            flat_idx = local_idx * args.completions_per_prefix + sample_idx
                            sample_records.append({
                                "step": global_step,
                                "epoch": epoch,
                                "block_idx": block_idx,
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

            if args.save_every_block:
                block_dir = output_dir / f"epoch_{epoch}_block_{block_idx}"
                model.save_pretrained(block_dir)
                tokenizer.save_pretrained(block_dir)

    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    if args.save_samples:
        pd.DataFrame(sample_records).to_csv(output_dir / "samples.csv", index=False)


if __name__ == "__main__":
    main()
