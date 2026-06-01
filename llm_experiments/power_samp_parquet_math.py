import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from grader_utils.parse_utils import parse_answer
from power_samp_utils import AutoregressiveSampler, format_prompt, mcmc_power_samp


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


def get_prompt_from_parquet_cell(prompt_cell):
    if prompt_cell is None:
        return None
    if isinstance(prompt_cell, np.ndarray):
        prompt_cell = prompt_cell.tolist()
    if isinstance(prompt_cell, list) and prompt_cell:
        first = prompt_cell[0]
        if isinstance(first, dict) and "content" in first:
            return first["content"]
    if isinstance(prompt_cell, dict) and "content" in prompt_cell:
        return prompt_cell["content"]
    if isinstance(prompt_cell, str):
        return prompt_cell
    return None


def build_input_text(row, model, tokenizer, cot, prompt_source):
    if prompt_source == "prompt" and "prompt" in row:
        prompt = get_prompt_from_parquet_cell(row["prompt"])
        if prompt:
            return prompt
    return format_prompt(str(row["question"]), model, tokenizer, cot)


def decode_ids(tokenizer, ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().view(-1).tolist()
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/train.parquet")
    parser.add_argument("--save_str", type=str, default="results/parquet_train")
    parser.add_argument("--model", type=str, default="qwen", choices=sorted(MODEL_NAME_BY_KEY))
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--cot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--block_num", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_source", type=str, default="question", choices=["question", "prompt"])
    args = parser.parse_args()

    if args.max_new_tokens % args.block_num != 0:
        raise ValueError("--max_new_tokens must be divisible by --block_num")

    seed_everything(args.seed)

    save_dir = os.path.join(args.save_str, args.model)
    os.makedirs(save_dir, exist_ok=True)

    dataset = pd.read_parquet(args.data_path)
    if args.max_examples is not None:
        dataset = dataset.head(args.max_examples)

    start = args.shard_size * args.batch_idx
    end = min(args.shard_size * (args.batch_idx + 1), len(dataset))
    shard = dataset.iloc[start:end].reset_index(drop=False).rename(columns={"index": "problem_id"})

    model_str = MODEL_NAME_BY_KEY[args.model]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).to(args.device)
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, args.device)

    results = []
    for _, row in tqdm(shard.iterrows(), total=len(shard), desc="Benchmark on parquet math"):
        question = str(row["question"])
        correct_answer = str(row.get("gt_answer", row.get("answer", "")))
        input_text = build_input_text(row, args.model, tokenizer, args.cot, args.prompt_source)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(args.device)
        prefix = input_ids[0].detach().cpu().tolist()

        naive_temp_output = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            temperature=args.temperature,
        )
        std_output = hf_model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
        )
        mcmc_output, _, _, acceptance_ratio = mcmc_power_samp(
            autoreg_sampler,
            prefix,
            args.temperature,
            args.mcmc_steps,
            max_new_tokens=args.max_new_tokens,
            block_num=args.block_num,
        )

        naive_generated_ids = naive_temp_output.sequences[0][len(prefix):]
        std_generated_ids = std_output.sequences[0][len(prefix):]
        mcmc_generated_ids = mcmc_output[len(prefix):]

        naive_completion = decode_ids(tokenizer, naive_generated_ids)
        std_completion = decode_ids(tokenizer, std_generated_ids)
        mcmc_completion = decode_ids(tokenizer, mcmc_generated_ids)

        results.append({
            "problem_id": int(row["problem_id"]),
            "question": question,
            "correct_answer": correct_answer,
            "naive_completion": naive_completion,
            "naive_answer": parse_answer(naive_completion),
            "std_completion": std_completion,
            "std_answer": parse_answer(std_completion),
            "mcmc_completion": mcmc_completion,
            "mcmc_answer": parse_answer(mcmc_completion),
            "acceptance_ratio": acceptance_ratio,
        })

    out_name = (
        f"{args.model}_parquet_math_power_samp_results_"
        f"{args.mcmc_steps}_{args.temperature}_{args.batch_idx}_{args.seed}.csv"
    )
    pd.DataFrame(results).to_csv(os.path.join(save_dir, out_name), index=False)


if __name__ == "__main__":
    main()
