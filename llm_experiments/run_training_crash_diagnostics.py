import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def run_case(name, command, env, log_dir):
    log_path = log_dir / f"{name}.log"
    print(f"\n===== {name} =====", flush=True)
    print(" ".join(command), flush=True)
    start = time.time()
    with log_path.open("w") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        proc = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - start
    status = "PASS" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"{name}: {status} in {elapsed:.1f}s; log={log_path}", flush=True)
    return {
        "name": name,
        "returncode": proc.returncode,
        "elapsed": elapsed,
        "log_path": log_path,
    }


def trainer_command(args, output_dir, *, name, gradient_checkpointing=True, micro_batch_size=2,
                    save_samples=False, save_every_steps=0, disable_tqdm=False,
                    disable_micro_batch_dump=False):
    command = [
        sys.executable,
        "blockwise_power_tb_buffer_train.py",
        "--data_path",
        args.data_path,
        "--eval_data_path",
        args.eval_data_path,
        "--model",
        args.model,
        "--max_examples",
        str(args.max_examples),
        "--batch_size",
        str(args.batch_size),
        "--micro_batch_size",
        str(micro_batch_size),
        "--score_micro_batch_size",
        str(args.score_micro_batch_size),
        "--epochs",
        "1",
        "--num_blocks",
        "1",
        "--block_size",
        str(args.block_size),
        "--completions_per_prefix",
        str(args.completions_per_prefix),
        "--max_completion_tokens",
        str(args.max_completion_tokens),
        "--temperature",
        str(args.temperature),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--skip_buffer_sampling",
        "--output_dir",
        str(output_dir),
        "--debug_dump_timeout_seconds",
        str(args.debug_dump_timeout_seconds),
        "--attn_implementation",
        args.attn_implementation,
        "--max_train_steps",
        str(args.max_train_steps),
    ]
    if gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if save_samples:
        command.append("--save_samples")
    if save_every_steps:
        command.extend(["--save_every_steps", str(save_every_steps)])
    if disable_tqdm:
        command.append("--disable_tqdm")
    if disable_micro_batch_dump:
        command.append("--disable_micro_batch_debug_dump")
    return name, command


def prepare_trainer_output(output_dir, source_buffer):
    buffer_dir = output_dir / "buffers"
    buffer_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_buffer, buffer_dir / "block_1.csv")


def main():
    parser = argparse.ArgumentParser(description="Run a bounded matrix of blockwise training crash diagnostics.")
    parser.add_argument("--source_output_dir", type=str, default="results/blockwise_buffer_mini500_qwen_seed0")
    parser.add_argument("--diag_output_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="../data/mini_train.parquet")
    parser.add_argument("--eval_data_path", type=str, default="data/MATH500.json")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--score_micro_batch_size", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--completions_per_prefix", type=int, default=4)
    parser.add_argument("--max_completion_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--max_train_steps", type=int, default=30)
    parser.add_argument("--max_replay_micro_batches", type=int, default=40)
    parser.add_argument("--debug_dump_timeout_seconds", type=int, default=60)
    parser.add_argument("--cuda_visible_devices", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = parser.parse_args()

    source_output_dir = Path(args.source_output_dir)
    if not source_output_dir.is_absolute():
        source_output_dir = SCRIPT_DIR / source_output_dir
    source_buffer = source_output_dir / "buffers" / "block_1.csv"
    source_checkpoint = source_output_dir / "checkpoint_latest"
    source_micro_glob = source_output_dir / "debug_logs" / "micro_batches" / "block1_step*_micro*.csv"

    if not source_buffer.exists():
        raise FileNotFoundError(f"Missing source buffer: {source_buffer}")
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {source_checkpoint}")

    if args.diag_output_dir:
        diag_output_dir = Path(args.diag_output_dir)
        if not diag_output_dir.is_absolute():
            diag_output_dir = SCRIPT_DIR / diag_output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        diag_output_dir = SCRIPT_DIR / "results" / f"training_crash_diagnostics_{stamp}"

    diag_log_dir = diag_output_dir / "case_logs"
    diag_log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cases = []
    replay_cmd = [
        sys.executable,
        "replay_blockwise_micro_batch.py",
        "--checkpoint_dir",
        str(source_checkpoint),
        "--micro_batch_glob",
        str(source_micro_glob),
        "--max_micro_batches",
        str(args.max_replay_micro_batches),
        "--fresh_model",
        "--run_backward",
        "--run_optimizer_step",
        "--step_every_micro_batches",
        "2",
        "--attn_implementation",
        args.attn_implementation,
    ]
    cases.append(("replay_fresh_per_step", replay_cmd))

    trainer_minimal_gc_dir = diag_output_dir / "trainer_minimal_gc"
    prepare_trainer_output(trainer_minimal_gc_dir, source_buffer)
    cases.append(trainer_command(
        args,
        trainer_minimal_gc_dir,
        name="trainer_minimal_gc",
        gradient_checkpointing=True,
        micro_batch_size=args.micro_batch_size,
        disable_tqdm=True,
        disable_micro_batch_dump=True,
    ))
    trainer_minimal_no_gc_dir = diag_output_dir / "trainer_minimal_no_gc"
    prepare_trainer_output(trainer_minimal_no_gc_dir, source_buffer)
    cases.append(trainer_command(
        args,
        trainer_minimal_no_gc_dir,
        name="trainer_minimal_no_gc",
        gradient_checkpointing=False,
        micro_batch_size=args.micro_batch_size,
        disable_tqdm=True,
        disable_micro_batch_dump=True,
    ))
    trainer_micro1_gc_dir = diag_output_dir / "trainer_micro1_gc"
    prepare_trainer_output(trainer_micro1_gc_dir, source_buffer)
    cases.append(trainer_command(
        args,
        trainer_micro1_gc_dir,
        name="trainer_micro1_gc",
        gradient_checkpointing=True,
        micro_batch_size=1,
        disable_tqdm=True,
        disable_micro_batch_dump=True,
    ))
    trainer_with_dump_tqdm_gc_dir = diag_output_dir / "trainer_with_dump_tqdm_gc"
    prepare_trainer_output(trainer_with_dump_tqdm_gc_dir, source_buffer)
    cases.append(trainer_command(
        args,
        trainer_with_dump_tqdm_gc_dir,
        name="trainer_with_dump_tqdm_gc",
        gradient_checkpointing=True,
        micro_batch_size=args.micro_batch_size,
        disable_tqdm=False,
        disable_micro_batch_dump=False,
    ))
    trainer_save_steps_gc_dir = diag_output_dir / "trainer_save_steps_gc"
    prepare_trainer_output(trainer_save_steps_gc_dir, source_buffer)
    cases.append(trainer_command(
        args,
        trainer_save_steps_gc_dir,
        name="trainer_save_steps_gc",
        gradient_checkpointing=True,
        micro_batch_size=args.micro_batch_size,
        save_every_steps=5,
        disable_tqdm=True,
        disable_micro_batch_dump=True,
    ))

    print(f"diagnostic output: {diag_output_dir}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    results = []
    for name, command in cases:
        results.append(run_case(name, command, env, diag_log_dir))

    print("\n===== summary =====", flush=True)
    for result in results:
        status = "PASS" if result["returncode"] == 0 else f"FAIL({result['returncode']})"
        print(f"{result['name']}: {status} ({result['elapsed']:.1f}s) {result['log_path']}", flush=True)

    summary_path = diag_output_dir / "summary.tsv"
    with summary_path.open("w") as f:
        f.write("name\treturncode\telapsed_seconds\tlog_path\n")
        for result in results:
            f.write(
                f"{result['name']}\t{result['returncode']}\t{result['elapsed']:.3f}\t{result['log_path']}\n"
            )
    print(f"summary written: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
