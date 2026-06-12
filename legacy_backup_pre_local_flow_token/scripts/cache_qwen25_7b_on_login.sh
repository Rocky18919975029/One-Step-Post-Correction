#!/usr/bin/env bash
set -euo pipefail

# Run this on a login node with internet access before submitting Slurm jobs.
# GPU compute nodes run offline and expect these files to already exist.

CONDA_ENV="${CONDA_ENV:-psamp}"
HF_HOME="${HF_HOME:-/data/user/zhongal/.cache/huggingface}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-7B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

module purge
module load miniconda3

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_ENDPOINT
export HF_HUB_DISABLE_XET

python - <<'PY'
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

model_repo = os.environ.get("MODEL_REPO", "Qwen/Qwen2.5-7B")
hf_home = os.environ.get("HF_HOME")
hf_endpoint = os.environ.get("HF_ENDPOINT")
disable_xet = os.environ.get("HF_HUB_DISABLE_XET")

print(f"Downloading {model_repo} into HF_HOME={hf_home}", flush=True)
print(f"Using HF_ENDPOINT={hf_endpoint}", flush=True)
print(f"Using HF_HUB_DISABLE_XET={disable_xet}", flush=True)
path = snapshot_download(
    repo_id=model_repo,
    resume_download=True,
    max_workers=2,
)
print(f"Snapshot ready: {path}", flush=True)

AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
print("Tokenizer ready.", flush=True)
PY

python - <<'PY'
import os
from pathlib import Path

hf_home = Path(os.environ["HF_HOME"])
model_repo = os.environ.get("MODEL_REPO", "Qwen/Qwen2.5-7B")
repo_dir = hf_home / "hub" / ("models--" + model_repo.replace("/", "--"))

if not repo_dir.exists():
    raise SystemExit(f"Missing cache directory: {repo_dir}")

weight_files = list(repo_dir.rglob("*.safetensors")) + list(repo_dir.rglob("*.bin"))
if not weight_files:
    raise SystemExit(f"No model weight files found under {repo_dir}")

print(f"Found {len(weight_files)} weight files under {repo_dir}", flush=True)
PY
