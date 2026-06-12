#!/usr/bin/env bash
set -euo pipefail

# Run on a login node with internet access. GPU compute nodes are offline.

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/psamp}"
PEFT_VERSION="${PEFT_VERSION:-0.13.2}"

module purge
module load miniconda3

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

python -m pip install "peft==${PEFT_VERSION}"
python - <<'PY'
import peft
print(f"peft={peft.__version__}", flush=True)
PY
