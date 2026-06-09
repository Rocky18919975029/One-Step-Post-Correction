#!/usr/bin/env bash
#SBATCH --job-name=ospc-ddp-smoke32
#SBATCH --partition=acd_u
#SBATCH --account=d_tonggeh+
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=640G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Same chess/H100 pipeline as submit_ddp_pipeline.slurm, but capped at 32
# training examples and a separate output directory for a quick HPC sanity run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MAX_EXAMPLES="${MAX_EXAMPLES:-32}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/ddp_hpc_smoke32}"

exec bash "$SCRIPT_DIR/submit_ddp_pipeline.slurm"
