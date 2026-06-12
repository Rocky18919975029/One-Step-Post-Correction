#!/usr/bin/env bash
#SBATCH --job-name=ospc-ddp-smoke32
#SBATCH --partition=acd_u
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
# Slurm executes a copied script from its spool directory, so resolve the shared
# script through REPO_DIR instead of BASH_SOURCE.

export REPO_DIR="${REPO_DIR:-/data/user/zhongal/One-Step-Post-Correction}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-32}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/ddp_hpc_smoke32}"

exec bash "$REPO_DIR/scripts/submit_ddp_pipeline.slurm"
