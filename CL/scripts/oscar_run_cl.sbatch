#!/bin/bash
#
# Example OSCAR Slurm script for COOM continual learning runs.
#
# Usage:
#   sbatch CL/scripts/oscar_run_cl.sbatch
#
# Then watch:
#   squeue -u $USER
#   tail -f logs/oscar/%x_%j.out
#
# Notes:
# - You must edit the "Environment setup" section to match your OSCAR modules/venv.
# - ViZDoom typically needs a working OpenGL/Mesa setup on the cluster; prefer running without rendering.
#

#SBATCH --job-name=coom_hace
#SBATCH --output=logs/oscar/%x_%j.out
#SBATCH --error=logs/oscar/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

echo "Job started: $(date)"
echo "Hostname: $(hostname)"
echo "CWD: $(pwd)"

# Ensure log directory exists (safe if already exists)
mkdir -p logs/oscar

# ---- Environment setup (EDIT ME) ----
# Option A: load modules + activate venv
module purge
module load python/3.10
# module load cuda/12.1  # uncomment if your env needs it explicitly

# If you created a venv in your home/scratch:
# source ~/venvs/coom/bin/activate

# Or if you use conda:
# module load anaconda/2023.09
# source activate coom

# Install deps once (do NOT do this every run unless you want to):
# pip install -e ".[cl]"

# ------------------------------------

# Helpful runtime defaults
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1

# Run from repo root (assumes you submit from repo root)
REPO_ROOT="$(pwd)"

# ---- Experiment config (EDIT ME) ----
SEQUENCE="CO8"
SEED=0

# HACE params
HACE_SETPOINT=100
HACE_SCALE=0.01

# Core training params (tune as needed)
STEPS_PER_ENV=200000
UPDATE_AFTER=5000
UPDATE_EVERY=500
N_UPDATES=50
BATCH_SIZE=128
LR=1e-3

python3 -u CL/run_cl.py \
  --sequence "${SEQUENCE}" \
  --seed "${SEED}" \
  --cl_method hace \
  --hace_health_setpoint "${HACE_SETPOINT}" \
  --hace_internal_reward_scale "${HACE_SCALE}" \
  --steps_per_env "${STEPS_PER_ENV}" \
  --update_after "${UPDATE_AFTER}" \
  --update_every "${UPDATE_EVERY}" \
  --n_updates "${N_UPDATES}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --no_test

echo "Job finished: $(date)"

