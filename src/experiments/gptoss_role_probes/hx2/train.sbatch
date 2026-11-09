#!/bin/bash
#SBATCH --job-name=role-probe-train
#SBATCH --account=hx2-users
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:60G
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=experiment_reports/gptoss_role_probes/training_data/logs/%x-%j.out
# Usage: sbatch train.sbatch --config CONFIG --output OUTPUT [--stage train|evaluate|audit|all]
# Activations for all five layers stay in host RAM (float16) until each fit; hence --mem.
source src/experiments/gptoss_role_probes/hx2/common.sh
srun --cpu-bind=none uv run --project src/experiments/gptoss_role_probes/fit_env --frozen --no-sync python -u src/experiments/gptoss_role_probes/run.py "$@"
