#!/bin/bash
#SBATCH --job-name=role-probe-smoke
#SBATCH --account=hx2-users
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:20G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=experiment_reports/gptoss_role_probes/training_data/logs/%x-%j.out
source src/experiments/gptoss_role_probes/hx2/common.sh
srun --cpu-bind=none uv run --frozen --no-sync python -u src/experiments/gptoss_role_probes/smoke.py
