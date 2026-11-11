#!/bin/bash
#SBATCH --job-name=header-role-probes
#SBATCH --account=hx2-users
#SBATCH --partition=interactive-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:20G
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=experiment_reports/gptoss_role_probes/header_roles/logs/%x-%j.out
# Usage: sbatch header_roles.sbatch [arguments of the fit stage, e.g. --layers 12 --backend cuml]
# Extracts and caches the activations if they are missing, then fits. It fits the interactive
# partition (64 GB host memory): activations are memory-mapped and one layer is loaded at a time.
source src/experiments/gptoss_role_probes/hx2/common.sh
run="uv run --project src/experiments/gptoss_role_probes/fit_env --frozen --no-sync python -u src/experiments/gptoss_role_probes/header_roles/fit.py"
srun --cpu-bind=none $run extract
srun --cpu-bind=none $run fit "$@"
