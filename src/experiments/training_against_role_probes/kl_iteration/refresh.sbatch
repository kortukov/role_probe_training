#!/bin/bash
#SBATCH --job-name=role-kl-refresh
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
flock experiment_reports/training_against_role_probes/kl_iteration/data-report.lock \
 .tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/kl_iteration/report.py
