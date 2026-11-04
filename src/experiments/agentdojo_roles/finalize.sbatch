#!/bin/bash
#SBATCH --job-name=agentdojo-role-report
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=experiment_reports/prompt_injection/agentdojo_same_batch_27000/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
.tools/bin/uv run --frozen --no-sync python src/experiments/agentdojo_roles/report.py
.tools/bin/uv run --frozen --no-sync python src/experiments/agentdojo_roles/publish.py
