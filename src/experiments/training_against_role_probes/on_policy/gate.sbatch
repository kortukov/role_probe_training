#!/bin/bash
#SBATCH --job-name=role-on-policy-gate
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy/appworld2700/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync \
    python src/experiments/training_against_role_probes/on_policy/gate.py
