#!/bin/bash
#SBATCH --job-name=role-serving-summary
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy/appworld2700/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
.tools/bin/uv run --frozen --no-sync python \
 src/experiments/training_against_role_probes/on_policy/benchmark_summary.py "$1"
