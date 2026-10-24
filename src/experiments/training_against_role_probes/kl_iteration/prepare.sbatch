#!/bin/bash
#SBATCH --job-name=role-kl-prepare
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
runtime=src/experiments/training_against_role_probes/runtime
.tools/bin/uv run --project "$runtime" --frozen --no-sync python src/experiments/training_against_role_probes/checks.py
.tools/bin/uv run --project "$runtime" --frozen --no-sync python src/experiments/training_against_role_probes/kl_iteration/checks.py
.tools/bin/uv run --project "$runtime" --frozen --no-sync python src/experiments/training_against_role_probes/kl_iteration/prepare.py "${1:?variant}"
