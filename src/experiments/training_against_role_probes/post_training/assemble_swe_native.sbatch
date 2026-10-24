#!/bin/bash
#SBATCH --job-name=role-swe-assemble
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/post_training/assemble_swe_native.py --root "${1:?root}" --collection-id "${2:?collection id}"
