#!/bin/bash
#SBATCH --job-name=role-base-export
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --exclude=vca-gpu-0701-03
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python src/experiments/training_against_role_probes/post_training/export_base.py --output experiment_reports/training_against_role_probes/base_model/data/swe_native/model
