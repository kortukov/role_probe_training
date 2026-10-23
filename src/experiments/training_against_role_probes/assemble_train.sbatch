#!/bin/bash
#SBATCH --job-name=role-train-merge
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root=experiment_reports/training_against_role_probes
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/assemble_train.py \
 --runs "$root/runs/train2700/$1" --output "$root/data/train2700"
cp "$root/data/train2700/manifest.json" "$root/train_corpus.json"
