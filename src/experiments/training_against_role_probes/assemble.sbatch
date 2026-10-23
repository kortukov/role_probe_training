#!/bin/bash
#SBATCH --job-name=role-data-merge
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=experiment_reports/agentic/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root=experiment_reports/agentic/training_against_role_probes
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/assemble.py \
 --runs "$root/runs/$1" --output "$root/data"
cp "$root/data/manifest.json" "$root/corpus.json"
