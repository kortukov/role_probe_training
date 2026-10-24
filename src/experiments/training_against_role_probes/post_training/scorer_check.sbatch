#!/bin/bash
#SBATCH --job-name=role-swe-scorer
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
scorer_scratch=$(mktemp -d /tmp/swe-scorer-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$scorer_scratch"' EXIT
export TMPDIR="$scorer_scratch" APPTAINER_TMPDIR="$scorer_scratch" APPTAINER_CACHEDIR="$scorer_scratch/cache"
export SWE_IMAGE_CACHE="$PWD/.cache/swe-images" HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/datasets"
export INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals"
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/training_against_role_probes/post_training/validate_swe_scorer.py --output "experiment_reports/training_against_role_probes/runs/swe_native/scorer-$SLURM_JOB_ID"
