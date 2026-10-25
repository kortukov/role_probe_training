#!/bin/bash
#SBATCH --job-name=role-swe-parity
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/swe_gap/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
gap_scratch=$(mktemp -d /tmp/swe-regrade-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$gap_scratch"' EXIT
export TMPDIR="$gap_scratch" APPTAINER_TMPDIR="$gap_scratch" APPTAINER_CACHEDIR="$gap_scratch/cache"
export SWE_IMAGE_CACHE="$PWD/.cache/swe-images" HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/datasets"
export INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals"
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/training_against_role_probes/swe_gap/regrade.py --sample-id psf__requests-1142 --inspect-script --output "experiment_reports/training_against_role_probes/swe_gap/data/inspect-regrade-$SLURM_JOB_ID"
