#!/bin/bash
#SBATCH --job-name=role-swe-gold-repairs
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
scratch=$(mktemp -d /tmp/swe-gold-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch" APPTAINER_TMPDIR="$scratch" APPTAINER_CACHEDIR="$scratch/cache"
export SWE_IMAGE_CACHE="$PWD/.cache/swe-images" HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/datasets"
export INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals"
out="experiment_reports/training_against_role_probes/swe_corrected/data/gold-repairs-survey-${SLURM_ARRAY_JOB_ID}/shard${SLURM_ARRAY_TASK_ID}"
export SWE_GRADING_ARTIFACTS="$PWD/$out/test_outputs"
export SWE_APPDIRS_FIXTURE="$PWD/experiment_reports/training_against_role_probes/swe_corrected/data/appdirs-fixture"
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/swe_corrected/gold_repairs_survey.py --output "$out" --shard-index "$SLURM_ARRAY_TASK_ID"
