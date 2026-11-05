#!/bin/bash
#SBATCH --job-name=role-swe-repairs-regrade
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
scorer_scratch=$(mktemp -d /tmp/swe-scorer-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$scorer_scratch"' EXIT
export TMPDIR="$scorer_scratch" APPTAINER_TMPDIR="$scorer_scratch" APPTAINER_CACHEDIR="$scorer_scratch/cache"
export SWE_IMAGE_CACHE="$PWD/.cache/swe-images" HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/datasets"
export INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals"
export SWE_GRADING_ARTIFACTS="$PWD/experiment_reports/training_against_role_probes/swe_corrected/data/saved-repairs-regrade-$SLURM_JOB_ID/test_outputs"
export SWE_APPDIRS_FIXTURE="$PWD/experiment_reports/training_against_role_probes/swe_corrected/data/appdirs-fixture"
.tools/bin/uv run --project src/experiments/eval_role_probes/agentic_eval/swe_bench/env --frozen --no-sync python src/experiments/training_against_role_probes/swe_corrected/regrade_repairs.py --sample-id astropy__astropy-7606 --sample-id sphinx-doc__sphinx-9711 --sample-id pylint-dev__pylint-4661 --sample-id psf__requests-1142 --output "experiment_reports/training_against_role_probes/swe_corrected/data/saved-repairs-regrade-$SLURM_JOB_ID"
