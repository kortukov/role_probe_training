#!/bin/bash
#SBATCH --job-name=role-swe-standard-probes
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=vca-gpu-0701-03
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root=${1:?root}
score_scratch=$(mktemp -d /tmp/swe-probes-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$score_scratch"' EXIT
export TMPDIR="$score_scratch/tmp" TORCHINDUCTOR_CACHE_DIR="$score_scratch/inductor" TRITON_CACHE_DIR="$score_scratch/triton" OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
args=()
if [[ "$root" == */base_model ]]; then args=(--base); fi
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python -m role_probe_training.post_training.full_history \
 --root "$root" --input "$root/data/swe_inspect_standard/evaluation/trajectories.jsonl" \
 --output "$root/predictions/swe_inspect_standard" --overlength exclude "${args[@]}"
