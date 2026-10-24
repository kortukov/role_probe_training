#!/bin/bash
#SBATCH --job-name=role-fresh-score
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Array jobs may lack a correctly owned /data/local/jobs directory. Use a
# unique node-local temporary directory; all durable outputs go to the repo.
score_scratch=$(mktemp -d /tmp/role-score-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$score_scratch"' EXIT
export LOCAL_JOB_DIR="$score_scratch"
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
root=${1:?report root}
benchmarks=(appworld_test tau2_retail swe_bench)
benchmark=${benchmarks[$SLURM_ARRAY_TASK_ID]}
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python -m role_probe_training.post_training.probes score \
 --root "$root" --input "$root/data/post_training/$benchmark/trajectories.jsonl" --output "$root/predictions/post_training/$benchmark"
