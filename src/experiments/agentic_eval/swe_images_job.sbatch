#!/bin/bash
#SBATCH --job-name=swe-images
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-31%16
#SBATCH --output=experiment_reports/agentic/swe_bench_full/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache"
export APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
export TMPDIR="$LOCAL_JOB_DIR/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TMPDIR"
# Include the verified official blobs needed by two damaged registry layers.
cp -a "$PWD/.cache/apptainer-repair/cache" "$APPTAINER_CACHEDIR/"
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python \
  src/experiments/agentic_eval/cache_swe_image.py --all \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 32
