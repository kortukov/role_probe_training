#!/bin/bash
#SBATCH --job-name=role-onpolicy-debug
#SBATCH --account=gpu
#SBATCH --partition=ber-testing
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --array=0-2%1
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy/appworld2700/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_LAUNCH_BLOCKING=1
mkdir -p "$TMPDIR"
modes=(flex-thread flex-main eager-thread)
mode=${modes[$SLURM_ARRAY_TASK_ID]}
root=experiment_reports/training_against_role_probes/on_policy/appworld2700
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync \
    python src/experiments/training_against_role_probes/on_policy/generation_diagnostic.py \
    --mode "$mode" --output "$root/runs/diagnostics/${SLURM_ARRAY_JOB_ID}/$mode"
