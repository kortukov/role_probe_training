#!/bin/bash
#SBATCH --job-name=role-hf-fixed-bench
#SBATCH --account=gpu
#SBATCH --partition=ber-testing
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy/appworld2700/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache" APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
unset CUDA_LAUNCH_BLOCKING
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
root=experiment_reports/training_against_role_probes/on_policy/appworld2700
run="$root/runs/hf_benchmark/$SLURM_JOB_ID"
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync \
 python src/experiments/training_against_role_probes/hf_generation/benchmark.py \
 --output "$run" --diagnostic "$root/runs/hf_cuda/$HF_DIAGNOSTIC_JOB/regular-trained"
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync \
 python src/experiments/training_against_role_probes/hf_generation/metrics.py \
 --hf "$run" --vllm "$root/runs/serving_benchmark/1691825/1" --output "$run/comparison.json"
