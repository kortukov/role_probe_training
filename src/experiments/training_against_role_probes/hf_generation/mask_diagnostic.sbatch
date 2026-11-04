#!/bin/bash
#SBATCH --job-name=role-hf-mask-fix
#SBATCH --account=gpu
#SBATCH --partition=ber-testing
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%1
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy/appworld2700/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
modes=(mask-trained mask-base)
unset CUDA_LAUNCH_BLOCKING
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync \
 python src/experiments/training_against_role_probes/hf_generation/mask_diagnostic.py \
 --mode "${modes[$SLURM_ARRAY_TASK_ID]}" \
 --output "experiment_reports/training_against_role_probes/on_policy/appworld2700/runs/hf_cuda/$SLURM_ARRAY_JOB_ID/${modes[$SLURM_ARRAY_TASK_ID]}"
