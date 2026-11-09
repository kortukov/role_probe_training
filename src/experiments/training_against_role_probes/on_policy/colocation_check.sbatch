#!/bin/bash
#SBATCH --job-name=role-colocation-check
#SBATCH --account=gpu
#SBATCH --partition=ber-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:55:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy/appworld2700/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/trainer-inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/trainer-triton" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache" APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
export FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=4
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
.tools/bin/uv run --frozen --no-sync \
 torchrun --standalone --nproc-per-node=2 src/experiments/training_against_role_probes/on_policy/colocation_check.py \
 --output "experiment_reports/training_against_role_probes/on_policy/appworld2700/runs/colocation/$SLURM_JOB_ID"
