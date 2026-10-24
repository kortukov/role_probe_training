#!/bin/bash
#SBATCH --job-name=role-export-model
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=vca-gpu-0701-03
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python -m role_probe_training.post_training.export_model \
 --checkpoint "${1:-experiment_reports/training_against_role_probes/runs/training/appworld2700/checkpoints/step00169}" \
 --output "${2:-experiment_reports/training_against_role_probes/data/post_training/model}" "${@:3}"
