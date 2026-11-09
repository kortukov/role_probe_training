#!/bin/bash
#SBATCH --job-name=role-kl-batching-check
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
.tools/bin/uv run --frozen --no-sync \
 python src/experiments/training_against_role_probes/on_policy_concurrent/kl_batching_check.py "$@"
