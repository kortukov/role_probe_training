#!/bin/bash
#SBATCH --job-name=role-probe-smoke
#SBATCH --account=gpu
#SBATCH --partition=ber-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=experiment_reports/role_probes/gptoss20b_baseline/logs/%x-%j.out
set -euo pipefail
source /etc/slurm/local_job_dir.sh
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface"
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
srun --cpu-bind=none uv run --frozen --no-sync python -u src/experiments/gptoss_role_probes/smoke.py
