#!/bin/bash
#SBATCH --job-name=role-probe-reproduce
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=experiment_reports/role_probes/gptoss20b_baseline/logs/%x-%j.out
set -euo pipefail
source /etc/slurm/local_job_dir.sh
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export MPLCONFIGDIR="$LOCAL_JOB_DIR/matplotlib"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
# Only small probes, summaries, plots and trajectories are written to shared storage.
# Activations stay in RAM and are released after each fit.
srun --cpu-bind=none uv run --frozen --no-sync python -u src/experiments/gptoss_role_probes/run.py "$@"
