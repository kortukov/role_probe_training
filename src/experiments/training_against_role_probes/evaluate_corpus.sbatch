#!/bin/bash
#SBATCH --job-name=corpus-probes
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export PATH="$PWD/.tools/bin:$PATH" HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
srun --cpu-bind=none uv run --frozen --no-sync python -u src/experiments/training_against_role_probes/evaluate_corpus.py "$@"
