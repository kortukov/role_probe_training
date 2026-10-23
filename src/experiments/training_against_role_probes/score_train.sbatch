#!/bin/bash
#SBATCH --job-name=role-corpus-probes
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=vca-gpu-0701-03
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_ENTITY=kortukov
export WANDB_PROJECT=training_against_role_probes
export WANDB_DIR="$PWD/experiment_reports/training_against_role_probes/wandb"
mkdir -p "$TMPDIR" "$WANDB_DIR"
runtime=src/experiments/training_against_role_probes/runtime
output=experiment_reports/training_against_role_probes/predictions/train2700/shard$SLURM_ARRAY_TASK_ID
.tools/bin/uv run --project "$runtime" --frozen --no-sync python -m role_probe_training.training.score \
 --input experiment_reports/training_against_role_probes/data/train2700/trajectories.jsonl \
 --output "$output" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 2
