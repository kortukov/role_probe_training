#!/bin/bash
#SBATCH --job-name=role-on-policy
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy/appworld2700/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache"
export APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_ENTITY=kortukov WANDB_PROJECT=training_against_role_probes
export WANDB_DIR="$PWD/experiment_reports/training_against_role_probes/on_policy/appworld2700/wandb"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$WANDB_DIR"
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync \
    torchrun --standalone --nproc-per-node=2 --module role_probe_training.on_policy.train \
    --config src/experiments/training_against_role_probes/on_policy/config.json "$@"
