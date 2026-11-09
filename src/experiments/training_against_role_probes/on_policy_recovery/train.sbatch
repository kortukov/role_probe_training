#!/bin/bash
#SBATCH --job-name=role-onpolicy-270
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=4
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache" APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_ENTITY=kortukov WANDB_PROJECT=training_against_role_probes
export WANDB_DIR="$PWD/$VARIANT_REPORT/wandb"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$WANDB_DIR"
.tools/bin/uv run --frozen --no-sync \
 torchrun --standalone --nproc-per-node=2 --module role_probe_training.on_policy_recovery.train \
 --config "$VARIANT_CONFIG" "$@"
