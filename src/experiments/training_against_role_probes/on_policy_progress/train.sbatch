#!/bin/bash
#SBATCH --job-name=ladder-train
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy_progress/logs/%x-%j.out
# sbatch --job-name=ladder-NAME train.sbatch NAME [--stop-after N] [--resume CHECKPOINT]
# NAME is a configuration in this folder (small_k1, small_k2, small_k6). The trainer generates
# its rollouts itself, from the current model, at the start of every epoch.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
name=${1:?configuration name}; shift
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=4
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache" APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_ENTITY=kortukov WANDB_PROJECT=training_against_role_probes
export WANDB_DIR="$PWD/experiment_reports/training_against_role_probes/on_policy_progress/$name/wandb"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$WANDB_DIR"
# The validation rollouts (8 dev tasks, base model) are shared by all runs; the first job copies
# them from the collection named in DEV_SOURCE.
dev=experiment_reports/training_against_role_probes/on_policy_progress/data/dev/trajectories.jsonl
if [[ ! -f "$dev" ]]; then
 mkdir -p "$(dirname "$dev")"; cp "${DEV_SOURCE:?validation rollouts are missing; set DEV_SOURCE}" "$dev.$SLURM_JOB_ID"; mv -n "$dev.$SLURM_JOB_ID" "$dev"; rm -f "$dev.$SLURM_JOB_ID"
fi
.tools/bin/uv run --frozen --no-sync \
 torchrun --standalone --nproc-per-node=1 --module role_probe_training.epoch_training.train \
 --config "src/experiments/training_against_role_probes/on_policy_progress/$name.json" "$@"
