#!/bin/bash
#SBATCH --job-name=role-throughput
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --exclude=vca-gpu-0701-03
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
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
output=experiment_reports/training_against_role_probes/data/throughput/$SLURM_JOB_ID
mkdir -p "$output"
# Validate loading, checkpointing, both replica gradients and optimizer first.
.tools/bin/uv run --project "$runtime" --frozen --no-sync torchrun --standalone --nproc-per-node=2 \
  --module role_probe_training.training.throughput --context 4096 --smoke \
  --data experiment_reports/training_against_role_probes/data/debug/trajectories.jsonl \
  --output "$output/smoke"
# Separate processes release all allocations and reset adapters across conditions.
for context in 4096 8192 16384 0; do
  if .tools/bin/uv run --project "$runtime" --frozen --no-sync torchrun --standalone --nproc-per-node=2 \
      --module role_probe_training.training.throughput --context "$context" \
      --data experiment_reports/training_against_role_probes/data/debug/trajectories.jsonl \
      --output "$output/$context"; then
    echo "Context $context complete"
  else
    echo "Context $context failed; retained logs; continuing remaining conditions"
  fi
done
