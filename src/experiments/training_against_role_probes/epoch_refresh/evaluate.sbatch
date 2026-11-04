#!/bin/bash
#SBATCH --job-name=role-epoch-retail
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root=${1:?checkpoint evaluation root}
checkpoint=${2:?checkpoint path}
code=src/experiments/training_against_role_probes
# Each child releases its GPU server and scratch before the next starts.
bash "$code/post_training/export.sbatch" "$checkpoint" "$root/data/post_training/model"
bash "$code/post_training/probes.sbatch" fit --root "$root"
SLURM_ARRAY_TASK_ID=2 bash "$code/kl_iteration/collect.sbatch" "$root" "epoch-retail"
bash "$code/kl_iteration/assemble_benchmark.sbatch" "$root" "$SLURM_JOB_ID" --benchmarks tau2_retail
SLURM_ARRAY_TASK_ID=1 bash "$code/post_training/whole_score.sbatch" "$root" --overlength exclude
bash "$code/on_policy_evaluation/results.sbatch" "$root" tau2_retail
.tools/bin/uv run --frozen --no-sync python "$code/epoch_refresh/report.py" --root "$(dirname "$(dirname "$root")")"
