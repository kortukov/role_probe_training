#!/bin/bash
#SBATCH --job-name=role-samebatch-eval
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=60:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src"
root=${1:?checkpoint evaluation root}
checkpoint=${2:?retained training checkpoint}
code=src/experiments/training_against_role_probes
bash "$code/on_policy_compact/evaluate.sbatch" prepare "$root" "$checkpoint"
bash "$code/on_policy_compact/evaluate.sbatch" app-retail "$root" "samebatch-$(basename "$root")"
.tools/bin/uv run --frozen --no-sync python "$code/same_batch_epochs/report.py" --root "$(dirname "$(dirname "$root")")"
