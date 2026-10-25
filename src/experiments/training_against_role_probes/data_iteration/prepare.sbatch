#!/bin/bash
#SBATCH --job-name=role-data-prepare
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
root=experiment_reports/training_against_role_probes/data_iteration/same_batch_27000
code=src/experiments/training_against_role_probes/data_iteration
runtime=src/experiments/training_against_role_probes/runtime
.tools/bin/uv run --project "$runtime" --frozen --no-sync python "$code/assemble.py" --runs "$root/runs/collection/${1:?collection job}" --output "$root/data/train27000"
.tools/bin/uv run --project "$runtime" --frozen --no-sync python "$code/prepare.py" --root "$root"
