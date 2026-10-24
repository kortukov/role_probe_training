#!/bin/bash
#SBATCH --job-name=role-kl-report
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Serial report writers avoid overwriting a simultaneous variant's table refresh.
flock experiment_reports/training_against_role_probes/kl_iteration/data-report.lock \
 .tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/kl_iteration/finalize.py --root "${1:?root}"
