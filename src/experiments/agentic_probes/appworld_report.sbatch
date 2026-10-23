#!/bin/bash
#SBATCH --job-name=appworld-probe-report
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=experiment_reports/agentic/appworld_probe_transfer/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.tools/bin/uv run --frozen --no-sync python src/experiments/agentic_probes/report.py \
 --dataset appworld_test_normal=experiment_reports/agentic/appworld_probe_transfer/predictions \
 --expected-count appworld_test_normal=168 \
 --layers 4 8 12 16 20 --output experiment_reports/agentic/appworld_probe_transfer
