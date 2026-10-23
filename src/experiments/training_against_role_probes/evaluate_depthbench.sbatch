#!/bin/bash
#SBATCH --job-name=depthbench-probes
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
collection=${1:?Pass the completed collection directory}
shift
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/assemble_depthbench.py \
 --collection "$collection" "$@"
bash src/experiments/training_against_role_probes/evaluate_corpus.sbatch \
 --input experiment_reports/training_against_role_probes/depthbench/data/train_1_2_3/trajectories.jsonl \
 --output experiment_reports/training_against_role_probes/depthbench/data/baseline
