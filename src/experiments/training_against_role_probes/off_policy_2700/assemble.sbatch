#!/bin/bash
#SBATCH --job-name=offpolicy2700-assemble
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/off_policy_2700/logs/%x-%j.out
# sbatch assemble.sbatch COLLECTION_ARRAY_JOB_ID
# Validates the 30 training shards, writes data/train2700 and data/dev, then the KL masks.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
root=experiment_reports/training_against_role_probes/off_policy_2700
runs="$root/runs/collection/${1:?collection array job id}"
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/assemble_train.py \
 --runs "$runs" --output "$root/data/train2700"
mkdir -p "$root/data/dev"
cp "$runs/shard30/dev/trajectories.jsonl" "$runs/shard30/dev/manifest.json" "$runs/shard30/dev/summary.json" "$root/data/dev/"
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/off_policy_2700/prepare.py \
 --trajectories "$root/data/train2700/trajectories.jsonl" --validation "$root/data/dev/trajectories.jsonl" --output "$root/data"
