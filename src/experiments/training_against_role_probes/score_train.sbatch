#!/bin/bash
#SBATCH --job-name=role-corpus-probes
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --exclude=vca-gpu-0701-03
#SBATCH --output=experiment_reports/training_against_role_probes/predictions/train2700/shard$SLURM_ARRAY_TASK_ID
.tools/bin/uv run --project "$runtime" --frozen --no-sync python -m role_probe_training.training.score \
 --input experiment_reports/training_against_role_probes/data/train2700/trajectories.jsonl \
 --output "$output" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 2
