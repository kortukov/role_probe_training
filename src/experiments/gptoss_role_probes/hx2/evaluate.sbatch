#!/bin/bash
#SBATCH --job-name=role-probe-evaluate
#SBATCH --account=hx2-users
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:20G
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=experiment_reports/gptoss_role_probes/training_data/logs/%x-%j.out
# Usage: sbatch evaluate.sbatch --config CONFIG --output OUTPUT [--from OTHER_OUTPUT]
# With --from, the shared generated conversations of another arm are copied first so
# every probe is scored on identical clean-chat conversations.
source src/experiments/gptoss_role_probes/hx2/common.sh
args=(); from=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) from="$2"; shift 2;;
    --output) output="$2"; args+=("$1" "$2"); shift 2;;
    *) args+=("$1"); shift;;
  esac
done
if [[ -n "$from" && ! -f "$output/artifacts/conversations.jsonl" ]]; then
  cp "$from/artifacts/conversations.jsonl" "$from/artifacts/generation_failures.json" "$output/artifacts/"
  echo "Copied shared conversations from $from"
fi
srun --cpu-bind=none uv run --frozen --no-sync python -u src/experiments/gptoss_role_probes/run.py "${args[@]}" --stage evaluate
