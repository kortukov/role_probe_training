#!/bin/bash
#SBATCH --job-name=norobots-score
#SBATCH --account=hx2-users
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:20G
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=experiment_reports/eval_role_probes/chat_eval/norobots/logs/%x-%j.out
source src/experiments/gptoss_role_probes/hx2/common.sh
root=experiment_reports/eval_role_probes/chat_eval/norobots
arms=(experiment_reports/gptoss_role_probes/training_data/c4_dolma experiment_reports/gptoss_role_probes/training_data/toucan experiment_reports/gptoss_role_probes/training_data/combined)
uv run --frozen --no-sync python -u src/experiments/eval_role_probes/chat_eval/norobots/score.py \
 --input "$root/inputs/samples.jsonl" --messages-key replay_messages --output "$root/metrics/version1_replay" --arms "${arms[@]}"
uv run --frozen --no-sync python -u src/experiments/eval_role_probes/chat_eval/norobots/score.py \
 --input "$root/artifacts/generated.jsonl" --messages-key messages --output "$root/metrics/version2_generated" --arms "${arms[@]}"
uv run --frozen --no-sync python src/experiments/eval_role_probes/chat_eval/norobots/report.py --root "$root"
