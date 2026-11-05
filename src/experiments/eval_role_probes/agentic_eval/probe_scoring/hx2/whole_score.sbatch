#!/bin/bash
#SBATCH --job-name=role-whole-score
#SBATCH --account=hx2-users
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:60G
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=experiment_reports/gptoss_role_probes/training_data/logs/%x-%j.out
# Score whole saved base-model histories with one or more probe sets on the unchanged model.
# Usage: whole_score.sbatch INPUT.jsonl OUTPUT_ROOT ARM_DIR [ARM_DIR ...]
#   Each ARM_DIR holds probes/layer_{4,8,12,16,20}.npz; predictions go to OUTPUT_ROOT/<arm name>.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
scratch="${TMPDIR:-/tmp}/whole-score-$SLURM_JOB_ID"
mkdir -p "$scratch/tmp" "$scratch/inductor" "$scratch/triton"
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch/inductor" TRITON_CACHE_DIR="$scratch/triton"
export HF_HOME=/hx2-weka/home/xy2222/.cache/huggingface HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
input=${1:?trajectories.jsonl}; output_root=${2:?output root}; shift 2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for arm_dir in "$@"; do
  arm=$(basename "$arm_dir"); probes=()
  for layer in 4 8 12 16 20; do probes+=(--probe "$layer=$arm_dir/probes/layer_$layer.npz"); done
  echo "=== scoring arm $arm ==="
  uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync \
    python -m role_probe_training.post_training.full_history --root "$arm_dir" --input "$input" \
    --output "$output_root/$arm" --base --overlength exclude "${probes[@]}"
done
