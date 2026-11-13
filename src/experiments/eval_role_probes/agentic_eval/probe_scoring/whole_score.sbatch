#!/bin/bash
#SBATCH --job-name=role-whole-score
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05
#SBATCH --output=experiment_reports/eval_role_probes/agentic_eval/probe_scoring/logs/%x-%j.out
# DE.BER1: score whole saved base-model histories with five-role probes on the unchanged model.
# Usage: whole_score.sbatch INPUT.jsonl OUTPUT_DIR PROBE_DIR [LAYER ...]   (default layers 12 16)
#   PROBE_DIR holds layer_<LAYER>.npz. Histories above the native context are excluded and listed.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
scratch=$(mktemp -d /tmp/role-score-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch/inductor" TRITON_CACHE_DIR="$scratch/triton"
export HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
input=${1:?trajectories.jsonl}; output=${2:?output directory}; probe_dir=${3:?probe directory}; shift 3
probes=()
for layer in "${@:-12 16}"; do for l in $layer; do probes+=(--probe "$l=$probe_dir/layer_$l.npz"); done; done
.tools/bin/uv run --frozen --no-sync python -m role_probe_training.post_training.full_history \
  --root "$probe_dir" --input "$input" --output "$output" --base --overlength exclude "${probes[@]}"
