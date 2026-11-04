#!/bin/bash
#SBATCH --job-name=agentdojo-roles
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=experiment_reports/prompt_injection/agentdojo_same_batch_27000/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
kind=${1:?base or trained}
root=experiment_reports/prompt_injection/agentdojo_same_batch_27000
scratch=$(mktemp -d /tmp/agentdojo-roles-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch" TRITON_CACHE_DIR="$scratch/triton" TORCHINDUCTOR_CACHE_DIR="$scratch/inductor"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
uv="$PWD/.tools/bin/uv"
runtime=src/experiments/training_against_role_probes/runtime
"$uv" run --project "$runtime" --frozen --no-sync python -m role_probe_training.post_training.agentdojo extract --root "$root" --model "$kind"
# CUDA-13 cuML runs in a separate interpreter from the CUDA-12 PEFT runtime.
"$uv" run --frozen --no-sync python -m role_probe_training.post_training.fit_saved --root "$root/role_perception/$kind"
# Verify the scoring path on full clean/attacked histories before the complete pass.
"$uv" run --project "$runtime" --frozen --no-sync python -m role_probe_training.post_training.agentdojo score --root "$root" --model "$kind" --limit-count 1
"$uv" run --project "$runtime" --frozen --no-sync python -m role_probe_training.post_training.agentdojo score --root "$root" --model "$kind"
