#!/bin/bash
#SBATCH --job-name=role-adapter-check
#SBATCH --account=gpu
#SBATCH --partition=ber-testing
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=experiment_reports/prompt_injection/agentdojo_same_batch_27000/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root=experiment_reports/prompt_injection/agentdojo_same_batch_27000
scratch=$(mktemp -d /tmp/adapter-check-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch" TORCHINDUCTOR_CACHE_DIR="$scratch/inductor" TRITON_CACHE_DIR="$scratch/triton" OMP_NUM_THREADS=4
export HF_HOME="$PWD/.cache/huggingface"
uvbin="$PWD/.tools/bin/uv"
"$uvbin" run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python src/experiments/adapter_release/prepare_vllm.py \
 --adapter "$root/data/hf_adapter" --base .cache/models/gpt-oss-20b --output "$root/data/adapter_reconstructed"
"$uvbin" run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python src/experiments/adapter_release/verify_export.py
