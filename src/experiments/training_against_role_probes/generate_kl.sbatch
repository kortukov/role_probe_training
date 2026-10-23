#!/bin/bash
#SBATCH --job-name=role-kl
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export HF_HOME="$PWD/.cache/huggingface" VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p "$TMPDIR"
export VLLM_USE_FLASHINFER_SAMPLER=0
project=src/experiments/agentic_eval
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=8
uv run --project "$project/server_current" --frozen --no-sync python src/experiments/training_against_role_probes/generate_kl.py \
 --input experiment_reports/training_against_role_probes/data/kl/prompts.json \
 --output experiment_reports/training_against_role_probes/data/kl/anchors.jsonl
