#!/bin/bash
#SBATCH --job-name=role-data-27000
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
#SBATCH --array=30-299%8
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export HF_HOME="$PWD/.cache/huggingface" VLLM_CACHE_ROOT="$LOCAL_JOB_DIR/vllm"
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p "$TMPDIR"
export VLLM_USE_FLASHINFER_SAMPLER=0
project=src/experiments/agentic_eval
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=8
run_dir="$PWD/experiment_reports/training_against_role_probes/data_iteration/same_batch_27000/runs/collection/$SLURM_ARRAY_JOB_ID/shard$SLURM_ARRAY_TASK_ID"
mkdir -p "$run_dir"
port=$((18000+SLURM_JOB_ID%10000))
server_started=$SECONDS
run="$run_dir"
source src/experiments/training_against_role_probes/kl_iteration/serve_retry.sh
trap 'kill "${server_pid:-}" 2>/dev/null || true' EXIT
start_server uv run --project "$project/server_current" --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
 --attention-backend TRITON_ATTN --moe-backend triton --tool-call-parser openai --enable-auto-tool-choice --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16
printf '{"server_startup_seconds": %s}\n' "$((SECONDS-server_started))" > "$run_dir/startup.json"
samples=90
seed=$((8301 + SLURM_ARRAY_TASK_ID))
uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_run.py" \
 --base-url "http://127.0.0.1:$port/v1" --split train --samples "$samples" \
 --seed "$seed" --concurrency 8 --output "$run_dir/train"
