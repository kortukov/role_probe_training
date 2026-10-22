#!/bin/bash
#SBATCH --job-name=agentic-appworld
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=experiment_reports/agentic/appworld/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export HF_HOME="$PWD/.cache/huggingface" VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p "$TMPDIR"
project=src/experiments/agentic_eval
run_dir="$PWD/experiment_reports/agentic/appworld/runs/$SLURM_JOB_ID"
mkdir -p "$run_dir"
port=$((18000+SLURM_JOB_ID%10000))
uv run --project "$project/server_current" --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
 --tool-call-parser openai --enable-auto-tool-choice --enforce-eager --disable-log-requests \
 > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<180;i++)); do
 if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$run_dir/server.log"; exit 1; fi
 sleep 5
done
[[ "$ready" == 1 ]]
if [[ "${1:-}" == "--pilot" ]]; then
 for concurrency in 1 8; do
  uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_run.py" \
   --base-url "http://127.0.0.1:$port/v1" --split train --samples 4 --concurrency "$concurrency" \
   --output "$run_dir/train-c$concurrency"
 done
else
 if [[ "${SWE_DIAGNOSTIC:-0}" == 1 ]]; then
  uv run --project "$project/swe" --frozen --no-sync python "$project/swe_replay.py" \
   --base-url "http://127.0.0.1:$port/v1" \
   --output "$run_dir/swe-request-diagnostic.json"
 fi
 for split in train test_normal test_challenge; do
  uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_run.py" \
   --base-url "http://127.0.0.1:$port/v1" --split "$split" --concurrency 8 --output "$run_dir/$split"
 done
fi
