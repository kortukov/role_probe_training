#!/bin/bash
#SBATCH --job-name=agentic-swe
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=experiment_reports/agentic/swe_bench_full/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export HF_HOME="$PWD/.cache/huggingface"
export INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals"
export HF_DATASETS_CACHE="$PWD/.cache/datasets"
export VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$TIKTOKEN_CACHE_DIR"
project=src/experiments/agentic_eval
client_project="$project/swe"
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache"
export APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
report_root="${AGENTIC_REPORT_ROOT:-experiment_reports/agentic/swe_bench_full}"
run_dir="$PWD/$report_root/runs/$SLURM_JOB_ID"
mkdir -p "$run_dir"
port=$((18000 + SLURM_JOB_ID % 10000))
cleanup() {
    if [[ -f "$run_dir/manifest.json" ]]; then
        uv run --project "$client_project" --frozen --no-sync python "$project/export.py" "$run_dir" || true
    fi
    kill "$server_pid" 2>/dev/null || true
}
uv run --project "$project/server" --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
    --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
    --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
    --tool-call-parser openai --enable-auto-tool-choice \
    --enforce-eager --disable-log-requests > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap cleanup EXIT
ready=0
for ((i=0;i<180;i++)); do
    if curl --silent --fail "http://127.0.0.1:$port/health" > /dev/null; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$run_dir/server.log"; exit 1; fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then echo 'Server startup timeout'; exit 1; fi
uv run --project "$client_project" --frozen --no-sync python "$project/smoke.py" "http://127.0.0.1:$port/v1" > "$run_dir/smoke.json"
if [[ "${1:-}" == "--smoke-only" ]]; then exit 0; fi
uv run --project "$client_project" --frozen --no-sync python "$project/swe_run.py" \
    --output "$run_dir" --base-url "http://127.0.0.1:$port/v1" --benchmark swe_bench --concurrency 8 --time-limit 1800 --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --shard-count "${SWE_SHARD_COUNT:-1}" "$@"
