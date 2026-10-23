#!/bin/bash
#SBATCH --job-name=depthbench-data
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export HF_HOME="$PWD/.cache/huggingface" VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
project=src/experiments/agentic_eval
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH" FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=8
run_dir="$PWD/experiment_reports/training_against_role_probes/depthbench/runs/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$run_dir"
port=$((18000+SLURM_JOB_ID%10000))
uv run --project "$project/server_current" --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
 --attention-backend TRITON_ATTN --moe-backend triton --tool-call-parser openai --enable-auto-tool-choice \
 --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 \
 --max-cudagraph-capture-size 16 > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<180;i++)); do
 if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$run_dir/server.log"; exit 1; fi
 sleep 5
done
[[ "$ready" == 1 ]]
client=src/experiments/training_against_role_probes/collection_runtime
uv run --project "$client" --frozen --no-sync python -u src/experiments/training_against_role_probes/collect_depthbench.py \
 --base-url "http://127.0.0.1:$port/v1" --output "$run_dir/smoke" --smoke --attempts 1 --concurrency 7
uv run --project "$client" --frozen --no-sync python -c \
 'import json,sys; s=json.load(open(sys.argv[1])); assert s["trajectories"]==7 and s["errors"]==0 and s["reasoning_bearing"]==7, s' "$run_dir/smoke/summary.json"
uv run --project "$client" --frozen --no-sync python -u src/experiments/training_against_role_probes/collect_depthbench.py \
 --base-url "http://127.0.0.1:$port/v1" --output "$run_dir/collection" --attempts 20 --concurrency 16
