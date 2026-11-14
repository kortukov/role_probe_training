#!/bin/bash
#SBATCH --job-name=ladder-concurrency
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy_progress/logs/%x-%j.out
# How many AppWorld rollouts can one GPU generate at once? Collects 64 training tasks from the
# base model through the Harmony proxy at each concurrency given (default 8 16 32 64) and
# leaves one timing.json and summary.json per level under concurrency_test/<job>/c<level>/.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export HF_HOME="$PWD/.cache/huggingface" VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p "$TMPDIR"
export VLLM_USE_FLASHINFER_SAMPLER=0
project=src/experiments/eval_role_probes/agentic_eval
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=8
run_dir="$PWD/experiment_reports/training_against_role_probes/on_policy_progress/concurrency_test/$SLURM_JOB_ID"
mkdir -p "$run_dir"
port=$((18000+SLURM_JOB_ID%10000)); proxy_port=$((port+10000))
uv run --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 32768 --max-num-seqs 64 --gpu-memory-utilization 0.80 \
 --attention-backend TRITON_ATTN --moe-backend triton \
 --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 32 64 --max-cudagraph-capture-size 64 \
 > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" ${proxy_pid:-} 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<240;i++)); do
 if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$run_dir/server.log"; exit 1; fi
 sleep 5
done
[[ "$ready" == 1 ]] || { echo 'Server startup timeout'; exit 1; }
for level in "${@:-8 16 32 64}"; do for c in $level; do
 out="$run_dir/c$c"
 uv run --frozen --no-sync python "$project/shared/harmony_proxy.py" --vllm-url "http://127.0.0.1:$port" \
  --model openai/gpt-oss-20b --output "$out/proxy" --port "$proxy_port" --reasoning-effort high \
  --max-tokens 8192 --context-limit 32768 --temperature 1 > "$out.proxy.log" 2>&1 &
 proxy_pid=$!
 for ((i=0;i<60;i++)); do
  curl --silent --fail "http://127.0.0.1:$proxy_port/v1/models" >/dev/null && break
  kill -0 "$proxy_pid" 2>/dev/null || { cat "$out.proxy.log"; exit 1; }
  sleep 1
 done
 # A failing level must not hide the levels already measured.
 uv run --frozen --no-sync python "$project/appworld/appworld_run.py" \
  --base-url "http://127.0.0.1:$proxy_port/v1" --requests "$out/proxy/requests.jsonl" \
  --split train --samples 64 --seed $((7000+c)) --concurrency "$c" --temperature 1 --output "$out" || echo "concurrency $c failed"
 kill "$proxy_pid" 2>/dev/null; wait "$proxy_pid" 2>/dev/null || true
 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$out.gpu.txt" || true
done; done
