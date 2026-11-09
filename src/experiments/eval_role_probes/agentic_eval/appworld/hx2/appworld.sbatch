#!/bin/bash
#SBATCH --job-name=appworld-hx2
#SBATCH --account=hx2-users
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:100G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=experiment_reports/eval_role_probes/agentic_eval/appworld/base_trajectories_hx2/logs/%x-%j.out
# hx2 port of appworld_job.sbatch: one vLLM replica, one GPU, Inspect behind the Harmony proxy.
#   sbatch appworld.sbatch --pilot            # 4 seeded training tasks at concurrency 1 and 8
#   sbatch appworld.sbatch test_normal [...]  # one or more splits, concurrency 8
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
scratch="${TMPDIR:-/tmp}/appworld-$SLURM_JOB_ID"
mkdir -p "$scratch/tmp" "$scratch/triton"
export TMPDIR="$scratch/tmp" TRITON_CACHE_DIR="$scratch/triton" FLASHINFER_WORKSPACE_BASE="$scratch"
export HF_HOME=/hx2-weka/home/xy2222/.cache/huggingface HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic" TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MAX_JOBS=8 VLLM_USE_FLASHINFER_SAMPLER=0
export APPTAINER_CACHEDIR=/hx2-weka/home/xy2222/.cache/apptainer
mkdir -p "$VLLM_CACHE_ROOT" "$TIKTOKEN_CACHE_DIR"
project=src/experiments/eval_role_probes/agentic_eval
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
run_dir="$PWD/experiment_reports/eval_role_probes/agentic_eval/appworld/base_trajectories_hx2/runs/$SLURM_JOB_ID"
mkdir -p "$run_dir"
port=$((18000+SLURM_JOB_ID%10000))
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
server_started=$SECONDS
uv run --project "$project/server_current" --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
 --attention-backend TRITON_ATTN --moe-backend triton \
 --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16 \
 > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; rm -rf "$scratch"' EXIT
ready=0
for ((i=0;i<240;i++)); do
 if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$run_dir/server.log"; exit 1; fi
 sleep 5
done
[[ "$ready" == 1 ]] || { echo 'Server startup timeout'; tail -60 "$run_dir/server.log"; exit 1; }
printf '{"server_startup_seconds": %s, "node": "%s"}\n' "$((SECONDS-server_started))" "$(hostname)" > "$run_dir/startup.json"
# One Harmony proxy per collection: renders prompts with oss-harmony, samples token IDs
# from vLLM and records them in <output>/proxy/requests.jsonl (root environment).
collect() {  # collect OUTPUT_DIR APPWORLD_RUN_ARGS...
 local output="$1"; shift
 local proxy_port=$((port+10000))  # never another job's vLLM port (18000-27999)
 uv run --frozen --no-sync python "$project/shared/harmony_proxy.py" --vllm-url "http://127.0.0.1:$port" \
  --model openai/gpt-oss-20b --output "$output/proxy" --port "$proxy_port" --reasoning-effort high \
  --max-tokens 8192 --context-limit 32768 --temperature 0 > "$output.proxy.log" 2>&1 &
 local proxy_pid=$!
 for ((i=0;i<60;i++)); do
  curl --silent --fail "http://127.0.0.1:$proxy_port/v1/models" >/dev/null && break
  kill -0 "$proxy_pid" 2>/dev/null || { cat "$output.proxy.log"; exit 1; }
  sleep 1
 done
 uv run --project "$project/swe_bench/env" --frozen --no-sync python "$project/appworld/appworld_run.py" \
  --base-url "http://127.0.0.1:$proxy_port/v1" --requests "$output/proxy/requests.jsonl" --output "$output" "$@"
 local status=$?
 kill "$proxy_pid" 2>/dev/null; wait "$proxy_pid" 2>/dev/null || true
 return $status
}
if [[ "${1:-}" == "--pilot" ]]; then
 for concurrency in 1 8; do
  collect "$run_dir/train-c$concurrency" --split train --samples 4 --concurrency "$concurrency"
 done
else
 for split in "$@"; do
  collect "$run_dir/$split" --split "$split" --concurrency 8
 done
fi
