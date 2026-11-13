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
#SBATCH --output=experiment_reports/eval_role_probes/agentic_eval/appworld/base_trajectories/logs/%x-%j.out
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
# DE.BER1: one vLLM replica, one GPU, Inspect behind the Harmony proxy (same flow as hx2/appworld.sbatch).
#   sbatch appworld_job.sbatch --pilot            # 4 seeded training tasks at concurrency 1 and 8
#   sbatch appworld_job.sbatch test_normal [...]  # one or more splits, concurrency 8
report_root="${AGENTIC_REPORT_ROOT:-experiment_reports/eval_role_probes/agentic_eval/appworld/base_trajectories}"
temperature="${AGENTIC_TEMPERATURE:-1}"  # 1 is the rule for every collection (the model default); 0 decodes greedily
run_dir="$PWD/$report_root/runs/$SLURM_JOB_ID"
mkdir -p "$run_dir"
port=$((18000+SLURM_JOB_ID%10000))
server_started=$SECONDS
uv run --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
 --attention-backend TRITON_ATTN --moe-backend triton \
 --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16 \
 > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
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
  --max-tokens 8192 --context-limit 32768 --temperature "$temperature" > "$output.proxy.log" 2>&1 &
 local proxy_pid=$!
 for ((i=0;i<60;i++)); do
  curl --silent --fail "http://127.0.0.1:$proxy_port/v1/models" >/dev/null && break
  kill -0 "$proxy_pid" 2>/dev/null || { cat "$output.proxy.log"; exit 1; }
  sleep 1
 done
 uv run --frozen --no-sync python "$project/appworld/appworld_run.py" \
  --base-url "http://127.0.0.1:$proxy_port/v1" --requests "$output/proxy/requests.jsonl" --output "$output" --temperature "$temperature" "$@"
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
