#!/bin/bash
#SBATCH --job-name=agentic-retail
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=experiment_reports/eval_role_probes/agentic_eval/tau2_retail/base_trajectories/logs/%x-%j.out
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
project=src/experiments/eval_role_probes/agentic_eval
# vLLM comes from the root environment's vllm extra (0.27.1, compiled Triton kernels).
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=8 VLLM_USE_FLASHINFER_SAMPLER=0
report_root="${AGENTIC_REPORT_ROOT:-experiment_reports/eval_role_probes/agentic_eval/tau2_retail/base_trajectories}"
run_dir="$PWD/$report_root/runs/$SLURM_JOB_ID"
mkdir -p "$run_dir"
port=$((18000 + SLURM_JOB_ID % 10000))
proxy_port=$((port + 10000))  # never another job's vLLM port (18000-27999)
cleanup() {
    if [[ -f "$run_dir/manifest.json" ]]; then
        requests=(); [[ -f "$run_dir/proxy/requests.jsonl" ]] && requests=(--requests "$run_dir/proxy/requests.jsonl")
        uv run --frozen --no-sync python "$project/shared/export.py" "$run_dir" "${requests[@]}" || true
    fi
    for pid in "${proxy_pid:-}" "$server_pid"; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done
}
uv run --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
    --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
    --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
    --attention-backend TRITON_ATTN --moe-backend triton --kernel-config '{"enable_jit_warmup":false}' \
    --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16 > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap cleanup EXIT
ready=0
for ((i=0;i<180;i++)); do
    if curl --silent --fail "http://127.0.0.1:$port/health" > /dev/null; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$run_dir/server.log"; exit 1; fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then echo 'Server startup timeout'; exit 1; fi
# Harmony proxy (root environment) between Inspect and vLLM: renders prompts with oss-harmony,
# samples token IDs from vLLM and records them in $run_dir/proxy/requests.jsonl. Its reasoning
# effort, max_tokens and temperature must equal what the run sends (the tau2 agent is generated at temperature 0).
uv run --frozen --no-sync python "$project/shared/harmony_proxy.py" --vllm-url "http://127.0.0.1:$port" \
    --model openai/gpt-oss-20b --output "$run_dir/proxy" --port "$proxy_port" --reasoning-effort high \
    --max-tokens 16384 --context-limit 32768 --temperature 0 > "$run_dir/proxy.log" 2>&1 &
proxy_pid=$!
for ((i=0;i<60;i++)); do
    curl --silent --fail "http://127.0.0.1:$proxy_port/v1/models" > /dev/null && break
    kill -0 "$proxy_pid" 2>/dev/null || { cat "$run_dir/proxy.log"; exit 1; }
    sleep 1
done
uv run --frozen --no-sync python "$project/shared/smoke.py" "http://127.0.0.1:$proxy_port/v1" \
    --max-tokens 16384 --temperature 0 > "$run_dir/smoke.json"
if [[ "${1:-}" == "--smoke-only" ]]; then exit 0; fi
uv run --frozen --no-sync python "$project/tau2_retail/run.py" \
    --output "$run_dir" --base-url "http://127.0.0.1:$proxy_port/v1" --requests "$run_dir/proxy/requests.jsonl" "$@"
