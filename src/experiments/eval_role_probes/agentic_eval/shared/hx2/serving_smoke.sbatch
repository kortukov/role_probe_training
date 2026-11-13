#!/bin/bash
#SBATCH --job-name=serving-smoke
#SBATCH --account=hx2-users
#SBATCH --partition=interactive-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:40G
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=experiment_reports/eval_role_probes/agentic_eval/appworld/base_trajectories_hx2/logs/%x-%j.out
# hx2 smoke of the serving stack from the root environment (installed with the vllm extra): vLLM by
# token ID, one tool round trip through the Harmony proxy, then one Inspect request through the proxy.
# Submit from the repository root without --qos (the partition applies its own). Server and proxy
# logs stay on node-local scratch; their tails are printed if a step fails.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
scratch="${TMPDIR:-/tmp}/serving-smoke-$SLURM_JOB_ID"
mkdir -p "$scratch/tmp" "$scratch/triton" "$scratch/vllm"
export TMPDIR="$scratch/tmp" TRITON_CACHE_DIR="$scratch/triton" FLASHINFER_WORKSPACE_BASE="$scratch" VLLM_CACHE_ROOT="$scratch/vllm"
export HF_HOME=/hx2-weka/home/xy2222/.cache/huggingface HF_HUB_OFFLINE=1 TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MAX_JOBS=8 VLLM_USE_FLASHINFER_SAMPLER=0
project=src/experiments/eval_role_probes/agentic_eval
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
port=$((18000 + SLURM_JOB_ID % 10000))
proxy_port=$((port + 10000))
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run --frozen --no-sync python -c "import vllm, torch, transformers, inspect_ai, importlib.metadata as m; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'inspect-ai', m.version('inspect-ai'), 'oss-harmony', m.version('oss-harmony'))"
uv run --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
    --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
    --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
    --attention-backend TRITON_ATTN --moe-backend triton --kernel-config '{"enable_jit_warmup":false}' \
    --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16 > "$scratch/server.log" 2>&1 &
server_pid=$!
failed() { echo "--- server.log"; tail -40 "$scratch/server.log"; echo "--- proxy.log"; tail -40 "$scratch/proxy.log" 2>/dev/null || true; }
trap 'status=$?; [[ $status == 0 ]] || failed; kill "${proxy_pid:-}" "$server_pid" 2>/dev/null || true; rm -rf "$scratch"' EXIT
ready=0
for ((i=0;i<240;i++)); do
    if curl --silent --fail "http://127.0.0.1:$port/health" > /dev/null; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then echo 'vLLM exited during startup'; exit 1; fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then echo 'Server startup timeout'; exit 1; fi
echo "vLLM ready after ${SECONDS}s"
uv run --frozen --no-sync python "$project/shared/harmony_proxy.py" --vllm-url "http://127.0.0.1:$port" \
    --model openai/gpt-oss-20b --output "$scratch/proxy" --port "$proxy_port" --reasoning-effort high \
    --max-tokens 16384 --context-limit 32768 --temperature 1 > "$scratch/proxy.log" 2>&1 &
proxy_pid=$!
for ((i=0;i<60;i++)); do
    curl --silent --fail "http://127.0.0.1:$proxy_port/v1/models" > /dev/null && break
    kill -0 "$proxy_pid" 2>/dev/null || { echo 'proxy exited during startup'; exit 1; }
    sleep 1
done
uv run --frozen --no-sync python "$project/shared/serving_smoke.py" --vllm-url "http://127.0.0.1:$port" \
    --proxy-url "http://127.0.0.1:$proxy_port/v1" --max-tokens 16384
uv run --frozen --no-sync python "$project/shared/smoke.py" "http://127.0.0.1:$proxy_port/v1" --max-tokens 16384 --temperature 1 > "$scratch/inspect_smoke.json"
echo "PASS Inspect request through the proxy: $(head -c 300 "$scratch/inspect_smoke.json" | tr '\n' ' ')"
echo "proxy recorded $(wc -l < "$scratch/proxy/requests.jsonl") requests"
