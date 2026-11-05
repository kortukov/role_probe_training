#!/bin/bash
#SBATCH --job-name=norobots-generate
#SBATCH --account=hx2-users
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:60G
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=experiment_reports/eval_role_probes/chat_eval/norobots/logs/%x-%j.out
# vLLM serves GPT-OSS-20B natively; the client (root env plus a pinned openai overlay) sends exact Harmony prefixes.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
scratch="${TMPDIR:-/tmp}/norobots-$SLURM_JOB_ID"; mkdir -p "$scratch/tmp" "$scratch/triton"
export TMPDIR="$scratch/tmp" TRITON_CACHE_DIR="$scratch/triton" FLASHINFER_WORKSPACE_BASE="$scratch"
export HF_HOME=/hx2-weka/home/xy2222/.cache/huggingface HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic" TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MAX_JOBS=8 VLLM_USE_FLASHINFER_SAMPLER=0
project=src/experiments/eval_role_probes/agentic_eval
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
root=experiment_reports/eval_role_probes/chat_eval/norobots
port=$((18000+SLURM_JOB_ID%10000))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run --project "$project/server_current" --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.85 \
 --attention-backend TRITON_ATTN --moe-backend triton \
 --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 32 64 --max-cudagraph-capture-size 64 \
 > "$root/logs/server-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; rm -rf "$scratch"' EXIT
for ((i=0;i<240;i++)); do
 curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null && break
 kill -0 "$server_pid" 2>/dev/null || { tail -60 "$root/logs/server-$SLURM_JOB_ID.log"; exit 1; }
 sleep 5
done
curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null || { echo 'Server startup timeout'; exit 1; }
# Fail fast on two prompts before the full run; inspect their raw continuations in the log.
uv run --frozen --no-sync --with 'openai==3.36.0' python src/experiments/eval_role_probes/chat_eval/norobots/generate.py \
 --base-url "http://127.0.0.1:$port/v1" --limit 2 --concurrency 2 --output "$root/artifacts/smoke/generated.jsonl"
python3 -c "import json; [print(json.dumps({k: r[k] for k in ('finish_reason','stop_reason','completion_tokens','raw')})[:700]) for r in map(json.loads, open('$root/artifacts/smoke/generated.jsonl'))]"
uv run --frozen --no-sync --with 'openai==3.36.0' python src/experiments/eval_role_probes/chat_eval/norobots/generate.py \
 --base-url "http://127.0.0.1:$port/v1" --output "$root/artifacts/generated.jsonl" "$@"
