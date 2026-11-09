#!/bin/bash
#SBATCH --job-name=role-kl
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
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
data=experiment_reports/training_against_role_probes/data/kl
server_log="experiment_reports/training_against_role_probes/logs/role-kl-$SLURM_JOB_ID.server.log"
mkdir -p "$data" "$(dirname "$server_log")"
port=$((18000 + SLURM_JOB_ID % 10000))
# vLLM only samples token IDs; generate_kl.py (root environment) renders the prompts with the
# repository's Harmony renderer, as the Harmony proxy does for trajectories.
uv run --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
    --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
    --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
    --attention-backend TRITON_ATTN --moe-backend triton --kernel-config '{"enable_jit_warmup":false}' \
    --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16 > "$server_log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<180;i++)); do
    if curl --silent --fail "http://127.0.0.1:$port/health" > /dev/null; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$server_log"; exit 1; fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then echo 'Server startup timeout'; exit 1; fi
uv run --frozen --no-sync python src/experiments/training_against_role_probes/generate_kl.py \
    --input "$data/prompts.json" --output "$data/anchors.jsonl" --vllm-url "http://127.0.0.1:$port"
