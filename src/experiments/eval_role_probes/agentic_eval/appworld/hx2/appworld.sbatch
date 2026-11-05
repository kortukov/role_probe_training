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
# hx2 port of appworld_job.sbatch: one vLLM replica, one GPU.
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
 --attention-backend TRITON_ATTN --moe-backend triton --tool-call-parser openai --enable-auto-tool-choice \
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
if [[ "${1:-}" == "--pilot" ]]; then
 for concurrency in 1 8; do
  uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_run.py" \
   --base-url "http://127.0.0.1:$port/v1" --split train --samples 4 --concurrency "$concurrency" \
   --output "$run_dir/train-c$concurrency"
 done
else
 for split in "$@"; do
  uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_run.py" \
   --base-url "http://127.0.0.1:$port/v1" --split "$split" --concurrency 8 --output "$run_dir/$split"
 done
fi
