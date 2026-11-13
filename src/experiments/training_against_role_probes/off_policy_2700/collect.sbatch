#!/bin/bash
#SBATCH --job-name=offpolicy2700-collect
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-30%8
#SBATCH --output=experiment_reports/training_against_role_probes/off_policy_2700/logs/%x-%A_%a.out
# Base-model AppWorld rollouts for off-policy training, through the Harmony proxy.
# Array tasks 0-29: the 90 training tasks once each, seed 8301 + task index (2,700 rollouts).
# Array task 30: 8 dev tasks, seed 9301, the validation set of the trainer.
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
root=experiment_reports/training_against_role_probes/off_policy_2700
run_dir="$PWD/$root/runs/collection/$SLURM_ARRAY_JOB_ID/shard$SLURM_ARRAY_TASK_ID"
mkdir -p "$run_dir"
if (( SLURM_ARRAY_TASK_ID < 30 )); then split=train; samples=90; seed=$((8301 + SLURM_ARRAY_TASK_ID))
else split=dev; samples=8; seed=9301; fi
port=$((18000+SLURM_JOB_ID%10000))
proxy_port=$((port+10000))  # never another job's vLLM port (18000-27999)
uv run --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
 --attention-backend TRITON_ATTN --moe-backend triton \
 --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16 \
 > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" ${proxy_pid:-} 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<240;i++)); do
 if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$run_dir/server.log"; exit 1; fi
 sleep 5
done
[[ "$ready" == 1 ]] || { echo 'Server startup timeout'; tail -60 "$run_dir/server.log"; exit 1; }
# The proxy renders prompts with oss-harmony, samples token IDs from vLLM and records them;
# its settings must equal what the runner sends (temperature 1, 8,192 output tokens).
uv run --frozen --no-sync python "$project/shared/harmony_proxy.py" --vllm-url "http://127.0.0.1:$port" \
 --model openai/gpt-oss-20b --output "$run_dir/$split/proxy" --port "$proxy_port" --reasoning-effort high \
 --max-tokens 8192 --context-limit 32768 --temperature 1 > "$run_dir/proxy.log" 2>&1 &
proxy_pid=$!
for ((i=0;i<60;i++)); do
 curl --silent --fail "http://127.0.0.1:$proxy_port/v1/models" >/dev/null && break
 kill -0 "$proxy_pid" 2>/dev/null || { cat "$run_dir/proxy.log"; exit 1; }
 sleep 1
done
uv run --frozen --no-sync python "$project/appworld/appworld_run.py" \
 --base-url "http://127.0.0.1:$proxy_port/v1" --requests "$run_dir/$split/proxy/requests.jsonl" \
 --split "$split" --samples "$samples" --seed "$seed" --concurrency 8 --temperature 1 --output "$run_dir/$split"
