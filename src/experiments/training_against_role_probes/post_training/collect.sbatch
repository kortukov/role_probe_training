#!/bin/bash
#SBATCH --job-name=role-new-rollouts
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-6%4
#SBATCH --exclude=vca-gpu-0701-03
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
job_scratch=$(mktemp -d /tmp/role-eval-${SLURM_JOB_ID}-XXXXXX)
export LOCAL_JOB_DIR="$job_scratch"
trap 'rm -rf "$job_scratch"' EXIT
root=${1:?report root}; name=${2:?model name}; mode=${3:-full}
project=src/experiments/agentic_eval
export PATH="$PWD/.tools/bin:$PATH" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/datasets"
export INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals" SWE_IMAGE_CACHE="$PWD/.cache/swe-images"
export SWE_SERVER_PROJECT=server_current VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH" FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=8
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache" APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
model="$PWD/$root/data/post_training/model"
revision=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["adapter_sha256"])' "$model/export.json")
served="openai/gpt-oss-20b-role-$name"
idx=${SLURM_ARRAY_TASK_ID:-0}
run="$root/runs/post_training/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/shard$idx"
mkdir -p "$run"
port=$((18000+SLURM_JOB_ID%10000))
uv run --project "$project/server_current" --frozen --no-sync vllm serve "$model" \
 --served-model-name "$served" --host 127.0.0.1 --port "$port" \
 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
 --attention-backend TRITON_ATTN --moe-backend triton --tool-call-parser openai --enable-auto-tool-choice \
 --worker-cls role_probe_training.post_training.vllm_worker.RoleWorker \
 --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16 \
 > "$run/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; rm -rf "$job_scratch"' EXIT
ready=0
for ((i=0;i<180;i++)); do
 if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$run/server.log"; exit 1; fi
 sleep 5
done
[[ "$ready" == 1 ]]
uv run --project "$project/swe" --frozen --no-sync python src/experiments/training_against_role_probes/post_training/server_smoke.py \
 --base-url "http://127.0.0.1:$port/v1" --model "$served" > "$run/smoke.json"
if [[ "$mode" == smoke ]]; then exit 0; fi
common=(--base-url "http://127.0.0.1:$port/v1" --model-id "$served" --model-revision "$revision" --seed 123)
if (( idx < 2 )); then
 uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_run.py" \
  "${common[@]}" --split test_normal --shard-index "$idx" --shard-count 2 --concurrency 8 --output "$run/appworld_test"
elif (( idx == 2 )); then
 : "${AZURE_OPENAI_API_KEY:?Azure credentials must be inherited}" "${AZURE_OPENAI_ENDPOINT:?}"
 uv run --project "$project/swe" --frozen --no-sync python "$project/swe_run.py" \
  "${common[@]}" --benchmark tau2_retail --all --concurrency 16 --time-limit 1800 --max-tokens 16384 \
  --token-limit 500000 --temperature 0 --user-model openai/azure/gpt-4.1 --output "$run/tau2_retail"
else
 uv run --project "$project/swe" --frozen --no-sync python "$project/swe_run.py" \
  "${common[@]}" --benchmark swe_bench --all --shard-index "$((idx-3))" --shard-count 4 \
  --concurrency 8 --time-limit 1800 --max-tokens 8192 --token-limit 2000000 --message-limit 200 \
  --temperature 1 --output "$run/swe_bench"
fi
