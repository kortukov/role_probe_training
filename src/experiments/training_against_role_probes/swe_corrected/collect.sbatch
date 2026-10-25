#!/bin/bash
#SBATCH --job-name=role-swe-corrected
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root=${1:?report root}; mode=${2:-full}
project=src/experiments/agentic_eval
native_scratch=$(mktemp -d /tmp/swe-native-${SLURM_JOB_ID}-XXXXXX)
export LOCAL_JOB_DIR="$native_scratch"
trap 'rm -rf "$native_scratch"' EXIT
export PATH="$PWD/.tools/bin:$PATH" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$native_scratch/tmp" TRITON_CACHE_DIR="$native_scratch/triton"
export HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/datasets"
export INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals" SWE_IMAGE_CACHE="$PWD/.cache/swe-images"
export SWE_SERVER_PROJECT=server_current VLLM_CACHE_ROOT="$native_scratch/vllm" TORCHINDUCTOR_CACHE_DIR="$native_scratch/inductor"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH" FLASHINFER_WORKSPACE_BASE="$native_scratch" MAX_JOBS=8
export APPTAINER_CACHEDIR="$native_scratch/apptainer-cache" APPTAINER_TMPDIR="$native_scratch/apptainer-tmp"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
worker=()
if [[ "$root" == */base_model ]]; then
 model="$PWD/$root/data/swe_native/model"
 revision=6cee5e81ee83917806bbde320786a8fb61efebee
else
 model="$PWD/$root/data/post_training/model"
 revision=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["adapter_sha256"])' "$model/export.json")
 worker=(--worker-cls role_probe_training.post_training.vllm_worker.RoleWorker)
fi
served=openai/gpt-oss-20b-swe-native
idx=${SLURM_ARRAY_TASK_ID:-0}
run="$root/runs/swe_corrected/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/shard$idx"
mkdir -p "$run"
port=$((18000+SLURM_JOB_ID%10000))
# Retry transient imports/server initialization only, before any benchmark call.
start_server() {
 local attempt i ready
 for attempt in 1 2 3; do
  "$@" > "$run/server.log" 2>&1 &
  server_pid=$!
  ready=0
  for ((i=0;i<180;i++)); do
   if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
   if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
   sleep 5
  done
  if [[ "$ready" == 1 ]]; then return 0; fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  cp "$run/server.log" "$run/server-startup-failure-$attempt.log"
  tail -12 "$run/server.log"
  sleep 5
 done
 echo 'Server failed all three startup attempts before benchmark generation.' >&2
 return 1
}

start_server uv run --project "$project/server_current" --frozen --no-sync vllm serve "$model" \
 --served-model-name "$served" --host 127.0.0.1 --port "$port" \
 --max-model-len 131072 --max-num-seqs 4 --gpu-memory-utilization 0.85 \
 --attention-backend TRITON_ATTN --moe-backend triton --tool-call-parser openai --enable-auto-tool-choice \
 "${worker[@]}" --kernel-config '{"enable_jit_warmup":false}' \
 --cudagraph-capture-sizes 1 2 4 --max-cudagraph-capture-size 4
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; rm -rf "$native_scratch"' EXIT
uv run --project "$project/swe" --frozen --no-sync python src/experiments/training_against_role_probes/post_training/native_context_check.py \
 --base-url "http://127.0.0.1:$port/v1" --model "$served" > "$run/context_check.json"
uv run --project "$project/swe" --frozen --no-sync python src/experiments/training_against_role_probes/post_training/server_smoke.py \
 --base-url "http://127.0.0.1:$port/v1" --model "$served" > "$run/tool_check.json"
selection=(--all --shard-index "$idx" --shard-count 4)
if [[ "$mode" == pilot ]]; then
 selection=(--sample-id astropy__astropy-13398 --sample-id astropy__astropy-14182 --sample-id django__django-11119 --sample-id django__django-14155 --sample-id sympy__sympy-13878 --sample-id scikit-learn__scikit-learn-9288)
elif [[ "$mode" != full ]]; then
 echo 'Mode must be pilot or full'; exit 1
fi
uv run --project "$project/swe" --frozen --no-sync python src/experiments/training_against_role_probes/swe_corrected/run.py \
 --base-url "http://127.0.0.1:$port/v1" --model-id "$served" --model-revision "$revision" \
 --benchmark swe_bench "${selection[@]}" --seed 123 --reasoning-effort high \
 --concurrency 4 --time-limit 7200 --max-tokens 16384 --token-limit 10000000 --message-limit 2000 \
 --temperature 1 --output "$run/swe_bench"
