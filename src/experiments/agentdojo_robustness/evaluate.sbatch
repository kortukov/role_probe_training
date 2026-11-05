#!/bin/bash
#SBATCH --job-name=role-agentdojo
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=experiment_reports/prompt_injection/agentdojo_same_batch_27000/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
model_kind=${1:?base or trained}; mode=${2:-full}
root=experiment_reports/prompt_injection/agentdojo_same_batch_27000
project=src/experiments/eval_role_probes/agentic_eval/server_current
driver=src/experiments/agentdojo_robustness/driver
scratch=$(mktemp -d /tmp/role-agentdojo-${SLURM_JOB_ID}-XXXXXX)
export PATH="$PWD/.tools/bin:$PATH" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$scratch/tmp" TRITON_CACHE_DIR="$scratch/triton" VLLM_CACHE_ROOT="$scratch/vllm" TORCHINDUCTOR_CACHE_DIR="$scratch/inductor"
export HF_HOME="$PWD/.cache/huggingface" TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$PWD/$project/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH" FLASHINFER_WORKSPACE_BASE="$scratch" MAX_JOBS=4
mkdir -p "$TMPDIR"
worker=()
if [[ "$model_kind" == trained ]]; then
 model="$PWD/experiment_reports/training_against_role_probes/data_iteration/same_batch_27000/data/post_training/model"
 worker=(--worker-cls role_probe_training.post_training.vllm_worker.RoleWorker)
elif [[ "$model_kind" == base ]]; then
 model="$PWD/experiment_reports/training_against_role_probes/base_model/data/swe_native/model"
else exit 2; fi
out="$root/runs/$model_kind"
extra=()
if [[ "$mode" == smoke ]]; then out="$root/runs/smoke-$model_kind"; extra=(--smoke); fi
mkdir -p "$out"
port=$((18000+SLURM_JOB_ID%10000))
uv run --project "$project" --frozen --no-sync vllm serve "$model" \
 --served-model-name role-agentdojo --host 127.0.0.1 --port "$port" \
 --max-model-len 12288 --max-num-seqs 4 --gpu-memory-utilization 0.85 \
 --attention-backend TRITON_ATTN --moe-backend triton --tool-call-parser openai --enable-auto-tool-choice \
 "${worker[@]}" --kernel-config '{"enable_jit_warmup":false}' \
 --cudagraph-capture-sizes 1 2 4 --max-cudagraph-capture-size 4 > "$out/server-$SLURM_JOB_ID.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; rm -rf "$scratch"' EXIT
ready=0
for ((i=0;i<240;i++)); do
 if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
 kill -0 "$server_pid" || { tail -60 "$out/server-$SLURM_JOB_ID.log"; exit 1; }
 sleep 5
done
[[ "$ready" == 1 ]]
uv run --project src/experiments/eval_role_probes/agentic_eval/swe_bench/env --frozen --no-sync python src/experiments/training_against_role_probes/post_training/server_smoke.py --base-url "http://127.0.0.1:$port/v1" --model role-agentdojo > "$out/tool-check-$SLURM_JOB_ID.json"
pids=()
suites=(workspace slack travel banking)
if [[ "$mode" == smoke ]]; then suites=(banking); fi
for suite in "${suites[@]}"; do
 uv run --project "$driver" --frozen --no-sync python src/experiments/agentdojo_robustness/run.py --port "$port" --suite "$suite" --output "$out" "${extra[@]}" > "$out/$suite-$SLURM_JOB_ID.log" 2>&1 &
 pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [[ "$mode" == full ]]; then
 flock "$root/.summary.lock" uv run --project "$driver" --frozen --no-sync python src/experiments/agentdojo_robustness/summarize.py
fi
exit "$rc"
