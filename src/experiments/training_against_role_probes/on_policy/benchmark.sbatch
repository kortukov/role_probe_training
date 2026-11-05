#!/bin/bash
#SBATCH --job-name=role-serving-bench
#SBATCH --account=gpu
#SBATCH --partition=ber-testing
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH --array=0-1%1
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy/appworld2700/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" VLLM_CACHE_ROOT="$LOCAL_JOB_DIR/vllm"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache" APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
root=experiment_reports/training_against_role_probes/on_policy/appworld2700
run="$root/runs/serving_benchmark/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID"
mkdir -p "$run"
if (( SLURM_ARRAY_TASK_ID == 0 )); then
 .tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync \
  python src/experiments/training_against_role_probes/on_policy/benchmark_hf.py --output "$run"
else
 project=src/experiments/eval_role_probes/agentic_eval/server_current
 export CUDA_HOME="$PWD/$project/.venv/lib/python3.12/site-packages/nvidia/cu13"
 export PATH="$CUDA_HOME/bin:$PATH" FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=4
 model="$PWD/experiment_reports/training_against_role_probes/kl_iteration/same_batch/data/post_training/model"
 port=$((18000+SLURM_JOB_ID%10000))
 started=$SECONDS
 .tools/bin/uv run --project "$project" --frozen --no-sync vllm serve "$model" \
  --served-model-name on-policy-role-model --host 127.0.0.1 --port "$port" \
  --max-model-len 32768 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  --no-enable-prefix-caching --generation-config vllm \
  --attention-backend TRITON_ATTN --moe-backend triton --tool-call-parser openai --enable-auto-tool-choice \
  --worker-cls role_probe_training.post_training.vllm_worker.RoleWorker \
  --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 \
  --max-cudagraph-capture-size 8 > "$run/server.log" 2>&1 &
 server_pid=$!
 trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
 ready=0
 for ((i=0;i<180;i++)); do
  if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
  sleep 5
 done
 if (( ready == 0 )); then tail -30 "$run/server.log"; exit 1; fi
 .tools/bin/uv run --frozen --no-sync python -c \
  'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); meta=json.loads(Path(sys.argv[3]).read_text()); (p/"startup.json").write_text(json.dumps(dict(backend="vLLM",load_seconds=int(sys.argv[2]),adapter_sha256=meta["adapter_sha256"]),indent=2)+"\n")' \
  "$run" "$((SECONDS-started))" "$model/export.json"
 for phase in warmup micro episodes; do
  .tools/bin/uv run --project src/experiments/eval_role_probes/agentic_eval/swe_bench/env --frozen --no-sync \
   python src/experiments/training_against_role_probes/on_policy/benchmark_client.py \
   --url "http://127.0.0.1:$port" --model on-policy-role-model --output "$run" --phase "$phase"
 done
fi
