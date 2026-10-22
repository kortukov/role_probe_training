#!/bin/bash
#SBATCH --job-name=agentic-appworld
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=experiment_reports/agentic/appworld/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
source /etc/slurm/local_job_dir.sh
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export HF_HOME="$PWD/.cache/huggingface" VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p "$TMPDIR"
project=src/experiments/agentic_eval
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=8
run_dir="$PWD/experiment_reports/agentic/appworld/runs/$SLURM_JOB_ID"
mkdir -p "$run_dir"
port=$((18000+SLURM_JOB_ID%10000))
if [[ -n "${APPWORLD_PILOT_ROOT:-}" ]]; then
 uv run --project "$project/swe" --frozen --no-sync python - "$APPWORLD_PILOT_ROOT" <<'PYCODE'
import json,sys
from pathlib import Path
for directory in ('train-c1','train-c8'):
 root=Path(sys.argv[1])/directory
 summary=json.loads((root/'summary.json').read_text())
 assert summary['samples']==4 and summary['sample_errors']==0, summary
 rows=[json.loads(line) for line in (root/'trajectories.jsonl').open()]
 assert all(any(m['role']=='tool' for m in r['messages']) for r in rows)
 assert all(any(m.get('thinking') for m in r['messages']) for r in rows)
print('Training pilot passed: complete tool-bearing trajectories with reasoning, no sample errors.')
PYCODE
fi
server_started=$SECONDS
uv run --project "$project/server_current" --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
 --attention-backend TRITON_ATTN --moe-backend triton --tool-call-parser openai --enable-auto-tool-choice --enforce-eager \
 > "$run_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<180;i++)); do
 if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
 if ! kill -0 "$server_pid" 2>/dev/null; then tail -60 "$run_dir/server.log"; exit 1; fi
 sleep 5
done
[[ "$ready" == 1 ]]
printf '{"server_startup_seconds": %s}\n' "$((SECONDS-server_started))" > "$run_dir/startup.json"
if [[ "${1:-}" == "--pilot" ]]; then
 for concurrency in 1 8; do
  uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_run.py" \
   --base-url "http://127.0.0.1:$port/v1" --split train --samples 4 --concurrency "$concurrency" \
   --output "$run_dir/train-c$concurrency"
 done
else
 if [[ "${SWE_DIAGNOSTIC:-0}" == 1 ]]; then
  uv run --project "$project/swe" --frozen --no-sync python "$project/swe_replay.py" \
   --base-url "http://127.0.0.1:$port/v1" \
   --output "$run_dir/swe-request-diagnostic.json"
 fi
 for split in train test_normal test_challenge; do
  uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_run.py" \
   --base-url "http://127.0.0.1:$port/v1" --split "$split" --concurrency 8 --output "$run_dir/$split"
 done
fi
