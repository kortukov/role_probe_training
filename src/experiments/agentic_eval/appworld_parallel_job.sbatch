#!/bin/bash
#SBATCH --job-name=appworld-parallel
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=experiment_reports/agentic/appworld/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
source /etc/slurm/local_job_dir.sh
export VLLM_USE_FLASHINFER_SAMPLER=0
project=src/experiments/agentic_eval
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="$PWD/.cache/huggingface" VLLM_CACHE_ROOT="$PWD/.cache/vllm-agentic"
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MAX_JOBS=8
run_dir="$PWD/experiment_reports/agentic/appworld/runs/$SLURM_JOB_ID"
mkdir -p "$run_dir"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#devices[@]}" == 2 ]]
if [[ -n "${APPWORLD_PILOT_ROOT:-}" ]]; then
 uv run --project "$project/swe" --frozen --no-sync python - "$APPWORLD_PILOT_ROOT" <<'PYCODE'
import json,sys
from pathlib import Path
for part in ('train-c1','train-c8'):
 summary=json.loads((Path(sys.argv[1])/part/'summary.json').read_text())
 assert summary['samples']==4 and summary['sample_errors']==0, summary
print('One-GPU pilot validated.')
PYCODE
fi
server_pids=()
cleanup() { for pid in "${server_pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
port=$((18000+SLURM_JOB_ID%10000))
server_started=$SECONDS
for shard in 0 1; do
 mkdir -p "$LOCAL_JOB_DIR/gpu$shard/tmp"
 CUDA_VISIBLE_DEVICES="${devices[$shard]}" TMPDIR="$LOCAL_JOB_DIR/gpu$shard/tmp" \
 TRITON_CACHE_DIR="$LOCAL_JOB_DIR/gpu$shard/triton" FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR/gpu$shard" \
 uv run --project "$project/server_current" --frozen --no-sync vllm serve "$PWD/.cache/models/gpt-oss-20b" \
  --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$((port+shard))" \
  --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
  --attention-backend TRITON_ATTN --moe-backend triton --tool-call-parser openai \
  --enable-auto-tool-choice --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16 > "$run_dir/server-$shard.log" 2>&1 &
 server_pids+=("$!")
done
for shard in 0 1; do
 ready=0
 for ((i=0;i<180;i++)); do
  if curl --silent --fail "http://127.0.0.1:$((port+shard))/health" >/dev/null; then ready=1; break; fi
  if ! kill -0 "${server_pids[$shard]}" 2>/dev/null; then tail -50 "$run_dir/server-$shard.log"; exit 1; fi
  sleep 5
 done
 [[ "$ready" == 1 ]]
done
printf '{"server_startup_seconds": %s, "gpus": 2}\n' "$((SECONDS-server_started))" > "$run_dir/startup.json"
# Request-only SWE diagnosis does not execute untrusted tools or affect rollout timing.
uv run --project "$project/swe" --frozen --no-sync python "$project/swe_replay.py" \
 --base-url "http://127.0.0.1:$port/v1" --output "$run_dir/swe-request-diagnostic.json"
for split in pilot train test_normal test_challenge; do
 split_args=(--split "$split")
 if [[ "$split" == pilot ]]; then split_args=(--split train --samples 4); fi
 client_pids=()
 for shard in 0 1; do
  uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_run.py" \
   --base-url "http://127.0.0.1:$((port+shard))/v1" "${split_args[@]}" --concurrency 8 \
   --shard-index "$shard" --shard-count 2 --output "$run_dir/$split/shard$shard" \
   > "$run_dir/$split-$shard.log" 2>&1 &
  client_pids+=("$!")
 done
 for pid in "${client_pids[@]}"; do wait "$pid"; done
 if [[ "$split" == pilot ]]; then
  uv run --project "$project/swe" --frozen --no-sync python - "$run_dir/pilot" <<'PYCODE'
import json,sys
from pathlib import Path
for shard in (0,1):
 root=Path(sys.argv[1])/f'shard{shard}'
 summary=json.loads((root/'summary.json').read_text())
 assert summary['samples']==2 and summary['sample_errors']==0, summary
 rows=[json.loads(line) for line in (root/'trajectories.jsonl').open()]
 assert all(any(m['role']=='tool' for m in r['messages']) for r in rows)
 assert all(any(m.get('thinking') for m in r['messages']) for r in rows)
print('Two-GPU pilot validated.')
PYCODE
 fi
done
uv run --project "$project/swe" --frozen --no-sync python "$project/appworld_report.py" \
 --dataset "train=$run_dir/train/shard0,$run_dir/train/shard1" \
 --dataset "test_normal=$run_dir/test_normal/shard0,$run_dir/test_normal/shard1" \
 --dataset "test_challenge=$run_dir/test_challenge/shard0,$run_dir/test_challenge/shard1" \
 --output experiment_reports/agentic/appworld
