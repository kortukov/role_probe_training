#!/bin/bash
#SBATCH --job-name=role-serve-check
#SBATCH --account=gpu
#SBATCH --partition=ber-testing
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=vca-gpu-0701-03
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
root=${1:?report root}
export TMPDIR="$LOCAL_JOB_DIR/tmp" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" OMP_NUM_THREADS=4
mkdir -p "$TMPDIR"
script=src/experiments/training_against_role_probes/post_training/check_serving.py
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python "$script" reference --root "$root"
project=src/experiments/agentic_eval
export CUDA_HOME="$PWD/$project/server_current/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH" MAX_JOBS=8 VLLM_USE_FLASHINFER_SAMPLER=0
port=$((18000+SLURM_JOB_ID%10000))
.tools/bin/uv run --project "$project/server_current" --frozen --no-sync vllm serve "$PWD/$root/data/post_training/model" --served-model-name check --host 127.0.0.1 --port "$port" --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization .8 --enforce-eager --attention-backend TRITON_ATTN --moe-backend triton --worker-cls role_probe_training.post_training.vllm_worker.RoleWorker --kernel-config '{"enable_jit_warmup":false}' > "$root/data/post_training/check_server.log" 2>&1 &
pid=$!; trap 'kill "$pid" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<120;i++)); do
 if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1;break;fi
 kill -0 "$pid" || exit 1
 sleep 5
done
[[ "$ready" == 1 ]]
.tools/bin/uv run --project "$project/swe" --frozen --no-sync python "$script" compare --root "$root" --base-url "http://127.0.0.1:$port/v1" --model check
