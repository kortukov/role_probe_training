#!/bin/bash
#SBATCH --job-name=ladder-eval
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
#SBATCH --output=experiment_reports/training_against_role_probes/on_policy_progress/logs/%x-%j.out
# sbatch --job-name=eval-NAME evaluate.sbatch NAME CHECKPOINT [five|header]
# The last argument is the role set of the fresh evaluation probes (default header, ten roles).
# Fast evaluation of one trained checkpoint: export it, fit fresh ten-role probes (layers 12 and 16) on
# the trained model, run AppWorld test and tau2 retail through the Harmony proxy at
# temperature 1, score the whole histories with the fresh probes, write eval/results.json.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
name=${1:?run name}; checkpoint=${2:?checkpoint folder}
root=experiment_reports/training_against_role_probes/on_policy_progress/$name/eval
code=src/experiments/training_against_role_probes
project=src/experiments/eval_role_probes/agentic_eval
: "${AZURE_OPENAI_API_KEY:?Azure credentials for the retail user simulator must be inherited}" "${AZURE_OPENAI_ENDPOINT:?}"
job_scratch=$(mktemp -d /tmp/ladder-eval-${SLURM_JOB_ID}-XXXXXX)
export LOCAL_JOB_DIR="$job_scratch"
export PATH="$PWD/.tools/bin:$PATH" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton" TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_DIR/inductor"
export HF_HOME="$PWD/.cache/huggingface" HF_DATASETS_CACHE="$PWD/.cache/datasets" INSPECT_EVALS_CACHE_DIR="$PWD/.cache/inspect-evals"
export VLLM_CACHE_ROOT="$LOCAL_JOB_DIR/vllm" TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken" VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH" FLASHINFER_WORKSPACE_BASE="$LOCAL_JOB_DIR" MAX_JOBS=8
export APPTAINER_CACHEDIR="$LOCAL_JOB_DIR/apptainer-cache" APPTAINER_TMPDIR="$LOCAL_JOB_DIR/apptainer-tmp"
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$root"
server_pid=; proxy_pid=
trap 'kill $proxy_pid $server_pid 2>/dev/null || true; rm -rf "$job_scratch"' EXIT

# 1. Export the adapter and fit fresh probes on the trained model.
model="$PWD/$root/data/post_training/model"
uv run --frozen --no-sync python -m role_probe_training.post_training.export_model --checkpoint "$checkpoint" --output "$model"
uv run --frozen --no-sync python -m role_probe_training.post_training.probes fit --root "$root" --layers 12 16 --roles "${3:-header}"
revision=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["adapter_sha256"])' "$model/export.json")

# 2. Serve the trained model. It is served under the base model's name because the retail
#    runner asks for that name; results.json records the checkpoint.
run="$root/runs"; mkdir -p "$run"
port=$((18000+SLURM_JOB_ID%10000)); proxy_port=$((port+10000))
source "$code/kl_iteration/serve_retry.sh"
start_server uv run --frozen --no-sync vllm serve "$model" \
 --served-model-name openai/gpt-oss-20b --host 127.0.0.1 --port "$port" \
 --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
 --attention-backend TRITON_ATTN --moe-backend triton \
 --worker-cls role_probe_training.post_training.vllm_worker.RoleWorker \
 --kernel-config '{"enable_jit_warmup":false}' --cudagraph-capture-sizes 1 2 4 8 16 --max-cudagraph-capture-size 16
proxy() {  # proxy OUTPUT MAX_TOKENS CONTEXT_LIMIT
 uv run --frozen --no-sync python "$project/shared/harmony_proxy.py" --vllm-url "http://127.0.0.1:$port" \
  --model openai/gpt-oss-20b --output "$1/proxy" --port "$proxy_port" --reasoning-effort high \
  --max-tokens "$2" --context-limit "$3" --temperature 1 > "$1.proxy.log" 2>&1 &
 proxy_pid=$!
 for ((i=0;i<60;i++)); do
  curl --silent --fail "http://127.0.0.1:$proxy_port/v1/models" >/dev/null && return 0
  kill -0 "$proxy_pid" 2>/dev/null || { cat "$1.proxy.log"; exit 1; }
  sleep 1
 done
 exit 1
}

# 3. AppWorld test (168 tasks), then tau2 retail (114 tasks).
proxy "$run/appworld_test" 8192 32768
uv run --frozen --no-sync python "$project/appworld/appworld_run.py" \
 --base-url "http://127.0.0.1:$proxy_port/v1" --requests "$run/appworld_test/proxy/requests.jsonl" \
 --split test_normal --concurrency 8 --temperature 1 --model-revision "$revision" --output "$run/appworld_test"
kill "$proxy_pid"; wait "$proxy_pid" 2>/dev/null || true
proxy "$run/tau2_retail" 16384 131072
uv run --frozen --no-sync python "$project/tau2_retail/run.py" --all --temperature 1 \
 --base-url "http://127.0.0.1:$proxy_port/v1" --requests "$run/tau2_retail/proxy/requests.jsonl" --output "$run/tau2_retail"
kill "$proxy_pid" "$server_pid"; wait "$proxy_pid" "$server_pid" 2>/dev/null || true
proxy_pid=; server_pid=

# 4. Score the whole histories with the fresh probes and summarise.
for benchmark in appworld_test tau2_retail; do
 uv run --frozen --no-sync python -m role_probe_training.post_training.full_history --root "$root" \
  --input "$run/$benchmark/trajectories.jsonl" --output "$root/predictions/$benchmark" --overlength exclude
done
uv run --frozen --no-sync python "$code/on_policy_progress/results.py" --root "$root" --name "$name" --checkpoint "$checkpoint"
