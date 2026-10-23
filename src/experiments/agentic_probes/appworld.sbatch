#!/bin/bash
#SBATCH --job-name=appworld-probes
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=experiment_reports/agentic/appworld_probe_transfer/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export PATH="$PWD/.tools/bin:$PATH" HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
input_file="experiment_reports/agentic/appworld/data/test_normal.jsonl"
output_dir="experiment_reports/agentic/appworld_probe_transfer/predictions/shard$SLURM_ARRAY_TASK_ID"
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none uv run --frozen --no-sync python -u src/experiments/agentic_probes/run.py \
  --input "$input_file" --output "$output_dir" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 4 \
  --probes experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz \
  experiment_reports/role_probes/gptoss20b_baseline/probes/layer_16.npz \
  experiment_reports/role_probes/gptoss20b_layers/probes/layer_4.npz \
  experiment_reports/role_probes/gptoss20b_layers/probes/layer_8.npz \
  experiment_reports/role_probes/gptoss20b_layers/probes/layer_20.npz
