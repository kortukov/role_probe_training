#!/bin/bash
#SBATCH --job-name=benchmark-probes
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=experiment_reports/agentic/probe_transfer/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /etc/slurm/local_job_dir.sh
export PATH="$PWD/.tools/bin:$PATH" HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1
export TMPDIR="$LOCAL_JOB_DIR/tmp" TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
input_file="$1"
output_dir="$2"
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none uv run --frozen --no-sync python -u src/experiments/agentic_probes/run.py \
  --input "$input_file" --output "$output_dir" \
  --probes experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz \
  experiment_reports/role_probes/gptoss20b_baseline/probes/layer_16.npz \
  experiment_reports/role_probes/gptoss20b_layers/probes/layer_4.npz \
  experiment_reports/role_probes/gptoss20b_layers/probes/layer_8.npz \
  experiment_reports/role_probes/gptoss20b_layers/probes/layer_20.npz
