# Shared hx2 job environment for the role-probe experiments. Source from sbatch scripts.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/hx2-weka/home/xy2222/.cache/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
# Node-local scratch: jobs must request `--gres=scratch:SIZE`; the prolog then mounts a
# private volume of that size at the job's /tmp ($TMPDIR). Without it /tmp is 1 GB.
scratch="${TMPDIR:-/tmp}/role-probe-$SLURM_JOB_ID"
mkdir -p "$scratch/tmp" "$scratch/triton" "$scratch/matplotlib"
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch/tmp" TRITON_CACHE_DIR="$scratch/triton" MPLCONFIGDIR="$scratch/matplotlib"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
