#!/bin/bash
#SBATCH --job-name=header-role-probes
#SBATCH --account=hx2-users
#SBATCH --partition=interactive-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=scratch:20G
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=experiment_reports/gptoss_role_probes/header_roles/logs/%x-%j.out
# Usage: sbatch header_roles_parallel.sbatch "FIT ARGUMENTS" "FIT ARGUMENTS" ...
# Runs one fit stage per argument string at the same time on the one GPU; the activations must
# already be cached. The partition runs one job per user, and an AdamW fit leaves most of the
# GPU idle. Each process keeps one layer in host memory (about 12 GB with the CUDA runtime), so
# four fit this job's 60 GB. Process N logs to logs/header-role-probes-JOB-N.out.
source src/experiments/gptoss_role_probes/hx2/common.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
run="uv run --project src/experiments/gptoss_role_probes/fit_env --frozen --no-sync python -u src/experiments/gptoss_role_probes/header_roles/fit.py"
logs=experiment_reports/gptoss_role_probes/header_roles/logs
pids=()
for arguments in "$@"; do
  # Unquoted on purpose: one argument string holds the options of one fit stage.
  $run fit $arguments > "$logs/$SLURM_JOB_NAME-$SLURM_JOB_ID-$((${#pids[@]} + 1)).out" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
