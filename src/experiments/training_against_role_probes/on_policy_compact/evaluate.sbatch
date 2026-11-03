#!/bin/bash
#SBATCH --job-name=role-compact-eval
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --exclude=vca-gpu-0701-03,vca-gpu-0701-05,vca-gpu-0702-02
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mode=${1:?prepare, app-retail or swe-finish}
root=${2:?report root}
code=src/experiments/training_against_role_probes
# Child scripts run in separate shells so their server/scratch traps finish
# before the next stage starts. Their SBATCH directives are not resubmitted.
case "$mode" in
 prepare)
  checkpoint=${3:?checkpoint}
  bash "$code/post_training/export.sbatch" "$checkpoint" "$root/data/post_training/model"
  bash "$code/post_training/probes.sbatch" fit --root "$root"
  ;;
 app-retail)
  name=${3:?model name}
  for shard in 0 1 2; do
   SLURM_ARRAY_TASK_ID=$shard bash "$code/kl_iteration/collect.sbatch" "$root" "$name"
  done
  bash "$code/kl_iteration/assemble_benchmark.sbatch" "$root" "$SLURM_JOB_ID" --benchmarks appworld_test tau2_retail
  for shard in 0 1; do
   SLURM_ARRAY_TASK_ID=$shard bash "$code/post_training/whole_score.sbatch" "$root" --overlength exclude
   benchmarks=(appworld_test tau2_retail)
   bash "$code/on_policy_evaluation/results.sbatch" "$root" "${benchmarks[$shard]}"
  done
  ;;
 swe-finish)
  collection=${3:?SWE collection job ID}
  bash "$code/swe_generous/assemble.sbatch" "$root" "$collection"
  bash "$code/swe_generous/score.sbatch" "$root"
  bash "$code/on_policy_evaluation/results.sbatch" "$root" swe_bench
  ;;
 *) echo "Unknown stage: $mode" >&2; exit 2;;
esac
