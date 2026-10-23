#!/bin/bash
#SBATCH --job-name=role-learning-report
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
report=experiment_reports/training_against_role_probes
run=${1:?Pass the completed training run directory}
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/report_training.py \
  --run "$run" --report "$report" --finished-job
# The user requested publication of reports, excluding data and checkpoints.
# --only isolates this commit from other agents' staged changes.
git diff --check -- "$report/README.md" "$report/APPENDIX.md"
git add -- "$report/training_results.json"
if ! git diff HEAD --quiet -- "$report/README.md" "$report/APPENDIX.md" "$report/training_results.json"; then
  git commit --only -m 'Report AppWorld role-recoverability training results' -- \
    "$report/README.md" "$report/APPENDIX.md" "$report/training_results.json"
  git push
fi
