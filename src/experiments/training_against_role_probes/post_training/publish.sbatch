#!/bin/bash
#SBATCH --job-name=role-publish-results
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root=${1:?report root}
# report.py already requires complete cohorts, a fresh probe, and matching hashes.
[[ -s "$root/post_training/results.json" ]]
exec 9> /tmp/role-probe-results-publish.lock
flock -w 120 9
paths=("$root/README.md" "$root/APPENDIX.md" "$root/post_training")
git diff --check -- "${paths[@]}"
git add -- "${paths[@]}"
if ! git diff --cached --quiet -- "${paths[@]}"; then
 git commit --only -m "Report fresh benchmark evaluation: $root" -- "${paths[@]}"
fi
git push
