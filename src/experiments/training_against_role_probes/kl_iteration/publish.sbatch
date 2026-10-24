#!/bin/bash
#SBATCH --job-name=role-kl-publish
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
# User approved automatic report/aggregate publication to origin/main on 2026-10-25.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root=experiment_reports/training_against_role_probes
iteration="$root/kl_iteration"
exec 8> "$iteration/data-report.lock"
flock -w 300 8
for variant in agentic800 same_batch; do
 .tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/kl_iteration/finalize.py --root "$iteration/$variant"
done
.tools/bin/uv run --frozen --no-sync python - <<'PY'
import json
from pathlib import Path
root=Path('experiment_reports/training_against_role_probes/kl_iteration')
p=root/'RUNNING.md'
s=p.read_text().replace('Full training\njobs **1504830** (fixed 800) and **1504852** (same batch) are running. Benchmark\nresults remain pending until their dependent evaluation and report jobs finish.', 'Full training jobs **1504830** (fixed 800) and **1504852** (same batch),\nall three benchmark evaluations, and final result validation have completed.')
p.write_text(s)
for variant in ('agentic800','same_batch'):
 results=json.loads((root/variant/'post_training/results.json').read_text())
 public={}
 for name,row in results.items():
  score=row['probe']; task=row['task']
  public[name]={'task':task['metrics'][row['task_metric']],
   'evaluation_errors':task['sample_errors'],
   'role_accuracy':score['accuracy'],'roles':score['roles'],
   'per_role_recall':score['per_role_recall'],'confusion_counts':score['confusion_counts'],
   'histories_scored':score['scored_trajectories'],'total_histories':score['trajectories'],
   'unscorable_histories':len(score['excluded_trajectories'])}
 (root/variant/'publication_metrics.json').write_text(json.dumps(public,indent=2)+'\n')
PY
paths=("$root/README.md" "$iteration/RUNNING.md")
for variant in agentic800 same_batch; do
 folder="$iteration/$variant"
 paths+=("$folder/README.md" "$folder/APPENDIX.md" "$folder/publication_metrics.json")
done
# Markdown and aggregate numeric metrics only: no weights, histories or task text.
git diff --check -- "${paths[@]}"
git add -- "${paths[@]}"
if ! git diff --cached --quiet -- "${paths[@]}"; then
 git commit --only -m "Report AppWorld agentic and same-batch KL benchmark results" -- "${paths[@]}"
fi
# A separate worktree preserves the remote's existing files and publishes only
# the reviewed report paths, without pushing other local commits.
publication_dir=$(mktemp -d /tmp/role-kl-publication-XXXXXX)
trap 'git worktree remove --force "$publication_dir" 2>/dev/null || true' EXIT
git fetch origin main
git worktree add --detach "$publication_dir" origin/main
for path in "${paths[@]}"; do
 mkdir -p "$publication_dir/$(dirname "$path")"
 cp "$path" "$publication_dir/$path"
done
git -C "$publication_dir" add -- "${paths[@]}"
if ! git -C "$publication_dir" diff --cached --quiet; then
 git -C "$publication_dir" commit -m "Report AppWorld agentic and same-batch KL benchmark results"
fi
git -C "$publication_dir" push origin HEAD:main
