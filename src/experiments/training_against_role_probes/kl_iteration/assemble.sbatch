#!/bin/bash
#SBATCH --job-name=role-kl-assemble
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
root=${1:?root}; collection=${2:?collection}; swe=${3:?SWE collection}
code=src/experiments/training_against_role_probes/post_training
.tools/bin/uv run --frozen --no-sync python "$code/assemble.py" --root "$root" --collection-id "$collection" --benchmarks appworld_test tau2_retail
.tools/bin/uv run --frozen --no-sync python "$code/assemble_swe_native.py" --root "$root" --collection-id "$swe"
.tools/bin/uv run --frozen --no-sync python - "$root" <<'PY'
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
p=root/'post_training/task_results.json'
tasks=json.loads(p.read_text())
tasks['swe_bench']=json.loads((root/'post_training/swe_native_task_results.json').read_text())
p.write_text(json.dumps(tasks,indent=2)+'\n')
(root/'data/post_training/swe_bench').symlink_to('../swe_native/evaluation',target_is_directory=True)
PY
