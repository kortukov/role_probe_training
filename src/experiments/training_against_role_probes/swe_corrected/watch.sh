#!/bin/bash
# One-shot continuation of the explicitly authorized corrected SWE evaluation.
set -euo pipefail
thread=${1:?thread UUID}
cd /data/cluster/users/kortukov/role_probe_training
manifest=experiment_reports/training_against_role_probes/swe_corrected/pipeline.json
start=$SECONDS
while true; do
 state=$(python3 - "$manifest" <<'PY'
import json,subprocess,sys
m=json.load(open(sys.argv[1]))
r=subprocess.run(['sacct','-n','-P','-j',','.join(m['watch_jobs']),'--format=JobID,State'],capture_output=True,text=True)
if r.returncode: print('RETRY');sys.exit()
rows=[s.split('|')[:2] for s in r.stdout.splitlines() if '|' in s and '.' not in s.split('|')[0]]
failed=[f'{j}:{s}' for j,s in rows if s.split()[0] in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED'}]
if failed:print('FAILURE '+', '.join(failed))
elif any(j==m['ready_job'] and s=='COMPLETED' for j,s in rows):print('READY')
else:print('WAIT')
PY
 )
 if [[ "$state" == FAILURE* || "$state" == READY ]]; then reason="$state";break;fi
 if (( SECONDS-start >= 1800 ));then reason='30-minute progress check';break;fi
 sleep 60
done
message="Automatic corrected SWE evaluation alarm requested by the user: $reason. Read experiment_reports/training_against_role_probes/swe_corrected/pipeline.json and inspect current jobs. Continue the authorized task: validate the corrected base-model evaluation, investigate remaining failures if far below the model-card 60%, then re-evaluate ALL SIX reported models with the validated setup and compute task success plus whole-history role-probe accuracy. Preserve concurrent KL changes, retain every model attempt, and do not mix old and new cohorts. No duplicate evaluations. Update and push the complete report after validation. If work is still running, rearm src/experiments/training_against_role_probes/swe_corrected/watch.sh with this same thread UUID."
for attempt in 1 2 3 4 5;do
 if codex queue --thread "$thread" --message "$message";then date -u '+Alarm delivered %Y-%m-%dT%H:%M:%SZ';exit 0;fi
 sleep 60
done
exit 1
