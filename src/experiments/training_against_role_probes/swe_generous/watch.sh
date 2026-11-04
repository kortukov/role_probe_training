#!/bin/bash
# One-shot production continuation; never restart the superseded investigation.
set -euo pipefail
thread=${1:?thread UUID}
cd /data/cluster/users/kortukov/role_probe_training
manifest=experiment_reports/training_against_role_probes/swe_generous/pipeline.json
start=$SECONDS
while true; do
 state=$(python3 - "$manifest" <<'PY'
import json, subprocess, sys
m=json.load(open(sys.argv[1]))
r=subprocess.run(['sacct','-n','-P','-j',','.join(m['watch_jobs']),'--format=JobID,State'],capture_output=True,text=True)
if r.returncode: print('RETRY');sys.exit()
rows=[line.split('|')[:2] for line in r.stdout.splitlines() if '|' in line and '.' not in line.split('|')[0]]
failed=[f'{j}:{s}' for j,s in rows if s.split()[0] in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED'}]
if failed:print('FAILURE '+', '.join(failed))
elif any(j==m['ready_job'] and s=='COMPLETED' for j,s in rows):print('READY')
else:print('WAIT')
PY
 )
 if [[ "$state" == FAILURE* || "$state" == READY ]]; then reason="$state";break;fi
 if (( SECONDS-start >= 1800 ));then reason='30-minute production progress check';break;fi
 sleep 60
done
message="Automatic GENEROUS-BUDGET SWE continuation: $reason. Read swe_generous/pipeline.json and latest CURRENT_HANDOFF. User explicitly STOPPED model-card/gold investigation and authorized a matched stock Inspect seven-model including27k evaluation ASAP. Never restart old corrected-SWE diagnostic alarms or optimize toward60%. User restored 2000-message budget after rejecting30-message run. Gate1615441 validates unchanged sandbox/scorer. Never resume30-message cohort. Monitor/recover submitted standard cohorts without regenerating saved attempts; all500 denominators, separate old cohorts, whole-history probes. Publish validated task report as soon as all six task results exist, then add role metrics; preserve concurrent KL/on-policy changes and use isolated publication checkout. Inspect remaining report narrative for stale old SWE claims before push. Rearm this production watcher only after it exits if work remains running."
for attempt in 1 2 3 4 5;do
 if codex queue --thread "$thread" --message "$message";then date -u '+Production alarm delivered %Y-%m-%dT%H:%M:%SZ';exit 0;fi
 sleep 60
done
exit 1
