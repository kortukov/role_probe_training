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
elif all(any(j==ready and s=='COMPLETED' for j,s in rows) for ready in m.get('ready_jobs',[m['ready_job']])):print('READY')
else:print('WAIT')
PY
 )
 if [[ "$state" == FAILURE* || "$state" == READY ]]; then reason="$state";break;fi
 if (( SECONDS-start >= 1800 ));then reason='30-minute progress check';break;fi
 sleep 60
done
message="Automatic SWE continuation: $reason. Read CURRENT pipeline.json and latest CURRENT_HANDOFF/APPENDIX/RUNNING; earlier queued alarms may be stale. Inspect combined repair gold survey1615435 ALL4 shards492 tasks (snapshot/appdirs/source-path/summary/identity), compare prior gold483/492. Wrapper1615433 failed before evaluation, no duplicate samples. Django isolation1615432 passed867 tests including failing templates; diagnostic subset only, full-suite interference unresolved and no exclusions. Provisional247/50049.4%,19 exact-hash saved patches recovered, no model calls. Requests timeout/external domains and other HTTP gold tasks unresolved. Five trained arrays HELD on OLD provider/scorer;baseline revoked. Preserve original attempts/all500/separate protocols/concurrent work; no duplicates, active-source edits or host networking changes. Continue validated six-model report/publication. Rearm only after watcher exits if work remains running."
for attempt in 1 2 3 4 5;do
 if codex queue --thread "$thread" --message "$message";then date -u '+Alarm delivered %Y-%m-%dT%H:%M:%SZ';exit 0;fi
 sleep 60
done
exit 1
