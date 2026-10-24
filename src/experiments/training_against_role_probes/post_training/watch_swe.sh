#!/bin/bash
# One-shot wake-up: watch the current SWE pipeline, then queue this same session.
set -euo pipefail
thread=${1:?Codex thread UUID}
cd /data/cluster/users/kortukov/role_probe_training
manifest=experiment_reports/training_against_role_probes/post_training/swe_native.json
start=$SECONDS
while true; do
    # Read IDs each time so an explicitly recorded repair is followed too.
    state=$(python3 - "$manifest" <<'PY'
import json,subprocess,sys
m=json.load(open(sys.argv[1]))
ids=[m['report_job'],m['gate_job'],*m['pilot_jobs'].values()]
for jobs in m['full_jobs'].values(): ids.extend(jobs.values())
r=subprocess.run(['sacct','-n','-P','-j',','.join(ids),'--format=JobIDRaw,State'],capture_output=True,text=True)
if r.returncode:
    print('RETRY')
    sys.exit()
rows=[line.split('|')[:2] for line in r.stdout.splitlines() if '|' in line and '.' not in line.split('|')[0]]
failures=[f'{job}:{state}' for job,state in rows if state.split()[0] in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED'}]
if failures: print('FAILURE '+', '.join(failures))
elif any(job==m['report_job'] and state=='COMPLETED' for job,state in rows): print('READY')
else: print('WAIT')
PY
    )
    if [[ "$state" == FAILURE* || "$state" == READY ]]; then
        reason="$state"
        break
    fi
    if (( SECONDS - start >= 1800 )); then
        reason='30-minute progress check; pipeline may still be running'
        break
    fi
    sleep 60
done
message="Automatic SWE evaluation alarm requested by the user: $reason. Read experiment_reports/training_against_role_probes/post_training/swe_native.json and inspect the current jobs. Continue the already authorized work: diagnose failed jobs, or validate/review and push the completed report. Preserve unrelated concurrent KL-training changes. If evaluations are still running, rearm the one-shot watcher with this same thread UUID. Do not submit duplicate evaluations or treat partial cohorts as complete."
# Retry transient daemon connection failures without starting another agent.
for attempt in 1 2 3 4 5; do
    if codex queue --thread "$thread" --message "$message"; then
        date -u '+Alarm delivered at %Y-%m-%dT%H:%M:%SZ'
        exit 0
    fi
    sleep 60
done
exit 1
