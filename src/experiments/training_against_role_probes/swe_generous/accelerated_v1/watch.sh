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
elif any(j in {m['ready_job'],m.get('meeting_snapshot_job')} and s=='COMPLETED' for j,s in rows):print('READY')
else:print('WAIT')
PY
 )
 if [[ "$state" == FAILURE* || "$state" == READY ]]; then reason="$state";break;fi
 if (( SECONDS-start >= 1800 ));then reason='30-minute production progress check';break;fi
 sleep 60
done
message="Automatic ACCELERATED SWE continuation: $reason. Read current swe_generous/pipeline.json and latest CURRENT_HANDOFF. Deadline 2026-10-30 15:00 Europe/London, meeting snapshot target14:00UTC. User authorized all models independently, four shards/model; ASTRA disjoint shards0/1 concurrency8 job1726408 and2/3 concurrency16 job1726409. Monitor BOTH collection_jobs and shard_jobs, not only collection_job. No duplicate attempts or restarting completed/running work; consider unstarted-shard migration only after throughput/error validation. Original active code frozen; accelerated_v1 assembly/report explicitly record ASTRA batching variation. Update accelerated_v1/snapshot.py output from saved logs with shared-task comparisons and completion-bias disclosure; publish meeting snapshot if full results pending. Keep500 final denominators and high reasoning/2000 messages; no model-card investigation. Publish validated seven-model task report ASAP then whole-history probes via isolated checkout; preserve concurrent work. After publishing snapshot remove its completed job from watcher readiness to avoid repeated READY alarms. Rearm accelerated_v1/watch.sh only after prior watcher exits if work remains."
for attempt in 1 2 3 4 5;do
 if codex queue --thread "$thread" --message "$message";then date -u '+Production alarm delivered %Y-%m-%dT%H:%M:%SZ';exit 0;fi
 sleep 60
done
exit 1
