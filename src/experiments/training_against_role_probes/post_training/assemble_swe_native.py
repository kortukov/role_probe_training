"""Assemble all 500 native-context SWE attempts; never select by outcome."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--collection-id',required=True)
a=p.parse_args()
rows=[]; manifests=[]; sources=[]
source_map=a.root/'post_training/swe_native_shard_sources.json'
source_jobs=json.loads(source_map.read_text())['shard_jobs'] if source_map.exists() else {}
for shard in range(4):
    collection=source_jobs.get(str(shard),a.collection_id)
    folder=a.root/f'runs/swe_native/{collection}/shard{shard}/swe_bench'
    manifest=json.loads((folder/'manifest.json').read_text())
    manifests.append(manifest)
    source=folder/'trajectories.jsonl'
    shard_rows=[json.loads(line) for line in source.open()]
    assert {str(r['sample_id']) for r in shard_rows}==set(map(str,manifest['sample_ids']))
    assert not any(r['export_errors'] for r in shard_rows)
    startup_failures = [r['sample_id'] for r in shard_rows if r.get('error') and r.get('generation_records', 0) == 0]
    assert not startup_failures, f'Infrastructure failures before any model attempt must be repaired before assembly: {startup_failures}'
    rows.extend(shard_rows)
    sources.append({'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
expected=set(map(str,manifests[0]['full_selection']))
assert len(rows)==len({str(r['sample_id']) for r in rows})==len(expected)==500
assert {str(r['sample_id']) for r in rows}==expected
settings=['temperature','reasoning_effort','max_tokens','message_limit','token_limit','time_limit','concurrency']
assert all({k:m['settings'][k] for k in settings}=={k:manifests[0]['settings'][k] for k in settings} for m in manifests)
output=a.root/'data/swe_native/evaluation'
output.mkdir(parents=True,exist_ok=False)
for r in rows:r['trajectory_id']=f"swe_bench/{r['sample_id']}"
target=output/'trajectories.jsonl'
with target.open('w') as f:
    for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
values=[r['scores']['swe_bench_scorer']['value'] for r in rows if not r['error'] and 'swe_bench_scorer' in r['scores']]
assert all(v in (0,1) for v in values)
result={'benchmark':'swe_bench','samples':500,'sample_errors':sum(bool(r['error']) for r in rows),
        'export_errors':0,'limited_samples':sum(bool(r['limit']) or bool(r['limit_events']) for r in rows),
        'samples_with_model_context_stops':sum(r['model_context_stops']>0 for r in rows),
        'metrics':{'swe_bench_scorer':{'successes':sum(values),'scored':len(values),'total':500,'success_rate':sum(values)/500}},
        'input_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'sources':sources,'settings':{k:manifests[0]['settings'][k] for k in settings}}
(a.root/'post_training/swe_native_task_results.json').write_text(json.dumps(result,indent=2)+'\n')
(output/'manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')
print(json.dumps(result))
