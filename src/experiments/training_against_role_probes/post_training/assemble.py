"""Assemble complete fresh benchmark cohorts without success-based filtering."""
import argparse
import hashlib
import json
from pathlib import Path
p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--collection-id',required=True)
p.add_argument('--benchmarks',nargs='+',choices=['appworld_test','tau2_retail','swe_bench'],default=['appworld_test','tau2_retail','swe_bench'])
a=p.parse_args(); results={}
source_file=a.root/'post_training/shard_sources.json'
source_jobs=json.loads(source_file.read_text())['shard_jobs'] if source_file.exists() else {}
for benchmark,shards,expected in [('appworld_test',[0,1],168),('tau2_retail',[2],114),('swe_bench',[3,4,5,6],500)]:
    if benchmark not in a.benchmarks: continue
    rows=[]; selected=set(); manifests=[]
    for shard in shards:
        collection=source_jobs.get(str(shard),a.collection_id)
        directory=a.root/f'runs/post_training/{collection}/shard{shard}/{benchmark}'
        manifest=json.loads((directory/'manifest.json').read_text());manifests.append(manifest)
        selected.update(map(str,manifest['full_selection']))
        rows.extend(json.loads(line) for line in (directory/'trajectories.jsonl').read_text().splitlines())
    startup_failures=[r['sample_id'] for r in rows if r.get('error') and r.get('generation_records',0)==0]
    assert not startup_failures, f'Infrastructure failures before any model attempt: {startup_failures}'
    ids=[str(r['sample_id']) for r in rows]
    assert len(ids)==len(set(ids))==len(selected)==expected
    assert set(ids)==selected
    assert all(not r.get('export_errors') for r in rows), 'Repair export from original logs before scoring'
    for row in rows:row['trajectory_id']=f"{benchmark}/{row['sample_id']}"
    output=a.root/f'data/post_training/{benchmark}';output.mkdir(parents=True,exist_ok=True)
    target=output/'trajectories.jsonl'
    if target.exists():raise FileExistsError(target)
    with target.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    names=sorted({k for r in rows for k in r['scores']}); task_metrics={}
    for name in names:
        values=[r['scores'][name]['value'] for r in rows if name in r['scores'] and not r.get('error')]
        assert all(v in ('C','I',0,1) for v in values)
        successes=sum(v in ('C',1) for v in values)
        task_metrics[name]={'successes':successes,'scored':len(values),'total':expected,'success_rate':successes/expected}
    result={'benchmark':benchmark,'samples':expected,'sample_errors':sum(bool(r.get('error')) for r in rows),
        'export_errors':0,'limited_samples':sum(bool(r.get('limit')) or bool(r.get('limit_events')) for r in rows),
        'metrics':task_metrics,'input_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'source_manifests':manifests}
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    results[benchmark]={k:v for k,v in result.items() if k!='source_manifests'}
small=a.root/'post_training';small.mkdir(exist_ok=True)
(small/'task_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
