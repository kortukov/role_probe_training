"""Survey environment validity with gold patches; no benchmark model calls."""
import argparse
import hashlib
import json
from pathlib import Path
from inspect_ai import eval
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench.solvers import swe_bench_oracle_solver
from role_probe_training.trajectories.swe_localhost import sandbox_config
from role_probe_training.trajectories.swe_scoring_localhost import fresh_swe_scorer

p=argparse.ArgumentParser(__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--shard-index',type=int,required=True)
p.add_argument('--shard-count',type=int,default=4)
a=p.parse_args();assert 0<=a.shard_index<a.shard_count
a.output.mkdir(parents=True,exist_ok=False)
task=swe_bench(sandbox_type='apptainer_repository_localhost',sandbox_config=sandbox_config,scorer=fresh_swe_scorer())
ids=sorted(str(s.id) for s in task.dataset if s.metadata['repo']!='psf/requests')[a.shard_index::a.shard_count]
manifest={'sample_ids':ids,'shard_index':a.shard_index,'shard_count':a.shard_count,
          'excluded_repo':'psf/requests (separate fixture controls)',
          'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/role_probe_training/trajectories/swe_localhost.py'),Path('src/role_probe_training/trajectories/swe_scoring_localhost.py')]}}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
logs=eval(task,model='mockllm/model',solver=swe_bench_oracle_solver(),sample_id=ids,
          max_samples=2,log_dir=str(a.output/'logs'),display='plain',fail_on_error=False)
rows=[]
for log in logs:
    for sample in log.samples or []:
        score=(sample.scores or {}).get('swe_bench_scorer')
        rows.append({'id':sample.id,'score':score.value if score else None,
                     'error':sample.error.message if sample.error else None,
                     'tests_status':score.metadata.get('official_report',{}).get('tests_status') if score else None})
assert len(rows)==len(ids)==len({r['id'] for r in rows})
result={'samples':len(rows),'gold_passes':sum(r['score']==1 for r in rows),
        'errors':sum(r['error'] is not None for r in rows),'results':rows}
(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='results'}))
