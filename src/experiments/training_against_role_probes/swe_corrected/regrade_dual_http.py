"""Regrade original v2 patches under localhost repair; never generate new attempts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import inspect_ai._util.zipfile
from inspect_ai import eval
from inspect_ai.solver import solver
from inspect_ai.util import sandbox
from inspect_evals.swe_bench import swe_bench
from role_probe_training.trajectories.swe_dual_http_v3_fixture import sandbox_config
from role_probe_training.trajectories.swe_scoring_dual_http_v3_fixture import fresh_swe_scorer
from swebench.harness.run_evaluation import GIT_APPLY_CMDS

p=argparse.ArgumentParser(__doc__)
p.add_argument('--source',type=Path,default=Path('experiment_reports/training_against_role_probes/base_model/runs/swe_corrected_v2/1516499'))
p.add_argument('--output',type=Path,required=True)
p.add_argument('--sample-id',action='append',required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
patches={}
for path in sorted(a.source.glob('shard*/swe_bench/logs/*.eval')):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith('samples/') or not name.endswith('.json'):
                continue
            sample=json.loads(archive.read(name))
            if sample['id'] not in a.sample_id:
                continue
            attachments=sample.get('attachments',{})
            def resolve(v):
                if isinstance(v,str) and v.startswith('attachment://'):return attachments[v.removeprefix('attachment://')]
                if isinstance(v,dict):return {k:resolve(x) for k,x in v.items()}
                return v
            score=resolve(sample['scores']['swe_bench_scorer'])
            patch=score['metadata']['model_patch']
            assert isinstance(patch,str)
            assert sample['id'] not in patches
            patches[sample['id']]={'patch':patch,'original_score':score['value'],'log':str(path),
                                   'patch_sha256':hashlib.sha256(patch.encode()).hexdigest()}
assert set(patches) == set(a.sample_id)
(a.output/'patches.json').write_text(json.dumps(patches)+'\n')

@solver
def apply_patch():
    async def solve(state,generate):
        patch=patches[str(state.sample_id)]['patch']
        if patch.strip():
            await sandbox().write_file('/tmp/saved.patch',patch)
            for command in GIT_APPLY_CMDS:
                result=await sandbox().exec(['bash','-c',command+' /tmp/saved.patch'],cwd='/testbed')
                if result.success:break
            if not result.success:raise RuntimeError('Saved patch application failed: '+result.stderr)
        return state
    return solve

task=swe_bench(sandbox_type='apptainer_repository_dual_http_v3_fixture',sandbox_config=sandbox_config,scorer=fresh_swe_scorer())
logs=eval(task,model='mockllm/model',solver=apply_patch(),sample_id=sorted(patches),max_samples=2,
          log_dir=str(a.output/'logs'),display='plain',fail_on_error=False)
rows=[]
for log in logs:
    for sample in log.samples or []:
        score=(sample.scores or {}).get('swe_bench_scorer')
        row={'id':sample.id,'original_score':patches[str(sample.id)]['original_score'],
             'score':score.value if score else None,'error':sample.error.message if sample.error else None,
             'patch_sha256':patches[str(sample.id)]['patch_sha256']}
        if score:
            retained=score.metadata['model_patch']
            # Git can normalize diff metadata; retain the original hash and both patches in logs.
            row['reexported_patch_sha256']=hashlib.sha256(retained.encode()).hexdigest()
        rows.append(row)
assert len(rows)==len(patches)
result={'protocol':'saved v2 patches; snapshot and local dual HTTP/TLS v3 fixture; no model calls',
        'samples':len(rows),'original_successes':sum(r['original_score']==1 for r in rows),
        'successes':sum(r['score']==1 for r in rows),'errors':sum(r['error'] is not None for r in rows),
        'results':rows}
(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='results'}))
