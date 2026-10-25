"""Validate and summarize completed SWE diagnostics without rerunning them."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT=Path('experiment_reports/training_against_role_probes/swe_gap')
p=argparse.ArgumentParser(__doc__)
p.add_argument('--original-env',required=True);p.add_argument('--fixed-env',required=True)
p.add_argument('--regrade',required=True);p.add_argument('--inspect-regrade',required=True)
a=p.parse_args()
inputs={key:ROOT/'data'/value/'results.json' for key,value in vars(a).items()}
rows=[]
for variant,key in [('original','original_env'),('fixed','fixed_env')]:
    for sample in json.loads(inputs[key].read_text()):
        assert sample['error'] is None
        for tool in ['direct','python_tool','bash_session']:
            raw=sample['environment']['environment_probe'][tool]
            raw=raw['stdout'] if isinstance(raw,dict) else raw
            data=json.JSONDecoder().raw_decode(raw[raw.index('{'):])[0]
            module='asgiref' if sample['id'].startswith('django') else 'mpmath'
            assert data['modules'][module] == (variant=='fixed')
            if variant=='fixed': assert '/envs/testbed/' in data['executable']
            rows.append({'variant':variant,'id':sample['id'],'tool':tool,
                         'python':data['executable'],'required_module':module,
                         'module_available':data['modules'][module]})
assert len(rows)==12
(ROOT/'environment_check.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={}
for key in ['regrade','inspect_regrade']:
    data=json.loads(inputs[key].read_text())
    assert data['errors']==0
    summary[key]={'samples':data['samples'],'errors':data['errors'],'disagreements':data['disagreements'],
                  'results':[{k:r[k] for k in ['id','original_score','official_score','error']} for r in data['results']]}
assert summary['regrade']['samples']==21
(ROOT/'regrade.json').write_text(json.dumps(summary,indent=2)+'\n')
inputs['selected_patches']=ROOT/'data/selected_patches.json'
inputs['attempt_audit']=ROOT/'data/attempt_audit.json'
versions={name:importlib.metadata.version(name) for name in ['inspect-ai','inspect-evals','swebench']}
provenance={'versions':versions,'inputs':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in inputs.items()},
 'jobs':{'audit':json.loads((ROOT/'audit.json').read_text())['slurm_job_id'],**{k:v.rsplit('-',1)[-1] for k,v in vars(a).items()}},
 'public_revisions':{'HarmonyAgent':'175dda4a212193b79e5d860186c5043fd9cdc8e1','SWE-bench_HEAD_inspected':'02e7a74ffd0b707aab73d203fe87bdc7c76afc8e'},
 'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*Path('src/experiments/training_against_role_probes/swe_gap').glob('*.py'),Path('src/role_probe_training/trajectories/swe_testbed.py')]}}
(ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('PASS: 12 environment assertions, 21 official regrades, one fresh Inspect-script control')
