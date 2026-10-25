"""Audit all saved base attempts and select fixed patches for scorer diagnostics."""
import collections
import hashlib
import json
import re
import os
from pathlib import Path
import zipfile
import inspect_ai._util.zipfile  # register Inspect's Zstandard support

ROOT = Path('experiment_reports/training_against_role_probes')
OUT = ROOT/'swe_gap'

def resolve(value, attachments):
    if isinstance(value, str) and value.startswith('attachment://'):
        return attachments[value.removeprefix('attachment://')]
    if isinstance(value, dict):
        return {k:resolve(v, attachments) for k,v in value.items()}
    if isinstance(value, list):
        return [resolve(v, attachments) for v in value]
    return value

def main():
    rows=[]; candidates=collections.defaultdict(list)
    (OUT/'data').mkdir(exist_ok=True)
    for path in sorted((ROOT/'base_model/runs/swe_native/1504808').glob('shard[0-3]/swe_bench/logs/*.eval')):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(set(archive.namelist())):
                if not (name.startswith('samples/') and name.endswith('.json')): continue
                raw=json.loads(archive.read(name)); sample=resolve(raw,raw.get('attachments',{}))
                score=sample.get('scores',{}).get('swe_bench_scorer',{})
                tools=[e for e in sample['events'] if e['event']=='tool']
                models=[e for e in sample['events'] if e['event']=='model']
                patch=score.get('metadata',{}).get('model_patch')
                errors=[{'function':e['function'],'error':e['error']} for e in tools if e.get('error')]
                missing_patch=[e for e in tools if 'apply_patch' in str(e.get('arguments','')) and ('command not found' in str(e.get('result','')) or 'not found' in str(e.get('error','')))]
                dependency_errors=[{'function':e['function'],'arguments':e.get('arguments'), 'matches':re.findall(r"(?m)^(?:ModuleNotFoundError|ImportError):[^\n]{0,200}", str(e.get('result','')))} for e in tools if re.search(r'(?m)^(?:ModuleNotFoundError|ImportError):',str(e.get('result','')))]
                row={'dependency_errors':dependency_errors,'id':sample['id'],'repo':sample['metadata']['repo'],'score':score.get('value'),
                     'error':sample.get('error',{}).get('message'), 'patch_bytes':None if patch is None else len(patch.encode()),
                     'tool_calls':len(tools),'tool_errors':errors,'missing_apply_patch_calls':len(missing_patch),
                     'tools':dict(collections.Counter(e['function'] for e in tools)),
                     'stops':dict(collections.Counter(c.get('stop_reason','unknown') for e in models for c in e.get('output',{}).get('choices',[]))),
                     'reasoning_efforts':sorted({str(e.get('call',{}).get('request',{}).get('reasoning_effort')) for e in models}),
                     'limits':[e for e in sample['events'] if e['event']=='sample_limit'],
                     'model_usage':sample.get('model_usage'), 'log':str(path)}
                rows.append(row)
                if patch is not None:
                    candidate={'id':sample['id'],'metadata':sample['metadata'],'patch':patch,'original_score':score.get('value')}
                    candidates[(row['repo'],row['score'])].append(candidate)
        print('Read',path,flush=True)
    assert len(rows)==500 and len({r['id'] for r in rows})==500
    selected=[]
    for key,items in sorted(candidates.items()):
        # One success and one failure per repository, chosen by hashed ID, not patch quality.
        selected.append(min(items,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()))
    (OUT/'data/selected_patches.json').write_text(json.dumps(selected)+'\n')
    (OUT/'data/attempt_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary={'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'samples':len(rows),'successes':sum(r['score']==1 for r in rows),
        'evaluation_errors':sum(r['error'] is not None for r in rows),
        'empty_patches':sum(r['patch_bytes']==0 for r in rows),
        'attempts_with_tool_errors':sum(bool(r['tool_errors']) for r in rows),
        'tool_error_calls':sum(len(r['tool_errors']) for r in rows),
        'attempts_missing_apply_patch':sum(bool(r['missing_apply_patch_calls']) for r in rows),
        'missing_apply_patch_calls':sum(r['missing_apply_patch_calls'] for r in rows),
        'tool_functions':dict(sum((collections.Counter(r['tools']) for r in rows),collections.Counter())),
        'model_stops':dict(sum((collections.Counter(r['stops']) for r in rows),collections.Counter())),
        'reasoning_efforts':dict(collections.Counter(x for r in rows for x in r['reasoning_efforts'])),
        'tool_error_types':dict(collections.Counter(str(e['error'].get('type')) for r in rows for e in r['tool_errors'])),
        'sample_limits':dict(collections.Counter(str(e.get('type')) for r in rows for e in r['limits'])),
        'attempts_with_import_errors':sum(bool(r['dependency_errors']) for r in rows),
        'import_error_messages':dict(collections.Counter(m for r in rows for e in r['dependency_errors'] for m in e['matches'])),
        'message_limited_successes':sum(r['score']==1 and bool(r['limits']) for r in rows),
        'selection_rule':'lowest SHA256(instance_id) in each repository x original score stratum; diagnostic, not a representative success estimate',
        'regrade_selection':[{'id':r['id'],'original_score':r['original_score']} for r in selected]}
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
