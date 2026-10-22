"""Combine completed shards with explicitly justified infrastructure/export repairs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from role_probe_training.trajectories.inspect_export import wilson

p=argparse.ArgumentParser(__doc__)
p.add_argument('--run',type=Path,nargs='+',required=True)
p.add_argument('--replacement-run',type=Path,nargs='*',default=[])
p.add_argument('--output',type=Path,required=True)
p.add_argument('--selection-manifest',type=Path,required=True)
p.add_argument('--benchmark',choices=['tau2_retail','swe_bench'],required=True)
a=p.parse_args(); (a.output/'data').mkdir(parents=True,exist_ok=True); (a.output/'metrics').mkdir(exist_ok=True)
expected=set(map(str,json.loads(a.selection_manifest.read_text())['full_selection']))
rows={}; sources=[]; repairs=[]
for replacement,roots in [(False,a.run),(True,a.replacement_run)]:
 for root in roots:
  source=root/'trajectories.jsonl'
  sources.append({'run':str(root),'replacement':replacement,'trajectories_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'manifest':json.loads((root/'manifest.json').read_text())})
  generation_counts=Counter()
  with (root/'generations.jsonl').open() as stream:
   for line in stream:generation_counts[str(json.loads(line)['sample_id'])]+=1
  for line in source.open():
   r=json.loads(line); key=str(r['sample_id']); r['generation_records']=generation_counts[key]
   if replacement:
    if key not in rows:raise ValueError('Replacement lacks original: '+key)
    old=rows[key]
    infrastructure='Image preparation failed' in str(old.get('error')) and old['generation_records']==0
    normalization=bool(old.get('export_errors')) and old['source_log']==r['source_log']
    if not (infrastructure or normalization):raise ValueError('Replacement is not infrastructure/export repair: '+key)
    repairs.append({'sample_id':key,'reason':'image_preparation_no_model_execution' if infrastructure else 'tool_name_normalization','original_source_log':old['source_log'],'selected_source_log':r['source_log']})
   elif key in rows:raise ValueError('Duplicate sample across shards: '+key)
   rows[key]=r
seen=set(rows)
if seen!=expected:raise ValueError({'missing':sorted(expected-seen),'unexpected':sorted(seen-expected)})
output=a.output/'data/trajectories.jsonl'; temporary=output.with_suffix('.tmp')
with temporary.open('w') as target:
 for r in rows.values():target.write(json.dumps(r,ensure_ascii=False)+'\n')
temporary.replace(output)
counts=Counter(samples=len(rows));cohorts=Counter();errors=Counter();metric_counts={}
scorers=sorted({name for r in rows.values() for name in r['scores']})
for name in scorers:
 values=[r['scores'][name]['value'] for r in rows.values() if name in r['scores'] and not r['error']]
 if not all(v in ('C','I',0,1) for v in values):raise ValueError('Nonbinary scorer')
 metric_counts[name]=dict(successes=sum(v in ('C',1) for v in values),scored=len(values),observed_samples=len(rows))
for r in rows.values():
 for key,value in {'sample_errors':bool(r['error']),'limited_samples':bool(r['limit']) or bool(r['limit_events']),'export_errors':bool(r['export_errors']),'agent_model_errors':len(r['agent_model_errors']),'all_model_errors':len(r['all_model_errors']),'generation_token_caps':r['generation_token_caps'],'samples_with_generation_caps':r['generation_token_caps']>0,'model_retries':r['model_retries'],'samples_with_limit_events':bool(r['limit_events']),'simulator_stop_missing':r['simulator_stop_seen'] is False,'generation_records':r['generation_records']}.items():counts[key]+=value
 for role in ['tool','assistant']:
  if any(m['role']==role for m in r['messages']):cohorts[role+'_bearing']+=1
 if r.get('error'):
  error=r['error'];message=error.get('message','') if isinstance(error,dict) else str(error)
  category='image_preparation' if 'Image preparation failed' in message else 'model_api_retry' if 'RetryError' in message else 'other'
  errors[category]+=1
metrics={}
for name,c in metric_counts.items():
 metrics[name]=dict(c)|{'success_rate_scored':c['successes']/c['scored'] if c['scored'] else None,'wilson_95_scored':wilson(c['successes'],c['scored']),'success_rate_observed_errors_as_failures':c['successes']/c['observed_samples']}
result={'benchmark':a.benchmark,**dict(counts),'metrics':metrics,'coverage':dict(cohorts),'error_categories':dict(errors),'expected_samples':len(expected),'observed_unique_samples':len(seen),'missing_sample_ids':[],'trajectories_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_runs':[str(p) for p in a.run],'repairs':repairs}
(a.output/'metrics/summary.json').write_text(json.dumps(result,indent=2)+'\n')
(a.output/'metrics/provenance.json').write_text(json.dumps({'selection_manifest':str(a.selection_manifest),'sources':sources,'repairs':repairs},indent=2)+'\n')
# README/RUNNING are authored scientific reports, not generated execution logs.
print(json.dumps(result))
