"""Aggregate predictions, refresh the appendix, and build an offline HTML viewer.

README.md and RUNNING.md are authored separately and are never overwritten."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import numpy as np
from role_probe_training.probes.harmony import ROLES

p=argparse.ArgumentParser(__doc__)
p.add_argument('--dataset',action='append',required=True,help='NAME=prediction-directory (repeatable; recursively reads *.json.gz)')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--layers',type=int,nargs='+',default=[12])
p.add_argument('--benchmark-summary',action='append',default=[],help='NAME=summary.json')
p.add_argument('--expected-count',action='append',default=[],help='NAME=count; reject incomplete reports')
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
benchmarks={}
for item in a.benchmark_summary:
 name,path=item.split('=',1); benchmarks[name]=json.loads(Path(path).read_text())
summary={'schema_version':1,'viewer_probability_decimal_places':6,'method':{'model':'openai/gpt-oss-20b','model_revision':'6cee5e81ee83917806bbde320786a8fb61efebee','activation':'zero-based post_attention_layernorm output / pre-MLP input','replay_precision':'MXFP4 checkpoint dequantized to BF16 in Transformers; matches probe fitting','context_policy':'one original eager forward per explicit context window; every token retained','label_policy':'structured roles/channels; content-only score mask; synthetic tool schemas excluded'},'roles':ROLES,'datasets':{},'benchmark_results':benchmarks}
expected={name:int(count) for name,count in (item.split('=',1) for item in a.expected_count)}
index=[]; blocks=[]
for entry in a.dataset:
 name,root=entry.split('=',1); accum={}; seen=set()
 for path in sorted(Path(root).rglob('*.json.gz')):
  raw=gzip.decompress(path.read_bytes()); r=json.loads(raw)
  if r['sample_id'] in seen: raise ValueError(f'duplicate trajectory {name}/{r["sample_id"]}')
  seen.add(r['sample_id'])
  r['layers']={k:v for k,v in r['layers'].items() if int(k) in a.layers}
  if set(r['layers']) != set(map(str,a.layers)): raise ValueError(f'Missing requested layers: {path}')
  item={'dataset':name,'sample_id':r['sample_id'],'tokens':len(r['input_ids']),'windows':len(r.get('context_window_starts',[0])),'layers':{}}
  mask=np.array(r['score_mask'],dtype=bool); y=np.array(r['labels'])[mask]
  for k,v in r['layers'].items():
   if k not in accum: accum[k]={'matrix':np.zeros((5,5),dtype=np.int64),'probability_sums':np.zeros(5),'trajectories':[],'single_window_matrix':np.zeros((5,5),dtype=np.int64)}
   cm=np.array(v['confusion_matrix']); st=accum[k]
   pred=np.asarray(v['predictions']); check=np.zeros((5,5),dtype=np.int64)
   if len(pred)!=len(r['input_ids']) or len(mask)!=len(pred): raise ValueError(f'Token length mismatch: {path}')
   np.add.at(check,(y,pred[mask]),1)
   if not np.array_equal(cm,check) or int(cm.sum())!=v['tokens']: raise ValueError(f'Confusion mismatch: {path}')
   st['matrix']+=cm
   if item['windows']==1: st['single_window_matrix']+=cm
   full_probs=np.asarray(v['probabilities'],dtype=np.float64)
   probs=full_probs[mask]
   for role in range(5): st['probability_sums'][role]+=probs[y==role,role].sum()
   st['trajectories'].append({'sample_id':r['sample_id'],'tokens':v['tokens'],'accuracy':v['accuracy'],'windows':item['windows']})
   item['layers'][k]={'accuracy':v['accuracy'],'tokens':v['tokens']}
   # Remove float32 repr noise in the viewer; retained predictions are unchanged.
   v['probabilities']=full_probs.round(6).tolist()
  index.append(item)
  payload=gzip.compress(json.dumps(r,ensure_ascii=False,separators=(',',':')).encode(),compresslevel=1,mtime=0)
  blocks.append(base64.b64encode(payload).decode())
 if name in expected and len(seen)!=expected[name]: raise ValueError(f'{name}: expected {expected[name]} trajectories, got {len(seen)}')
 layers={}
 for k,st in sorted(accum.items(),key=lambda item:int(item[0])):
  cm=st['matrix']; totals=cm.sum(axis=1); recalls=np.divide(cm.diagonal(),totals,out=np.zeros(5),where=totals>0)
  sw=st['single_window_matrix']
  layers[k]={'trajectories':len(st['trajectories']),'scored_tokens':int(cm.sum()),'accuracy':float(cm.trace()/cm.sum()) if cm.sum() else None,'macro_trajectory_accuracy':float(np.mean([r['accuracy'] for r in st['trajectories'] if r['accuracy'] is not None])),'majority_role_accuracy':float(totals.max()/totals.sum()),'uniform_five_role_accuracy':0.2,'balanced_role_accuracy':float(recalls[totals>0].mean()),'roles':{role:{'tokens':int(totals[i]),'recall':float(recalls[i]) if totals[i] else None,'mean_true_role_probability':float(st['probability_sums'][i]/totals[i]) if totals[i] else None} for i,role in enumerate(ROLES)},'confusion_matrix':cm.tolist(),'windowed_trajectories':sum(r['windows']>1 for r in st['trajectories']),'single_window_accuracy':float(sw.trace()/sw.sum()) if sw.sum() else None,'trajectory_results':st['trajectories']}
 summary['datasets'][name]={'prediction_directory':root,'layers':layers}
(a.output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
# Keep authored README/RUNNING prose intact when refreshing measured artifacts.
from role_probe_training.trajectories.reporting import write_probe_appendix
write_probe_appendix(summary, a.output/'APPENDIX.md')
template=Path(__file__).with_name('viewer.html').read_text()
def js(value): return json.dumps(value,ensure_ascii=True,separators=(',',':')).replace('<','\\u003c')
with (a.output/'viewer.html').open('w') as out:
 page=template.replace('/*SUMMARY*/null',js(summary)).replace('/*INDEX*/[]',js(index))
 before,after=page.split('<!--DATA-->')
 out.write(before)
 for i,block in enumerate(blocks): out.write(f'<script type="application/octet-stream" id="data-{i}">'+block+'</script>\n')
 out.write(after)
print(json.dumps({'trajectories':len(index),'viewer_bytes':(a.output/'viewer.html').stat().st_size}))
