"""Aggregate predictions and build reports plus an offline, extensible HTML viewer."""
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
  if not r['layers']: continue
  item={'dataset':name,'sample_id':r['sample_id'],'tokens':len(r['input_ids']),'windows':len(r.get('context_window_starts',[0])),'layers':{}}
  mask=np.array(r['score_mask'],dtype=bool); y=np.array(r['labels'])[mask]
  for k,v in r['layers'].items():
   if k not in accum: accum[k]={'matrix':np.zeros((5,5),dtype=np.int64),'probability_sums':np.zeros(5),'trajectories':[],'single_window_matrix':np.zeros((5,5),dtype=np.int64)}
   cm=np.array(v['confusion_matrix']); st=accum[k]; st['matrix']+=cm
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
lines=['# '+('Role-probe layer comparison on agentic trajectories' if len(a.layers)>1 else 'Layer-12 role-probe transfer to agentic trajectories'),'','## Motivation & RQ','','How well do fixed role probes transfer to realistic on-policy benchmark traces and off-policy Dolci tool-use data?','','## Evaluation setup','','GPT-OSS-20B, revision `6cee5e81ee83917806bbde320786a8fb61efebee`; zero-based post-attention-layernorm / pre-MLP probes. Replay uses the checkpoint dequantized to BF16 in Transformers, matching probe fitting; benchmark generation uses vLLM MXFP4. Ground truth comes from structured roles and analysis channels. Accuracy is argmax correctness, not mean true-role probability. Structural tokens and synthetic tool-schema context are excluded from scores. All content tokens are retained; long traces use explicitly recorded 16,384-token context windows with position/context resets. Shorter trajectories preserve causal context throughout. This is standardized retrospective Harmony replay, not a byte-identical reconstruction of serving prompts. Dolci has no analysis-channel labels. Probe fitting warnings in the baseline remain applicable.','','## Factual observations','','| Dataset | Layer | Trajectories | Tokens | Token accuracy | Mean trajectory accuracy | Windowed traces |','|---|---:|---:|---:|---:|---:|---:|']
appendix=['# Confusion matrices','','Rows are ground truth; columns are predictions. Counts exclude headers, special tokens and synthetic tool-schema context. Missing roles have no recall estimate.']
for name,d in summary['datasets'].items():
 for k,v in d['layers'].items():
  lines.append(f"| {name} | {k} | {v['trajectories']} | {v['scored_tokens']:,} | {v['accuracy']:.2%} | {v['macro_trajectory_accuracy']:.2%} | {v['windowed_trajectories']} |")
  appendix+=['',f'## {name} layer {k}','','| True / predicted | '+' | '.join(ROLES)+' |','|---|'+'---:|'*5]
  for role,row in zip(ROLES,v['confusion_matrix']): appendix.append('| '+role+' | '+' | '.join(map(str,row))+' |')
  appendix+=['','Row-normalized percentages:','','| True / predicted | '+' | '.join(ROLES)+' |','|---|'+'---:|'*5]
  for role,row in zip(ROLES,v['confusion_matrix']): appendix.append('| '+role+' | '+' | '.join(f'{x/sum(row):.1%}' if sum(row) else 'n/a' for x in row)+' |')
for name,d in summary['datasets'].items():
 first=next(iter(d['layers'].values()))
 lines+=['',f"{name}: always predicting the most frequent ground-truth role yields {first['majority_role_accuracy']:.2%} token accuracy; uniform random prediction over the fixed five classes yields 20%. Role-balanced accuracy and mean trajectory accuracy address different aspects of the substantial token imbalance."]
if benchmarks:
 lines+=['','Benchmark task performance (distinct from probe accuracy):','','| Dataset | Scorer | Successes / scored | Observed | Scored success rate | Errors |','|---|---|---:|---:|---:|---:|']
 for name, result in benchmarks.items():
  for scorer,v in result.get('metrics',{}).items():
   lines.append(f"| {name} | {scorer} | {v['successes']} / {v['scored']} | {v['observed_samples']} | {v['success_rate_scored']:.2%} | {result.get('sample_errors',0)} |")
 lines+=['','Scored-only success rates exclude errored samples; full summaries also give errors-as-failures rates and confidence intervals. Interpret serving failures, context/time/token limits, and harness differences before comparing with published model scores. Dolci has no task-success metric because GPT-OSS did not generate these static trajectories.']
lines+=['','See [confusion matrices](APPENDIX.md#confusion-matrices) and [full metrics](metrics.json) for per-role recall, mean true-role probabilities, individual trajectories, and results restricted to traces fitting one context window.','','## Message','','These measurements describe transfer under the documented replay and context-window policy. Differences between Dolci and generated benchmarks combine policy, task, role-frequency, formatting and trajectory-length effects; they do not isolate an on-policy causal effect. Dolci is an evaluation sample here, not probe training data. Benchmark task success is distinct from role-probe accuracy.','','## Reproduction and standalone viewer','','The local `viewer.html` embeds all prediction data and works without network access in a modern browser. It supports dataset, layer and trajectory selection, role-colored tokens, ground-truth/predicted coloring, mismatch filtering and token probability inspection. The file contains data and is deliberately excluded from Git. See [RUNNING.md](RUNNING.md) for commands and data paths.']
(a.output/'README.md').write_text('\n'.join(lines)+'\n'); (a.output/'APPENDIX.md').write_text('\n'.join(appendix)+'\n')
commands=' '.join('--dataset '+repr(x) for x in a.dataset)
(a.output/'RUNNING.md').write_text('# Reproduction\n\nPredictions use schema v1, one gzip JSON per trajectory. Each contains original messages, normalized replay messages, token IDs/offsets, labels, score mask, context boundaries and per-layer probabilities and argmax predictions. Input/probe hashes and job metadata are in each prediction directory’s manifest.\n\nGenerate the report and standalone viewer from the repository root:\n\n```bash\n.tools/bin/uv run --frozen --no-sync python src/experiments/agentic_probes/report.py '+commands+' --layers '+' '.join(map(str,a.layers))+' --output '+str(a.output)+'\n```\n\nAdd another `--dataset NAME=PATH` to include a dataset. See `src/experiments/agentic_probes/run.py --help` for inference and `src/experiments/agentic_eval/dolci.py` for seeded sampling. Generated predictions, trajectories and HTML remain on cluster storage; code, manifests and aggregate metrics are versioned.\n')
template=Path(__file__).with_name('viewer.html').read_text()
def js(value): return json.dumps(value,ensure_ascii=True,separators=(',',':')).replace('<','\\u003c')
with (a.output/'viewer.html').open('w') as out:
 page=template.replace('/*SUMMARY*/null',js(summary)).replace('/*INDEX*/[]',js(index))
 before,after=page.split('<!--DATA-->')
 out.write(before)
 for i,block in enumerate(blocks): out.write(f'<script type="application/octet-stream" id="data-{i}">'+block+'</script>\n')
 out.write(after)
print(json.dumps({'trajectories':len(index),'viewer_bytes':(a.output/'viewer.html').stat().st_size}))
