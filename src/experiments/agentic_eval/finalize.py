"""Combine disjoint completed shards without reparsing large Inspect event logs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from role_probe_training.trajectories.inspect_export import wilson

p=argparse.ArgumentParser(__doc__)
p.add_argument('--run',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--selection-manifest',type=Path,required=True)
p.add_argument('--benchmark',choices=['tau2_retail','swe_bench'],required=True)
a=p.parse_args(); (a.output/'data').mkdir(parents=True,exist_ok=True); (a.output/'metrics').mkdir(exist_ok=True)
expected=set(map(str,json.loads(a.selection_manifest.read_text())['full_selection']))
seen=set(); counts=Counter(); metric_counts={}; sources=[]; cohorts=Counter(); errors=Counter()
output=a.output/'data/trajectories.jsonl'; temporary=output.with_suffix('.tmp')
with temporary.open('w') as target:
 for root in a.run:
  source=root/'trajectories.jsonl'; summary=json.loads((root/'summary.json').read_text())
  sources.append({'run':str(root),'trajectories_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'manifest':json.loads((root/'manifest.json').read_text())})
  for key,value in summary.items():
   if isinstance(value,int) and not isinstance(value,bool):counts[key]+=value
  for name,metric in summary['metrics'].items():
   c=metric_counts.setdefault(name,Counter())
   for key in ['successes','scored','observed_samples']:c[key]+=metric[key]
  with source.open() as f:
   for line in f:
    r=json.loads(line); key=str(r['sample_id'])
    if key in seen: raise ValueError('Duplicate sample across shards: '+key)
    seen.add(key);target.write(line if line.endswith('\n') else line+'\n')
    if any(m['role']=='tool' for m in r['messages']):cohorts['tool_bearing']+=1
    if any(m['role']=='assistant' for m in r['messages']):cohorts['assistant_bearing']+=1
    if r.get('error'):
     error=r['error']; message=error.get('message','') if isinstance(error,dict) else str(error)
     category='image_preparation' if 'Image preparation failed' in message else 'model_api_retry' if 'RetryError' in message else 'other'
     errors[category]+=1
if seen!=expected: raise ValueError({'missing':sorted(expected-seen),'unexpected':sorted(seen-expected)})
temporary.replace(output)
metrics={}
for name,c in metric_counts.items():
 metrics[name]=dict(c)|{'success_rate_scored':c['successes']/c['scored'] if c['scored'] else None,'wilson_95_scored':wilson(c['successes'],c['scored']),'success_rate_observed_errors_as_failures':c['successes']/c['observed_samples']}
result={'benchmark':a.benchmark,**dict(counts),'metrics':metrics,'coverage':dict(cohorts),'error_categories':dict(errors),'expected_samples':len(expected),'observed_unique_samples':len(seen),'missing_sample_ids':[],'trajectories_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_runs':[str(p) for p in a.run]}
(a.output/'metrics/summary.json').write_text(json.dumps(result,indent=2)+'\n')
(a.output/'metrics/provenance.json').write_text(json.dumps({'selection_manifest':str(a.selection_manifest),'sources':sources},indent=2)+'\n')
title='Full τ²-bench retail' if a.benchmark=='tau2_retail' else 'Full SWE-bench Verified'
lines=[f'# {title}: GPT-OSS-20B','','## Motivation & RQ','','Measure model task performance and collect realistic on-policy trajectories for role-probe transfer evaluation.','','## Evaluation setup','','GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`, high reasoning, temperature 0, seed 123, vLLM 0.11.0. One attempt per task in this selected run; all dataset tasks included. Per-call output limit 16,384; per-sample cumulative token limit 500,000 and wall-time limit 1,800 seconds. Source manifests record exact versions, task IDs, code hashes, Slurm jobs and model configuration.']
if a.benchmark=='tau2_retail':lines+=['','114 τ²-bench retail tasks; Azure GPT-4.1 customer simulator (`gpt-4.1-2025-04-14`), 16 concurrent samples. The original upstream 100-turn cap is retained.']
else:lines+=['','All 500 SWE-bench Verified tasks, pinned dataset revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Stock Inspect solver/scorer, 30-message cap; four shards with eight concurrent samples each. Validated Apptainer isolation and gold-patch scoring. Official images are pinned by digest in [image_manifest.json](image_manifest.json) and retrieved from Google’s public Docker Hub cache. The earlier rate-limited attempt is preserved but excluded from these results.']
lines+=['','## Factual observations','','| Scorer | Successes / scored | All tasks | Scored rate | Errors as failures | 95% Wilson CI (scored) |','|---|---:|---:|---:|---:|---|']
for name,m in metrics.items():
 ci=m['wilson_95_scored'];lines.append(f"| {name} | {m['successes']} / {m['scored']} | {m['observed_samples']} | {m['success_rate_scored']:.2%} | {m['success_rate_observed_errors_as_failures']:.2%} | {ci[0]:.2%}–{ci[1]:.2%} |")
lines+=['',f"Sample errors: {counts['sample_errors']}; limited samples: {counts['limited_samples']}; samples with generation-token caps: {counts['samples_with_generation_caps']}; failed agent model events: {counts['agent_model_errors']}. Tool-bearing trajectories: {cohorts['tool_bearing']}/{len(seen)}. See [full metrics](metrics/summary.json), [provenance](metrics/provenance.json), and the [probe report](../probe_transfer/README.md) with its [confusion appendix](../probe_transfer/APPENDIX.md).",'','## Message']
if a.benchmark=='tau2_retail':lines+=['','This measures τ² retail, whereas OpenAI’s published high-reasoning GPT-OSS-20B retail score is 54.8% on the original τ-bench. Treat that as context, not a matched reproduction. Time/token limits and malformed model headers can affect this run. [OpenAI model card](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations).']
else:lines+=['','This Inspect configuration does not reproduce OpenAI’s published 60.7% SWE-bench result. The published setup uses a different terminal harness and evaluation subset; this run uses stock Inspect tools, a 30-message cap, temperature 0 and a 32K serving context. Malformed generated Harmony headers cause additional API failures. The official repository recommends temperature 1/top-p 1; no sampling/harness change was made midway through this run. Do not interpret the observed score as the model’s best achievable SWE-bench performance. [OpenAI model card](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations), [model sampling recommendation](https://github.com/openai/gpt-oss).']
lines+=['','The trace set includes both successful and unsuccessful/error-ending attempts. Probe metrics describe the retained conversational tokens, not task success.','','See [RUNNING.md](RUNNING.md) for reproduction and standalone viewer generation.']
(a.output/'README.md').write_text('\n'.join(lines)+'\n')
(a.output/'RUNNING.md').write_text('# Reproduction\n\nGeneration runs and immutable inputs are recorded in [metrics/provenance.json](metrics/provenance.json). Their raw `.eval` logs and `generations.jsonl` retain individual model calls. The consolidated chat data is `data/trajectories.jsonl`; its hash is in [metrics/summary.json](metrics/summary.json). Data is kept on cluster storage and excluded from Git.\n\nCombine completed disjoint shards:\n\n```bash\n.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/agentic_eval/finalize.py --benchmark '+a.benchmark+' --run '+' '.join(map(str,a.run))+' --selection-manifest '+str(a.selection_manifest)+' --output '+str(a.output)+'\n```\n\n## Probe predictions and standalone viewer\n\nSee [probe reproduction](../probe_transfer/RUNNING.md) and [layer-comparison reproduction](../probe_layer_comparison/RUNNING.md). The shared `agentic_probes/report.py` script accepts repeated `--dataset NAME=PREDICTION_DIRECTORY`, optional benchmark summary paths, and layer choices. It embeds all data in a standalone `viewer.html`, which stays local and can be downloaded and opened without a server.\n')
print(json.dumps(result))
