"""One-shot lightweight status for long runs; invoke between timed waits."""
from pathlib import Path
from collections import Counter
import json
from inspect_ai.log import read_eval_log_sample_summaries

for root in ('tau2_retail_full','swe_bench_full'):
 for p in sorted(Path('experiment_reports/agentic',root,'runs').glob('[0-9]*/logs/*.eval')):
  rows=read_eval_log_sample_summaries(p)
  print(root,p.parent.parent.name,'completed',len(rows),'errors',sum(bool(r.error) for r in rows),'scores',Counter(str({k:v.value for k,v in (r.scores or {}).items()}) for r in rows))
for dataset in ('retail','swe'):
 p=Path('experiment_reports/agentic/probe_transfer/predictions')/dataset
 print('predictions',dataset,len(list(p.rglob('*.json.gz'))))
