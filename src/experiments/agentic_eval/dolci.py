"""Uniform seeded subset of pinned Dolci parquet rows, in trajectory schema v1."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import pyarrow.parquet as pq

p = argparse.ArgumentParser(__doc__)
p.add_argument('--source', type=Path, default=Path('.cache/dolci-tool-use/data'))
p.add_argument('--output', type=Path, default=Path('experiment_reports/agentic/dolci_1000'))
p.add_argument('--seed', type=int, default=123)
p.add_argument('--samples', type=int, default=1000)
a = p.parse_args()
files = sorted(a.source.glob('*.parquet'))
counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
indices = sorted(random.Random(a.seed).sample(range(sum(counts)), a.samples))
selected = set(indices)
provenance = {'dataset': 'allenai/Dolci-Instruct-SFT-Tool-Use', 'revision': 'dc042846f0f2de0f15eedae3d6ced04223ed47eb', 'split': 'train', 'seed': a.seed, 'population': sum(counts), 'indices': indices, 'selection': 'uniform without replacement: Python Random(seed).sample over concatenated sorted parquet files', 'source_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
(a.output/'data').mkdir(parents=True, exist_ok=True)
sources, roles = Counter(), Counter()
offset = 0
with (a.output/'data/trajectories.jsonl').open('w') as out:
 for f in files:
  for batch in pq.ParquetFile(f).iter_batches(batch_size=1024):
   for i in sorted(selected.intersection(range(offset, offset+len(batch)))):
    row = batch.slice(i-offset, 1).to_pylist()[0]
    messages, tools = [], []
    for original in row['messages']:
     m = {k:v for k,v in original.items() if v is not None}
     m['content'] = original['content'] or ''
     if m['role'] == 'environment':
      m.update(role='tool', name='environment', source_role='environment')
     if m['role'] not in ('system','user','assistant','tool'):
      raise ValueError(f"Unknown Dolci role: {m['role']}")
     if m.get('functions'):
      tools.extend(json.loads(m['functions']))
     roles[m['role']] += 1
     messages.append(m)
    record = {'schema_version':1, 'benchmark':'dolci_tool_use', 'sample_id':row['id'], 'epoch':1, 'model':None, 'messages':messages, 'tools':tools, 'scores':{}, 'error':None, 'export_errors':[], 'provenance':{'dataset':provenance['dataset'], 'revision':provenance['revision'], 'source_index':i, 'dataset_source':row['dataset_source']}}
    sources[row['dataset_source']] += 1
    out.write(json.dumps(record, ensure_ascii=False)+'\n')
   offset += len(batch)
provenance.update(samples=sum(sources.values()), sources=dict(sources), message_roles=dict(roles), trajectories_sha256=hashlib.sha256((a.output/'data/trajectories.jsonl').read_bytes()).hexdigest(), normalization={'environment':'tool; name=environment because aggregate outputs do not identify one tool', 'function_calls':'retained verbatim; not parsed as executable Python', 'channels':'not supplied by Dolci; no inferred analysis labels'})
(a.output/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({k:provenance[k] for k in ('samples','sources','message_roles','trajectories_sha256')}))
