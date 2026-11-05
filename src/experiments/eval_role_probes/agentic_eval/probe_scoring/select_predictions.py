"""Select one prediction per canonical trajectory, preferring explicit repairs.

The output uses symlinks to immutable prediction files and records their hashes.
Canonical messages, tool schemas, scores, errors and source logs must match.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(__doc__)
p.add_argument('--trajectories',type=Path,required=True)
p.add_argument('--predictions',type=Path,nargs='+',required=True,help='Later roots take precedence')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
canonical={str(r['sample_id']):r for r in map(json.loads,a.trajectories.open())}
selected={}
for root in a.predictions:
 for path in sorted(root.rglob('*.json.gz')):
  with gzip.open(path,'rt') as f:r=json.load(f)
  key=str(r['sample_id'])
  if key not in canonical:raise ValueError('Unexpected prediction: '+key)
  expected=canonical[key]
  if all(r['trajectory'].get(k)==expected.get(k) for k in ['messages','tools','scores','error','source_log']):
   selected[key]=path
if set(selected)!=set(canonical):raise ValueError({'missing_matching_predictions':sorted(set(canonical)-set(selected))})
a.output.mkdir(parents=True,exist_ok=True)
if list(a.output.glob('*.json.gz')):raise ValueError('Output must not already contain predictions')
entries=[]
for index,(key,path) in enumerate(sorted(selected.items())):
 target=a.output/f'{index:06d}.json.gz';target.symlink_to(path.resolve())
 entries.append({'sample_id':key,'prediction':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_link':str(target)})
(a.output/'selection.json').write_text(json.dumps({'trajectories':str(a.trajectories),'trajectories_sha256':hashlib.sha256(a.trajectories.read_bytes()).hexdigest(),'predictions':entries},indent=2)+'\n')
print(f'Selected and verified {len(entries)} predictions')
