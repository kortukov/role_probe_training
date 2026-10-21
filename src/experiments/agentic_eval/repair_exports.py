"""Re-export explicitly selected samples into a new run; never mutate original runs."""
import argparse
import json
from pathlib import Path
from role_probe_training.trajectories.inspect_export import export_logs

p = argparse.ArgumentParser(__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--sample-id', action='append', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = json.loads((a.source / 'manifest.json').read_text())
rows = [json.loads(line) for line in (a.source / 'trajectories.jsonl').open()]
selected = {str(r['sample_id']): r['source_log'] for r in rows if str(r['sample_id']) in a.sample_id}
if set(selected) != set(a.sample_id):
    raise ValueError('Requested sample missing')
manifest.update(sample_ids=a.sample_id, repair_source=str(a.source), repair_reason='Preserve malformed tool identity with explicit unknown-name metadata')
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
summary = export_logs(sorted(set(selected.values())), a.output, manifest['benchmark'], manifest, sample_sources=selected)
print({k: summary[k] for k in ['samples', 'export_errors']})
