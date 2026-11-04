"""Validate complete saved outputs after Slurm OOM; never call a model or scorer."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

folder = Path('experiment_reports/training_against_role_probes/base_model/runs/swe_inspect_generous/1691740/shard1/swe_bench')
manifest = json.loads((folder / 'manifest.json').read_text())
summary = json.loads((folder / 'summary.json').read_text())
rows = [json.loads(line) for line in (folder / 'trajectories.jsonl').open()]
assert len(rows) == len({str(r['sample_id']) for r in rows}) == 125
assert {str(r['sample_id']) for r in rows} == set(map(str, manifest['sample_ids']))
assert not any(r['export_errors'] for r in rows)
assert not any(r['error'] and r['generation_records'] == 0 for r in rows)
assert summary['samples'] == 125 and not summary['missing_sample_ids']
assert all(log['status'] == 'success' for log in summary['inspect_logs'])
counts = Counter()
generation_hash = hashlib.sha256()
with (folder / 'generations.jsonl').open('rb') as stream:
    for line in stream:
        generation_hash.update(line)
        generation = json.loads(line)
        sid = str(generation['sample_id'])
        assert generation['generation_index'] == counts[sid]
        counts[sid] += 1
assert counts == Counter({str(row['sample_id']): row['generation_records'] for row in rows})
assert sum(counts.values()) == summary['generation_records']
trajectory_hash = hashlib.sha256((folder / 'trajectories.jsonl').read_bytes()).hexdigest()
assert trajectory_hash == summary['trajectory_sha256']
successes = sum(not r['error'] and r['scores'].get('swe_bench_scorer', {}).get('value') == 1 for r in rows)
assert successes == summary['metrics']['swe_bench_scorer']['successes']
evidence = {'validated_utc': datetime.now(timezone.utc).isoformat(), 'failed_job':'1691740_1',
            'samples':len(rows), 'successes':successes, 'sample_errors':sum(bool(r['error']) for r in rows),
            'generation_records':sum(counts.values()), 'trajectory_sha256':trajectory_hash,
            'generations_sha256':generation_hash.hexdigest(), 'source_folder':str(folder),
            'new_model_calls':0, 'new_scoring_calls':0, 'original_outputs_unchanged':True}
Path('experiment_reports/training_against_role_probes/swe_generous/oom_saved_validation.json').write_text(json.dumps(evidence, indent=2)+'\n')
print(json.dumps(evidence))
