"""Read saved attempts only; produce a clearly provisional meeting snapshot."""
import json
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import inspect_ai._util.zipfile  # Support Inspect's live ZIP logs.

report = Path('experiment_reports/training_against_role_probes/swe_generous')
pipeline = json.loads((report / 'pipeline.json').read_text())
models = []
for model in pipeline['models']:
    root = Path(model['root'])
    samples = {}
    shards = []
    for shard in range(4):
        job = model.get('shard_jobs', {}).get(str(shard), model['collection_job'])
        folder = root / f'runs/swe_inspect_generous/{job}/shard{shard}/swe_bench'
        records = []
        for log in sorted((folder / 'logs').glob('*.eval')):
            try:
                with zipfile.ZipFile(log) as archive:
                    for name in archive.namelist():
                        if not (name.startswith('samples/') and name.endswith('.json')):
                            continue
                        sample = json.loads(archive.read(name))
                        if not sample.get('completed_at'):
                            continue
                        sid = str(sample['id'])
                        assert sid not in samples, f'Duplicate saved attempt: {model["label"]}/{sid}'
                        value = sample.get('scores', {}).get('swe_bench_scorer', {}).get('value')
                        record = {'success': not bool(sample.get('error')) and value == 1,
                                  'error': bool(sample.get('error')), 'scored': value in (0, 1),
                                  'seconds': sample.get('total_time'), 'source': str(log)}
                        samples[sid] = record
                        records.append(record)
            except (zipfile.BadZipFile, EOFError):
                # A live archive can be between writes. Revisit at the next snapshot.
                continue
        durations = [r['seconds'] for r in records if r['seconds'] is not None]
        shards.append({'shard': shard, 'job': job,
                       'concurrency': model.get('shard_concurrency', {}).get(str(shard), 4),
                       'saved': len(records), 'errors': sum(r['error'] for r in records),
                       'median_seconds': statistics.median(durations) if durations else None})
    assert len(samples) <= 500
    models.append({'model': model['label'], 'samples': samples, 'shards': shards})
common = set.intersection(*(set(m['samples']) for m in models))
base = models[0]['samples']
now = datetime.now(timezone.utc).isoformat()
lines = ['# SWE-bench meeting snapshot', '', f'Updated: {now}. Meeting: 30 October 2026, 15:00 Europe/London.', '',
         '**Provisional saved-log snapshot; final validated 500-task scores and whole-history probe metrics remain pending.**', '',
         'Completed tasks are selected by finish time and can be systematically easier or shorter. Shared-task subsets reduce task mismatch but remain completion-biased; do not interpret these as final benchmark scores.', '',
         'ASTRA shards 0–1 use concurrency 8 and shards 2–3 use 16; other models use 4. Per-attempt budgets, reasoning effort and scorer are unchanged. Batching and contention can affect sampling and timing.', '',
         '| Model | Saved / assigned | Successes among saved | Saved errors | Final score |',
         '|---|---:|---:|---:|---|']
for m in models:
    ss=m['samples']; lines.append(f"| {m['model']} | {len(ss)}/500 | {sum(s['success'] for s in ss.values())} | {sum(s['error'] for s in ss.values())} | Pending validation |")
lines += ['', f'All-seven shared completed task count: {len(common)}.', '',
          '| Trained model vs baseline | Shared completed tasks | Base successes | Trained successes |',
          '|---|---:|---:|---:|']
for m in models[1:]:
    shared = set(base) & set(m['samples'])
    lines.append(f"| {m['model']} | {len(shared)} | {sum(base[s]['success'] for s in shared)} | {sum(m['samples'][s]['success'] for s in shared)} |")
lines += ['', 'Each row above uses its own shared task IDs; rows with different subsets are not comparable to each other. Errors are retained in subset denominators. The final evaluation denominator remains 500 per model.', '',
          'Whole-history role metrics will follow validated task collection. No attempt is regenerated to improve the score.', '']
(report / 'meeting_snapshot.json').write_text(json.dumps({'updated_utc':now,'common_ids':sorted(common),'models':models},indent=2)+'\n')
(report / 'MEETING_SNAPSHOT.md').write_text('\n'.join(lines))
print(json.dumps({'updated_utc':now,'saved':{m['model']:len(m['samples']) for m in models},'all_seven_common':len(common)}))
