"""Validate and combine 27000 AppWorld training attempts, preserving provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--original-runs', type=Path, default=Path('experiment_reports/training_against_role_probes/runs/train2700/1501202'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    timings = []
    audited_errors = []
    for shard, expected in enumerate([90] * 300):
        root = (args.original_runs if shard < 30 else args.runs) / f'shard{shard}' / 'train'
        manifest = json.loads((root / 'manifest.json').read_text())
        summary = json.loads((root / 'summary.json').read_text())
        seed = 8301 + shard
        assert manifest['settings']['seed'] == seed
        assert manifest['settings']['split'] == 'train'
        assert manifest['model_revision'] == '6cee5e81ee83917806bbde320786a8fb61efebee'
        assert manifest['settings']['max_steps'] == 50
        assert manifest['settings']['max_tokens'] == 8192
        assert set(manifest['sample_ids']) == set(Path('.cache/appworld/data/datasets/train.txt').read_text().splitlines())
        assert summary['samples'] == expected
        assert summary['export_errors'] == 0
        batch = [json.loads(line) for line in (root / 'trajectories.jsonl').open()]
        assert len(batch) == expected
        assert {r['sample_id'] for r in batch} == set(manifest['sample_ids'])
        assert len({r['sample_id'] for r in batch}) == expected
        assert summary['sample_errors'] == sum(bool(r['error']) for r in batch)
        for row in batch:
            assert row['messages'] and not row['export_errors']
            if row['error']:
                # Audited model-caused failure: preserve the original attempt,
                # rather than selecting a replacement trajectory for this task.
                assert (shard, row['sample_id']) == (184, '6104387_1')
                assert 'debug_request() missing 1 required positional argument' in str(row['error'])
                assert 'requester.request = debug_request' in json.dumps(row['messages'])
                audited_errors.append({'shard': shard, 'sample_id': row['sample_id'],
                    'seed': seed, 'classification': 'model-caused requester monkey-patch',
                    'policy': 'original saved history retained; missing task scores count as unsuccessful'})
            row['trajectory_id'] = f"appworld_train/{row['sample_id']}/seed{seed}"
            row['generation_seed'] = seed
        rows.extend(batch)
        timings.append(json.loads((root / 'timing.json').read_text()))
    counts = Counter(row['sample_id'] for row in rows)
    assert len(rows) == len({r['trajectory_id'] for r in rows}) == 27000
    assert len(counts) == 90 and set(counts.values()) == {300}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'trajectories.jsonl'
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(target)
    result = {'purpose': 'model_training', 'dataset_split': 'train', 'path': str(target), 'trajectories': 27000, 'unique_tasks': 90,
              'attempts_per_task': dict(sorted(counts.items())),
              'generation_seeds': list(range(8301, 8601)),
              'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'limited_attempts': sum(bool(r['limit']) or bool(r['limit_events']) for r in rows),
              'reasoning_bearing': sum(any(m.get('thinking') for m in r['messages']) for r in rows),
              'original_runs': str(args.original_runs), 'additional_runs': str(args.runs),
              'audited_errors': audited_errors,
              'collection_shards': [{k: t[k] for k in ('started_utc', 'completed_utc', 'wall_seconds', 'tasks')}
                                    for t in timings]}
    scorer_names = sorted({name for row in rows for name in row['scores']})
    result['task_metrics'] = {}
    for name in scorer_names:
        assert all(name in row['scores'] or row['error'] for row in rows)
        values = [row['scores'].get(name, {'value': 0})['value'] for row in rows]
        assert all(value in ('C', 'I', 0, 1) for value in values)
        successes = sum(value in ('C', 1) for value in values)
        result['task_metrics'][name] = {'successes': successes, 'attempts': len(rows),
                                      'success_rate': successes / len(rows)}
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'attempts_per_task'}, indent=2))


if __name__ == '__main__':
    main()
