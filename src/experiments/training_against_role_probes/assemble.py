"""Validate and combine 300 fixed AppWorld training attempts, preserving provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    timings = []
    for shard, expected in enumerate((90, 90, 90, 30)):
        root = args.runs / f'shard{shard}' / 'train'
        manifest = json.loads((root / 'manifest.json').read_text())
        summary = json.loads((root / 'summary.json').read_text())
        seed = 7301 + shard
        assert manifest['settings']['seed'] == seed
        assert manifest['settings']['split'] == 'train'
        assert summary['samples'] == expected
        assert summary['sample_errors'] == summary['export_errors'] == 0
        batch = [json.loads(line) for line in (root / 'trajectories.jsonl').open()]
        assert len(batch) == expected
        assert {r['sample_id'] for r in batch} == set(manifest['sample_ids'])
        assert len({r['sample_id'] for r in batch}) == expected
        for row in batch:
            assert row['messages'] and not row['error'] and not row['export_errors']
            row['trajectory_id'] = f"appworld_train/{row['sample_id']}/seed{seed}"
            row['generation_seed'] = seed
        rows.extend(batch)
        timings.append(json.loads((root / 'timing.json').read_text()))
    counts = Counter(row['sample_id'] for row in rows)
    assert len(rows) == len({r['trajectory_id'] for r in rows}) == 300
    assert len(counts) == 90 and set(counts.values()) == {3, 4}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'train.jsonl'
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(target)
    result = {'trajectories': 300, 'unique_tasks': 90,
              'attempts_per_task': dict(sorted(counts.items())),
              'generation_seeds': list(range(7301, 7305)),
              'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'limited_attempts': sum(bool(r['limit']) or bool(r['limit_events']) for r in rows),
              'reasoning_bearing': sum(any(m.get('thinking') for m in r['messages']) for r in rows),
              'collection_shards': [{k: t[k] for k in ('started_utc', 'completed_utc', 'wall_seconds', 'tasks')}
                                    for t in timings]}
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'attempts_per_task'}, indent=2))


if __name__ == '__main__':
    main()
