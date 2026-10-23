"""Verify all task/attempt pairs and explicitly merge the 1/2/3-call corpus."""
import argparse
from collections import Counter
import json
from pathlib import Path
from role_probe_training.trajectories.corpora import corpus_summary, sha256, write_jsonl


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('experiment_reports/training_against_role_probes/depthbench'))
    a = p.parse_args()
    manifest = json.loads((a.root/'data_manifest.json').read_text())
    settings = json.loads((a.collection/'manifest.json').read_text())
    assert settings['attempts'] == 20 and not settings['smoke']
    assert settings['dataset_revision'] == manifest['revision']
    grouped = {}
    provenance = {'collection': str(a.collection), 'settings': settings, 'groups': {}}
    for n in range(7):
        tasks = [json.loads(line) for line in (a.root/f'data/tasks/calls_{n}.jsonl').open()]
        expected = {(t['task_id'], i) for t in tasks for i in range(20)}
        rows = [json.loads(p.read_text()) for p in sorted((a.collection/f'calls_{n}').glob('*.json'))
                if not p.name.endswith('.generations.json')]
        actual = {(r['task_id'], r['attempt']) for r in rows}
        assert actual == expected and len(rows) == len(expected), (n, len(rows), len(expected))
        assert all(r['required_calls'] == n and r['dataset_split'] == 'train' for r in rows)
        assert all(not r['error'] and not r['export_errors'] for r in rows)
        assert len({r['trajectory_id'] for r in rows}) == len(rows)
        output = a.root/f'data/calls_{n}/trajectories.jsonl'
        write_jsonl(output, rows)
        grouped[n] = rows
        provenance['groups'][n] = corpus_summary(rows) | {'sha256': sha256(output),
            'path': str(output), 'limits': dict(Counter(str(r['limit']) for r in rows))}
    combined = [r for n in (1, 2, 3) for r in grouped[n]]
    assert len(combined) == manifest['expected_training_rollouts']
    output = a.root/'data/train_1_2_3/trajectories.jsonl'
    write_jsonl(output, combined)
    provenance['training'] = corpus_summary(combined) | {'groups': [1, 2, 3],
        'sha256': sha256(output), 'path': str(output),
        'limits': dict(Counter(str(r['limit']) for r in combined))}
    (a.root/'collection_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance['training'], indent=2))


if __name__ == '__main__':
    main()
