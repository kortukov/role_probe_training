"""Validate completed KL cohorts, save results and detailed role confusions."""
import argparse
import hashlib
import json
from pathlib import Path
from report import BENCHMARKS, ROOT, main as refresh_table
from validate import validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root
    validate(root)
    tasks = json.loads((root/'post_training/task_results.json').read_text())
    export = json.loads((root/'data/post_training/model/export.json').read_text())
    probe = root/'post_training/probes/layer_12.npz'
    metadata = json.loads(probe.with_suffix('.json').read_text())
    assert metadata['fresh_fit'] and metadata['layer'] == 12
    assert metadata['activation'] == 'post_attention_layernorm_output_pre_mlp'
    assert metadata['model_revision'] == export['adapter_sha256']
    results = {}; coverage = []; appendix = ['# Role confusions\n',
        'Rows are true roles; columns are predicted roles. Values are row percentages.\n']
    table = ['| Benchmark | Task success | Overall role | Histories scored |', '|---|---:|---:|---:|']
    for key, name, expected, metric in BENCHMARKS:
        task = tasks[key]
        score = json.loads((root/f'predictions/post_training/{key}/summary.json').read_text())
        assert task['samples'] == score['trajectories'] == expected
        assert task['input_sha256'] == score['input_sha256'] == digest(root/f'data/post_training/{key}/trajectories.jsonl')
        assert score['probe_sha256'] == digest(probe)
        assert score['model_revision'] == metadata['model_revision']
        assert score['context_policy'] == 'Whole conversation in one forward pass; no chunking or truncation'
        assert not score.get('segmented_trajectories')
        excluded = score['excluded_trajectories']; scored = score['scored_trajectories']
        assert scored + len(excluded) == expected
        assert all(x['tokens'] > score['context_limit'] for x in excluded)
        assert score['roles'] == ['system','user','cot','assistant','tool']
        counts = score['confusion_counts']
        assert len(counts) == 5 and all(len(row) == 5 and all(x >= 0 for x in row) for row in counts)
        total = sum(map(sum, counts)); assert total > 0
        assert abs(sum(counts[i][i] for i in range(5))/total-score['accuracy']) < 1e-10
        for i, row in enumerate(counts):
            if sum(row): assert abs(row[i]/sum(row)-score['per_role_recall'][i]) < 1e-10
        m = task['metrics'][metric]
        assert m['total'] == expected and abs(m['successes']/expected-m['success_rate']) < 1e-10
        results[key] = dict(task=task, probe=score, task_metric=metric)
        table.append(f"| {name} | {100*m['success_rate']:.1f} | {100*score['accuracy']:.1f} | {scored}/{expected} |")
        coverage.append(dict(benchmark=key,scored=scored,total=expected,excluded=excluded))
        appendix += [f'## {name}\n', '| True / predicted | System | User | Reasoning | Assistant | Tool |', '|---|---:|---:|---:|---:|---:|']
        for label, row in zip(['System','User','Reasoning','Assistant','Tool'],counts):
            appendix.append('| '+label+' | '+' | '.join(f'{100*x/sum(row):.2f}%' if sum(row) else '—' for x in row)+' |')
        appendix += [f'\nCoverage: {scored}/{expected}; {len(excluded)} histories exceed native context. Task success includes all attempts.\n']
    (root/'post_training/results.json').write_text(json.dumps(results,indent=2)+'\n')
    (root/'APPENDIX.md').write_text('\n'.join(appendix)+'\n')
    (root/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    (root/'README.md').write_text('# AppWorld KL variant: '+root.name+'\n\n## Motivation & RQ\n\nDoes changing KL data preserve task success while improving role recovery?\n\n## Evaluation setup\n\n[Shared setup and reproduction](../RUNNING.md). Task success includes all assigned attempts. Overall role accuracy is token-weighted accuracy on whole histories within native context, using a fresh standard-data layer-12 probe.\n\n## Factual observations\n\nAll values are percentages.\n\n'+'\n'.join(table)+'\n\n[Confusions and coverage](APPENDIX.md).\n\n## Message\n\nInterpret the task and role metrics together in the [main comparison](../../README.md); role recovery alone does not establish retained agent capability.\n\n[Reproduction](../RUNNING.md)\n')
    refresh_table()


if __name__ == '__main__':
    main()
