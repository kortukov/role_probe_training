"""Validate matched cohorts and update measured tables without replacing prose."""
import argparse
import fcntl
import json
from pathlib import Path

ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
NAMES = ['System', 'User', 'Reasoning', 'Assistant', 'Tool']
EXPECTED = {'appworld_test': 168, 'tau2_retail': 114}


def validate(row, expected):
    task, probe = row['task'], row['probe']
    if task['samples'] != expected or probe['trajectories'] != expected:
        raise ValueError('Incomplete assigned cohort')
    if task['input_sha256'] != probe['input_sha256'] or probe['roles'] != ROLES:
        raise ValueError('Task/probe cohort or role order mismatch')
    matrix = probe['confusion_counts']
    if len(matrix) != 5 or any(len(x) != 5 or any(v < 0 for v in x) for x in matrix):
        raise ValueError('Invalid confusion matrix')
    total = sum(map(sum, matrix))
    if not total or abs(sum(matrix[i][i] for i in range(5))/total-probe['accuracy']) > 1e-8:
        raise ValueError('Confusion matrix and accuracy differ')
    if probe['scored_trajectories'] + len(probe['excluded_trajectories']) != expected:
        raise ValueError('Coverage does not account for every attempt')
    for i, counts in enumerate(matrix):
        support = sum(counts)
        if support and abs(counts[i]/support-probe['per_role_recall'][i]) > 1e-8:
            raise ValueError('Confusion matrix and role recall differ')
    for metric in task['metrics'].values():
        if metric['total'] != expected or abs(metric['successes']/expected-metric['success_rate']) > 1e-8:
            raise ValueError('Task success denominator changed')
    export = row.get('model_revision', probe['model_revision'])
    if export != probe['model_revision']:
        raise ValueError('Model revision mismatch')


def update(root):
    lines = []
    appendix = ['# Checkpoint role perception', '',
                'Rows are true roles and columns are probe predictions. Counts include only content tokens. '
                'Each percentage row divides by its true-role token count. Headers and synthetic schemas are excluded. '
                'Role coverage excludes whole histories above native context; task success retains every assigned attempt.', '']
    summary = []
    for benchmark, expected in EXPECTED.items():
        title = {'appworld_test': 'AppWorld normal test', 'tau2_retail': 'τ² retail'}[benchmark]
        lines += [f'**{title}**', '',
                  '| Total epochs | Task success | Overall role | System | User | Reasoning | Assistant | Tool |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        seen = set()
        for path in sorted((root/'evaluations').glob(f'epoch*/post_training/results/{benchmark}.json')):
            row = json.loads(path.read_text())
            epoch = int(path.parents[2].name.removeprefix('epoch'))
            if epoch not in (1, 2, 3, 5):
                raise ValueError('Unexpected evaluated epoch')
            validate(row, expected)
            seen.add(epoch)
            probe = row['probe']
            metric = row['task']['metrics'][{'appworld_test':'appworld_success','tau2_retail':'retail_scorer'}[benchmark]]
            values = [metric['success_rate'], probe['accuracy'], *probe['per_role_recall']]
            lines.append('| '+str(epoch)+' | '+' | '.join(f'{100*x:.1f}' for x in values)+' |')
            relative = path.relative_to(root)
            appendix += [f'## {title}, epoch {epoch}', '',
                         f"Task success: {metric['successes']}/{expected}. Role coverage: {probe['scored_trajectories']}/{expected} histories. "
                         f"Native context: {probe['context_limit']:,} tokens. [Counts and provenance]({relative}).", '',
                         '| True / predicted | '+' | '.join(NAMES)+' |', '|---|---:|---:|---:|---:|---:|']
            for name, counts in zip(NAMES, probe['confusion_counts']):
                appendix.append('| '+name+' | '+' | '.join(str(v) for v in counts)+' |')
            appendix += ['', 'Row percentages:', '', '| True / predicted | '+' | '.join(NAMES)+' |',
                         '|---|---:|---:|---:|---:|---:|']
            for name, counts in zip(NAMES, probe['confusion_counts']):
                support = sum(counts)
                appendix.append('| '+name+' | '+' | '.join(f'{100*v/support:.1f}' if support else '—' for v in counts)+' |')
            appendix += ['', 'Mean probability assigned to the true role, averaged across tokens of that role '
                         '(distinct from argmax recall):', '', '| '+' | '.join(NAMES)+' |', '|---:|---:|---:|---:|---:|']
            appendix += ['| '+' | '.join(f'{100*v:.2f}' for v in probe['mean_true_role_probability'])+' |', '',
                         f"Excluded histories: {len(probe['excluded_trajectories'])}; identities and lengths are in the linked result.", '']
            summary.append(dict(epoch=epoch,benchmark=benchmark,task=metric,
                                accuracy=probe['accuracy'],per_role_recall=probe['per_role_recall'],
                                mean_true_role_probability=probe['mean_true_role_probability'],
                                confusion_counts=probe['confusion_counts'],
                                scored_histories=probe['scored_trajectories'],result=str(relative)))
        for epoch in sorted({1,2,3,5}-seen):
            lines.append('| '+str(epoch)+' | '+' | '.join(['pending']*7)+' |')
        lines += ['']
    lines += ['All values are percentages. Epoch 1 reuses the original saved evaluation; later epochs use fresh attempts. '
              '[Coverage, probability summaries, and confusion matrices](APPENDIX.md).']
    readme = root/'README.md'
    before, rest = readme.read_text().split('<!-- epoch-results -->')
    _, after = rest.split('<!-- /epoch-results -->')
    readme.write_text(before+'<!-- epoch-results -->\n'+'\n'.join(lines)+'\n<!-- /epoch-results -->'+after)
    (root/'APPENDIX.md').write_text('\n'.join(appendix)+'\n')
    (root/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    with (args.root/'.report.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        update(args.root)
