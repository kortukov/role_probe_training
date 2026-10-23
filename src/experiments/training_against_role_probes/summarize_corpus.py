"""Publish compact baseline metrics and detailed confusion matrices."""
import argparse
import json
from pathlib import Path
import shutil

ROLES = ('system', 'user', 'cot', 'assistant', 'tool')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-trajectories', type=int, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.baseline / 'metrics.json').read_text())
    assert metrics['trajectories'] == args.expected_trajectories
    manifest = json.loads((args.baseline / 'manifest.json').read_text())
    assert metrics['input_sha256'] == manifest['input_sha256']
    args.report.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.baseline / 'metrics.json', args.report / 'baseline_metrics.json')
    shutil.copyfile(args.baseline / 'manifest.json', args.report / 'baseline_manifest.json')
    text = ['# Baseline supporting results', '',
        f"All {metrics['trajectories']:,} training histories were evaluated before model updates. "
        'Roles come from message/channel metadata. `cot` denotes assistant reasoning. '
        'Only original content tokens contribute; headers, boundaries and synthetic tool schemas are excluded.', '',
        '## Context lengths', '',
        'These lengths include the complete rendered context, including tool schemas and structural tokens; '
        'they differ from the scored content-token denominator.', '',
        '| Total tokens | Median history | 95th percentile | Longest history |',
        '| ---: | ---: | ---: | ---: |',
        '| ' + ' | '.join(f"{metrics['tokens'][key]:,.0f}" for key in ('total', 'median', 'p95', 'max')) + ' |', '',
        '## Per-role results', '',
        'Recall is the fraction of a role’s content tokens whose highest-probability label is correct. '
        'Mean correct-label probability averages the probability assigned to the ground-truth role; '
        'it is not argmax accuracy. Both are percentages.', '',
        '| Layer | Role | Content tokens | Recall (%) | Mean correct-label probability (%) |',
        '| ---: | --- | ---: | ---: | ---: |']
    for layer, result in metrics['layers'].items():
        for role in ROLES:
            row = result['roles'][role]
            fmt = lambda value: '—' if value is None else f'{100 * value:.2f}'
            text.append(f"| {layer} | {role} | {row['tokens']:,} | {fmt(row['recall'])} | {fmt(row['mean_true_probability'])} |")
    text += ['', '## Confusion matrices', '',
        'Rows are ground-truth roles and columns are predicted roles. Cells are content-token counts; '
        'row-normalized percentages follow each count matrix. No probe fitting uses these histories.', '']
    for layer, result in metrics['layers'].items():
        text += [f'### Layer {layer}', '', '| True / predicted | ' + ' | '.join(ROLES) + ' |',
                 '| --- | ' + ' | '.join(['---:'] * len(ROLES)) + ' |']
        for role, row in zip(ROLES, result['confusion_matrix']):
            text.append('| ' + role + ' | ' + ' | '.join(f'{v:,}' for v in row) + ' |')
        text += ['', '| True / predicted (%) | ' + ' | '.join(ROLES) + ' |',
                 '| --- | ' + ' | '.join(['---:'] * len(ROLES)) + ' |']
        for role, row in zip(ROLES, result['confusion_matrix']):
            total = sum(row)
            text.append('| ' + role + ' | ' + ' | '.join(f'{100*v/total:.2f}' if total else '—' for v in row) + ' |')
        text.append('')
    text += ['Machine-readable results: [baseline_metrics.json](baseline_metrics.json). '
             'Execution metadata: [baseline_manifest.json](baseline_manifest.json).', '']
    (args.report / 'APPENDIX.md').write_text('\n'.join(text))
    print('| Layer | Token accuracy (%) | Macro recall (%) | Mean trajectory accuracy (%) |')
    print('| ---: | ---: | ---: | ---: |')
    for layer, result in metrics['layers'].items():
        print('| ' + layer + ' | ' + ' | '.join(f'{100 * result[k]:.2f}' for k in
              ('token_accuracy', 'macro_recall', 'trajectory_accuracy')) + ' |')
    print('Majority-role token accuracy:', next(iter(metrics['layers'].values()))['majority_role_accuracy'])


if __name__ == '__main__':
    main()
