"""Publish compact learning results from a training run's numeric event log."""
import argparse
import json
from pathlib import Path
from statistics import mean

ROLES = ('system', 'user', 'reasoning', 'assistant', 'tool')


def replace_section(text, name, body, before=None):
    begin, end = f'<!-- {name} -->', f'<!-- /{name} -->'
    block = f'{begin}\n{body.rstrip()}\n{end}\n\n'
    if begin in text:
        return text[:text.index(begin)] + block + text[text.index(end) + len(end):].lstrip('\n')
    if before:
        return text.replace(before, block + before)
    return text.rstrip() + '\n\n' + block


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--finished-job', action='store_true', help='The training job has exited; distinguish stopped from running.')
    parser.add_argument('--report', type=Path, default=Path('experiment_reports/training_against_role_probes'))
    args = parser.parse_args()
    rows = []
    lines = (args.run / 'rank0.jsonl').read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # The writer may be appending its last row during a live report.
            if index != len(lines) - 1:
                raise
    updates = {r['step']: r for r in rows if r['event'] == 'update'}
    validations = {r['step']: r for r in rows if r['event'] == 'validation'}
    if not updates or not validations:
        raise ValueError('No complete training update and validation available')
    latest = max(updates)
    first, last = validations[min(validations)], validations[max(validations)]
    completed = any(r['event'] == 'completed' for r in rows)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    result = dict(completed=completed, completed_updates=latest,
                  expected_trajectories=manifest['expected_trajectories'],
                  source_sha256=manifest['sha256'],
                  updates=[updates[k] for k in sorted(updates)],
                  validations=[validations[k] for k in sorted(validations)])
    (args.report / 'training_results.json').write_text(json.dumps(result, indent=2) + '\n')
    state = 'completed' if completed else ('stopped before completion' if args.finished_job else 'in progress')
    body = f'### Learning results\n\nTraining is **{state}**, with **{latest} model updates** recorded. '
    body += f'The latest development evaluation is after update {last["step"]}.\n\n'
    body += '| Model / probe | Overall | Macro recall | System | User | Reasoning | Assistant | Tool |\n'
    body += '|---|' + '---:|' * 7 + '\n'
    for label, row in [('Base model / original probe', first), (f'Update {last["step"]} / refitted probe', last)]:
        m = row['metrics']
        values = [m['accuracy'], m['macro_recall'], *m['per_role_recall']]
        body += '| ' + label + ' | ' + ' | '.join(f'{x:.1%}' for x in values) + ' |\n'
    recent = [updates[k] for k in sorted(updates)[-10:]]
    body += f'\nAcross the last {len(recent)} recorded updates, mean role loss was '
    body += f'{mean(r["role_loss"] for r in recent):.3f}, mean KL was '
    body += f'{mean(r["kl"] for r in recent):.4f}, and mean update-plus-refit time was '
    body += f'{mean(r["total_seconds"] for r in recent):.1f} seconds '
    body += '(excluding periodic validation and checkpoint saving).\n\n'
    body += 'The development set contains eight tasks; this is an early signal, not a precise\n'
    body += 'generalization estimate. Model and probe both change between rows.\n'
    changes = [(role, after - before) for role, before, after in zip(
        ROLES, first['metrics']['per_role_recall'], last['metrics']['per_role_recall'])]
    declines = [(role, delta) for role, delta in changes if delta <= -0.05]
    if declines:
        body += 'Per-role regressions of at least five percentage points: ' + ', '.join(
            f'{role} ({delta * 100:+.1f} points)' for role, delta in declines) + '.\n'
        body += 'Overall accuracy should therefore not be read as uniform improvement across roles.\n'
    body += '[Development confusion matrices](APPENDIX.md#development-confusions) separate the roles.\n'
    path = args.report / 'README.md'
    text = replace_section(path.read_text(), 'training-results', body, before='## Message')
    text = text.replace('production training has not started.', 'production training has started.')
    text = text.replace('No learning result is available yet.', 'The table above reports the current learning result.')
    message_start = text.index('## Message')
    reproduction_start = text.index('See [RUNNING.md]', message_start)
    direction = 'increased' if last['metrics']['accuracy'] > first['metrics']['accuracy'] else 'did not increase'
    conclusion = f'## Message\n\nAfter {last["step"]} updates, role accuracy {direction} on the small fixed development\nset with the continually refitted probe. '
    if declines:
        conclusion += 'The per-role regressions show that this does not\nmean all roles became easier to recover. '
    conclusion += 'An independently fitted probe and new task\nrollouts are needed to establish broader recoverability and behavioral effects.\n\n'
    text = text[:message_start] + conclusion + text[reproduction_start:]
    path.write_text(text.rstrip() + '\n')
    appendix = '## Development confusions\n\nRows are true content roles; columns are predicted roles. Values are token counts.\n'
    for label, row in [('Before training', first), (f'After update {last["step"]}', last)]:
        appendix += f'\n### {label}\n\n| True / predicted | ' + ' | '.join(ROLES) + ' |\n|---|' + '---:|' * 5 + '\n'
        for role, counts in zip(ROLES, row['metrics']['confusion_counts']):
            appendix += '| ' + role + ' | ' + ' | '.join(map(str, counts)) + ' |\n'
        appendix += '\nRow-normalized percentages:\n\n| True / predicted | ' + ' | '.join(ROLES) + ' |\n|---|' + '---:|' * 5 + '\n'
        for role, counts in zip(ROLES, row['metrics']['confusion_counts']):
            n = sum(counts)
            appendix += '| ' + role + ' | ' + ' | '.join(f'{x/n:.1%}' if n else '—' for x in counts) + ' |\n'
    path = args.report / 'APPENDIX.md'
    path.write_text(replace_section(path.read_text(), 'development-results', appendix).rstrip() + '\n')
    print(json.dumps(dict(updates=latest, completed=completed, development_accuracy=last['metrics']['accuracy'])))


if __name__ == '__main__':
    main()
