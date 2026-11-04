"""Update only measured epoch results in the authored experiment reports."""
import argparse
import fcntl
import json
from pathlib import Path


def update(root):
    results = []
    for path in sorted((root/'checkpoints').glob('epoch*/post_training/results/tau2_retail.json')):
        row = json.loads(path.read_text())
        epoch = int(path.parents[2].name.removeprefix('epoch'))
        task, probe = row['task'], row['probe']
        if task['samples'] != 114 or probe['trajectories'] != 114:
            raise ValueError('Expected every retail task')
        if task['input_sha256'] != probe['input_sha256']:
            raise ValueError('Task and role evaluation cohorts differ')
        if probe['roles'] != ['system', 'user', 'cot', 'assistant', 'tool']:
            raise ValueError('Unexpected probe role order')
        matrix = probe['confusion_counts']
        total = sum(map(sum, matrix))
        if not total or abs(sum(matrix[i][i] for i in range(5))/total-probe['accuracy']) > 1e-9:
            raise ValueError('Confusion matrix and accuracy differ')
        results.append((epoch, path, row))
    lines = ['| Epoch | Task success | Overall role | System | User | Reasoning | Assistant | Tool |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    appendix = ['# Retail checkpoint details', '',
                'Confusion matrices count content tokens: rows are true roles and columns are predicted roles. '
                'The percentage matrix normalizes each true-role row. Reasoning is the `cot` label. '
                'Headers and synthetic schemas are excluded. Whole histories above native context are excluded only from role metrics.', '']
    small = []
    for epoch, path, row in results:
        probe = row['probe']; task = row['task']['metrics']['retail_scorer']
        values = [task['success_rate'], probe['accuracy'], *probe['per_role_recall']]
        lines.append('| '+str(epoch)+' | '+' | '.join(f'{100*x:.1f}' for x in values)+' |')
        rel = path.relative_to(root)
        appendix += [f'## Epoch {epoch}', '',
                     f"Task success: {task['successes']}/{task['total']}. Role coverage: {probe['scored_trajectories']}/114 histories. "
                     f"Native context: {probe['context_limit']:,} tokens. [Counts and provenance]({rel}).", '',
                     '| True / predicted | System | User | Reasoning | Assistant | Tool |',
                     '|---|---:|---:|---:|---:|---:|']
        names = ['System','User','Reasoning','Assistant','Tool']
        matrix = probe['confusion_counts']
        for name, counts in zip(names, matrix):
            appendix.append('| '+name+' | '+' | '.join(str(x) for x in counts)+' |')
        appendix += ['', 'Row percentages:', '',
                     '| True / predicted | System | User | Reasoning | Assistant | Tool |',
                     '|---|---:|---:|---:|---:|---:|']
        for name, counts in zip(names, matrix):
            support = sum(counts)
            appendix.append('| '+name+' | '+' | '.join(f'{100*x/support:.1f}' if support else '—' for x in counts)+' |')
        appendix += ['', 'Excluded histories: '+str(len(probe['excluded_trajectories']))+'. '
                     'The linked result records every exclusion and its token length.', '']
        small.append(dict(epoch=epoch, task=task, role_accuracy=probe['accuracy'],
                          per_role_recall=probe['per_role_recall'],confusion_counts=matrix,
                          scored_histories=probe['scored_trajectories'],result=str(rel)))
    if not results:
        lines += ['', 'No checkpoint evaluations have completed yet. Epochs 0–10 will appear only after validation.']
        appendix += ['No checkpoint evaluations have completed yet.', '']
    else:
        lines += ['', 'All values are percentages. Epoch 0 is the untrained checkpoint evaluated with the same protocol. '
                  '[Checkpoint coverage and confusion matrices](APPENDIX.md).']
    path = root/'README.md'
    text = path.read_text()
    start, end = '<!-- epoch-results -->', '<!-- /epoch-results -->'
    before, rest = text.split(start)
    _, after = rest.split(end)
    text = before+start+'\n'+'\n'.join(lines)+'\n'+end+after
    message = 'Results are pending. No conclusion about change across epochs is supported yet.'
    if results:
        latest_epoch, _, latest = results[-1]
        task = latest['task']['metrics']['retail_scorer']['success_rate']
        role = latest['probe']['accuracy']
        message = (f'The latest evaluated checkpoint is epoch {latest_epoch}: task success is '
                   f'{100*task:.1f}% and overall role accuracy is {100*role:.1f}%.')
        if results[0][0] == 0 and latest_epoch > 0:
            base = results[0][2]
            dt = 100*(task-base['task']['metrics']['retail_scorer']['success_rate'])
            dr = 100*(role-base['probe']['accuracy'])
            message += (f' Relative to epoch 0, these change by {dt:+.1f} and {dr:+.1f} percentage points, '
                        'respectively. The table gives the measured progression across completed epochs.')
        message += (' These single-attempt measurements are descriptive. Role accuracy and task success '
                    'measure different behavior; differing role-score coverage can affect comparisons. '
                    'Compare both refresh schedules at matched epochs before attributing differences to refresh frequency.')
    ms, me = '<!-- epoch-message -->', '<!-- /epoch-message -->'
    if ms in text:
        before, rest = text.split(ms)
        _, after = rest.split(me)
        text = before+ms+'\n'+message+'\n'+me+after
    path.write_text(text)
    (root/'APPENDIX.md').write_text('\n'.join(appendix))
    (root/'metrics.json').write_text(json.dumps(small,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    with (args.root/'.report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        update(args.root)
