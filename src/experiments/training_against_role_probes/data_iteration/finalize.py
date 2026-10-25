"""Validate the expanded-data model and refresh only the design-iteration report."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path('experiment_reports/training_against_role_probes')
ROOT = BASE / 'data_iteration/same_batch_27000'
BENCHMARKS = [('appworld_test', 'AppWorld test_normal', 168, 'appworld_success'),
              ('tau2_retail', 'τ² retail', 114, 'retail_scorer')]
ROLES = ['System', 'User', 'Reasoning', 'Assistant', 'Tool']
VARIANTS = [('AppWorld + agentic KL (800)', BASE / 'kl_iteration/agentic800'),
            ('AppWorld + same-batch KL (2,700)', BASE / 'kl_iteration/same_batch'),
            ('AppWorld + same-batch KL (27,000)', ROOT)]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_block(text, marker, body):
    start, end = f'<!-- {marker} -->', f'<!-- /{marker} -->'
    assert text.count(start) == text.count(end) == 1
    left, rest = text.split(start)
    _, right = rest.split(end)
    return left + start + '\n' + body + '\n' + end + right


def main():
    # Independently replay all saved prediction arrays against confusion counts.
    validator = 'src/experiments/training_against_role_probes/kl_iteration/validate.py'
    subprocess.run([sys.executable, validator, '--root', str(ROOT)], check=True)
    manifest = read(ROOT / 'runs/training/epoch1/manifest.json')
    corpus = read(ROOT / 'data/train27000/manifest.json')
    assert corpus['trajectories'] == manifest['expected_trajectories'] == 27000
    assert len(corpus['attempts_per_task']) == 90
    assert set(corpus['attempts_per_task'].values()) == {300}
    assert manifest['kl_sampling'] == 'same_role_batch' and manifest['epochs'] == 1
    assert manifest['sha256']['trajectories'] == corpus['sha256'] == digest(Path(manifest['trajectories']))
    assert manifest['sha256']['kl_anchors'] == digest(Path(manifest['kl_anchors']))
    events = [json.loads(line) for line in (ROOT / 'runs/training/epoch1/rank0.jsonl').open()]
    assert events[-1]['event'] == 'completed' and events[-1]['updates'] == 1688
    assert {e['step'] for e in events if e['event'] == 'update'} == set(range(1, 1689))
    export = read(ROOT / 'data/post_training/model/export.json')
    checkpoint = ROOT / 'runs/training/epoch1/checkpoints/step01688'
    assert Path(export['checkpoint']).resolve() == checkpoint.resolve()
    assert export['adapter_sha256'] == digest(checkpoint / 'adapter/adapter_model.safetensors')
    metadata = read(ROOT / 'post_training/probes/layer_12.json')
    assert metadata['fresh_fit'] and metadata['model_revision'] == export['adapter_sha256']
    tasks = read(ROOT / 'post_training/task_results.json')
    results, public = {}, {}
    appendix = ['# Role confusions', '', 'Rows are true roles; columns are predicted roles. Values are row percentages.', '']
    for key, label, expected, metric in BENCHMARKS:
        task = tasks[key]
        assembled = read(ROOT / f'data/post_training/{key}/manifest.json')
        assert all(m['model_revision'] == export['adapter_sha256'] for m in assembled['source_manifests'])
        probe = read(ROOT / f'predictions/post_training/{key}/summary.json')
        assert task['samples'] == probe['trajectories'] == expected
        assert task['input_sha256'] == probe['input_sha256'] == digest(ROOT / f'data/post_training/{key}/trajectories.jsonl')
        assert probe['context_policy'] == 'Whole conversation in one forward pass; no chunking or truncation'
        assert not probe.get('segmented_trajectories')
        assert probe['roles'] == ['system', 'user', 'cot', 'assistant', 'tool']
        assert all(r['tokens'] > probe['context_limit'] for r in probe['excluded_trajectories'])
        m = task['metrics'][metric]
        assert m['total'] == expected and abs(m['successes']/expected-m['success_rate']) < 1e-12
        counts = probe['confusion_counts']
        for i, row in enumerate(counts):
            if sum(row):
                assert abs(row[i]/sum(row)-probe['per_role_recall'][i]) < 1e-12
        results[key] = dict(task=task, probe=probe, task_metric=metric)
        public[key] = dict(task=m, evaluation_errors=task['sample_errors'],
                           role_accuracy=probe['accuracy'], roles=probe['roles'],
                           per_role_recall=probe['per_role_recall'], confusion_counts=counts,
                           histories_scored=probe['scored_trajectories'], total_histories=expected,
                           unscorable_histories=len(probe['excluded_trajectories']))
        appendix += [f'## {label}', '', '| True / predicted | ' + ' | '.join(ROLES) + ' |',
                     '|---|---:|---:|---:|---:|---:|']
        for role, row in zip(ROLES, counts):
            appendix.append('| ' + role + ' | ' + ' | '.join(f'{100*x/sum(row):.2f}%' if sum(row) else '—' for x in row) + ' |')
        appendix += ['', f"Coverage: {probe['scored_trajectories']}/{expected}. All attempts remain in task denominators.", '']
    write(ROOT / 'post_training/results.json', results)
    write(ROOT / 'publication_metrics.json', public)
    (ROOT / 'APPENDIX.md').write_text('\n'.join(appendix))

    comparison = read(BASE / 'post_training/comparison.json')
    comparison = comparison.get('models', comparison)
    models = {name: comparison[name] for name in ('Base', 'AppWorld-trained')}
    models.update({name: read(root / 'post_training/results.json') for name, root in VARIANTS})
    table = ['| Benchmark | Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    coverage = []
    for key, label, expected, metric in BENCHMARKS:
        for name, model in models.items():
            row = model[key]
            values = [row['task']['metrics'][metric]['success_rate'], row['probe']['accuracy'], *row['probe']['per_role_recall']]
            cells = [f'{100*v:.1f}' for v in values]
            table.append('| ' + label + ' | ' + name + ' | ' + ' | '.join(cells) + ' |')
        coverage.append(f"{label}: {results[key]['probe']['scored_trajectories']}/{expected}")
    old = models['AppWorld + same-batch KL (2,700)']
    differences = []
    for key, label, _, metric in BENCHMARKS:
        task_delta = 100*(results[key]['task']['metrics'][metric]['success_rate']-old[key]['task']['metrics'][metric]['success_rate'])
        role_delta = 100*(results[key]['probe']['accuracy']-old[key]['probe']['accuracy'])
        differences.append(f'{label}: task success changes by {task_delta:+.1f} percentage points and overall role accuracy by {role_delta:+.1f} points.')
    conclusion = ' '.join(differences) + '\n\nThe data-size comparison holds the same-batch KL objective and optimizer settings fixed, but one epoch on ten times more data also increases updates from 169 to 1,688. It does not isolate data diversity from training compute. Each model is evaluated on its own histories, and overall role accuracy need not imply uniform improvement across roles.'
    body = '''<a id="appworld-training-kl-data-iteration"></a>
### AppWorld training: design iterations

We vary KL data and then the number of training trajectories per task. The original AppWorld model uses 2,700 trajectories and ordinary non-tool KL data capped at 2,048 tokens. Agentic KL instead samples from 800 saved base-model training rollouts. Same-batch KL scores the exact trajectories used for each role-loss update. Both agentic variants retain complete histories and score model-authored reasoning, answers and tool-call arguments.

The data-size iteration expands the same 90 training tasks from 30 to 300 trajectories each (2,700 to 27,000), retaining the original attempts and adding 270 seeds per task without outcome filtering. It starts from the base model and keeps same-batch KL and optimizer settings fixed. All runs train for one epoch; the larger corpus therefore also uses ten times as many updates. Evaluation tasks do not enter training or KL.

All values are percentages. Base and original AppWorld rows repeat the comparison above. Task success includes every assigned attempt; role accuracy uses whole histories within native context.

''' + '\n'.join(table) + '''

The original coverage and generation-precision limitations also apply here. New models use fresh trajectories and independently fitted standard-data probes. This design comparison covers AppWorld and retail; SWE is managed by a separate evaluation effort.

''' + '**27,000-trajectory model role-score coverage:** ' + '; '.join(coverage) + '.\n\n[KL setup and earlier coverage](kl_iteration/RUNNING.md); [larger-data reproduction](data_iteration/same_batch_27000/RUNNING.md). Detailed confusions: [fixed 800](kl_iteration/agentic800/APPENDIX.md), [same batch, 2,700](kl_iteration/same_batch/APPENDIX.md), [same batch, 27,000](data_iteration/same_batch_27000/APPENDIX.md).'
    path = BASE / 'README.md'
    original_text = path.read_text()
    # Existing-model task/role cells must still agree with the independently
    # maintained first comparison, even if its SWE rows changed concurrently.
    first_block = original_text.split('<!-- fresh-rollout-results -->')[1].split('<!-- /fresh-rollout-results -->')[0]
    for key, label, _, _ in BENCHMARKS:
        for name in ('Base', 'AppWorld-trained'):
            matching = [line for line in table if line.startswith(f'| {label} | {name} |')]
            assert len(matching) == 1 and matching[0] in first_block
    text = replace_block(original_text, 'kl-iteration-results', body)
    text = replace_block(text, 'kl-iteration-message', conclusion)
    assert text.rstrip().endswith('[How to reproduce the experiment](RUNNING.md)')
    path.write_text(text)
    child = ROOT / 'README.md'
    text = child.read_text()
    before, rest = text.split('## Factual observations\n', 1)
    text = before + '## Factual observations\n\nAll values are percentages.\n\n' + '\n'.join([table[0], table[1], *[r for r in table[2:] if 'same-batch KL (27,000)' in r]]) + '\n\n[Confusions and coverage](APPENDIX.md).\n\n## Message\n\n' + conclusion + '\n\n[Reproduction](RUNNING.md)\n'
    child.write_text(text)
    print('Validated expanded-data results and updated the ten-row design comparison.', flush=True)


if __name__ == '__main__':
    main()
