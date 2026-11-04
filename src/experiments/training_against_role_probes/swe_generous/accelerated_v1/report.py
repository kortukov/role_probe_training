"""Validate matched task/probe results and update the concise SWE comparison."""
import argparse
import hashlib
import json
from pathlib import Path

REPORT = Path('experiment_reports/training_against_role_probes/swe_generous')
COHORT = 'swe_inspect_generous'
p = argparse.ArgumentParser(__doc__)
p.add_argument('--require-tasks', action='store_true')
p.add_argument('--require-probes', action='store_true')
a = p.parse_args()
manifest = json.loads((REPORT / 'pipeline.json').read_text())
rows = []
reference_settings = None
reference_ids = None
for model in manifest['models']:
    root = Path(model['root'])
    task_file = root / f'post_training/{COHORT}_task_results.json'
    row = {'model': model['label'], 'root': str(root), 'task': None, 'probe': None}
    if task_file.exists():
        task = json.loads(task_file.read_text())
        assert task['protocol'] == 'swe_inspect_generous_v1'
        assert task['samples'] == task['metrics']['swe_bench_scorer']['total'] == 500
        source = root / f'data/{COHORT}/evaluation/trajectories.jsonl'
        digest = hashlib.sha256()
        ids = set()
        successes = errors = 0
        with source.open('rb') as stream:
            for line in stream:
                digest.update(line)
                attempt = json.loads(line)
                assert attempt['sample_id'] not in ids
                ids.add(attempt['sample_id'])
                errors += bool(attempt['error'])
                successes += not attempt['error'] and attempt.get('scores', {}).get('swe_bench_scorer', {}).get('value') == 1
        assert len(ids) == 500
        assert digest.hexdigest() == task['input_sha256']
        assert successes == task['metrics']['swe_bench_scorer']['successes']
        assert errors == task['sample_errors']
        if reference_settings is None:
            reference_settings, reference_ids = {k:v for k,v in task['settings'].items() if k != 'concurrency'}, ids
        assert {k:v for k,v in task['settings'].items() if k != 'concurrency'} == reference_settings and ids == reference_ids
        row['task'] = task
        probe_file = root / f'predictions/{COHORT}/summary.json'
        if probe_file.exists():
            probe = json.loads(probe_file.read_text())
            assert probe['input_sha256'] == task['input_sha256']
            assert probe['trajectories'] == 500
            assert probe['scored_trajectories'] + len(probe['excluded_trajectories']) == 500
            row['probe'] = probe
    rows.append(row)
if a.require_tasks or a.require_probes:
    assert all(r['task'] is not None for r in rows), 'Task results incomplete'
if a.require_probes:
    assert all(r['probe'] is not None for r in rows), 'Probe results incomplete'
(REPORT / 'results.json').write_text(json.dumps({'protocol': manifest['protocol'], 'models': rows}, indent=2) + '\n')
base = rows[0]['task']
base_rate = base['metrics']['swe_bench_scorer']['success_rate'] if base else None
lines = ['# Matched SWE-bench utility comparison', '', '## Motivation and research question', '',
         'Does role-probe training change agentic task utility relative to the base model under the same Inspect harness? The target is a within-harness comparison, not reproduction of a model-card score.', '',
         '## Evaluation setup', '',
         'All seven models receive the same 500 SWE-bench Verified tasks with one attempt per task. We use the stock agent, tools and scorer from pinned Inspect Evals, with high reasoning effort, BF16 weights and native context. The explicit 2,000-message and 210-second tool limits are shared across models. A cluster sandbox adapter activates repository Python, supplies localhost resolution and snapshots initial files before model edits.', '',
         'Task success is successes divided by all 500 assigned tasks; evaluation errors remain in the denominator. Change from base is the difference in percentage points. Layer-12 role accuracy is the fraction of scored content tokens classified correctly by each model’s independently fitted probe. Histories are replayed whole; over-context histories retain task outcomes but have no role score.', '',
         'Residual test-environment and offline-network limitations can depress scores. They are held fixed across models; absolute benchmark parity is not claimed. No task-specific fixture, test-name remapping or custom scoring repair is used in this cohort.', '',
         'ASTRA uses concurrency 8 for shards 0–1 and 16 for shards 2–3; other models use 4. Per-attempt budgets and scoring are unchanged. Batching may affect numerical sampling and wall-clock contention, so this scheduling difference limits strict matching.', '', '## Observations', '', '| Model | Task success | Change from base | Evaluation errors | Limit stops | Role accuracy | Role coverage |',
         '|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    task, probe = row['task'], row['probe']
    success = f"{100*task['metrics']['swe_bench_scorer']['success_rate']:.1f}%" if task else 'Pending'
    delta = f"{100*(task['metrics']['swe_bench_scorer']['success_rate']-base_rate):+.1f}" if task and base_rate is not None else '—'
    error = str(task['sample_errors']) if task else '—'
    accuracy = f"{100*probe['accuracy']:.1f}%" if probe else 'Pending'
    coverage = f"{probe['scored_trajectories']}/500" if probe else '—'
    lines.append(f"| {row['model']} | {success} | {delta} | {error} | {task['limited_samples'] if task else '—'} | {accuracy} | {coverage} |")
lines += ['', 'Only this matched cohort enters the comparison. Earlier baseline diagnostics and previous SWE cohorts remain separate.', '', '## Interpretation', '']
if all(r['task'] for r in rows):
    changes = [r['task']['metrics']['swe_bench_scorer']['success_rate'] - base_rate for r in rows[1:]]
    if all(x < 0 for x in changes):
        lines.append('All six trained models have lower task success than the matched baseline in this harness.')
    elif all(x > 0 for x in changes):
        lines.append('All six trained models have higher task success than the matched baseline in this harness.')
    else:
        lines.append('The direction and size of the utility change vary by trained model; the table reports each matched difference.')
    lines.append('These are descriptive single-attempt comparisons, not evidence of model-card parity or statistical significance.')
else:
    lines.append('Collection is in progress. No conclusion about the matched utility change is available yet.')
lines += ['', '[Role breakdowns and coverage](APPENDIX.md). [Reproduction and configuration](RUNNING.md).', '']
(REPORT / 'README.md').write_text('\n'.join(lines))
appendix = ['# Role metrics and coverage', '', 'Rows in each confusion matrix are true roles; columns are predicted roles. Values are row-normalized percentages. Zero-support rows are marked unavailable. All metrics use whole histories.', '']
for row in rows:
    appendix += [f"## {row['model']}", '']
    probe = row['probe']
    if not probe:
        appendix += ['Pending.', '']
        continue
    roles = probe['roles']
    appendix += [f"Coverage: {probe['scored_trajectories']}/500 histories; {len(probe['excluded_trajectories'])} above native context.", '', '| True / predicted | ' + ' | '.join(roles) + ' |', '|---|' + '---:|' * len(roles)]
    for role, counts in zip(roles, probe['confusion_counts']):
        total = sum(counts)
        appendix.append('| ' + role + ' | ' + ' | '.join(f'{100*v/total:.1f}' if total else '—' for v in counts) + ' |')
    appendix.append('')
(REPORT / 'APPENDIX.md').write_text('\n'.join(appendix))
print(json.dumps({'task_models': sum(r['task'] is not None for r in rows), 'probe_models': sum(r['probe'] is not None for r in rows)}))
# Replace only SWE rows when all seven matched task results are available.
# Other benchmarks and their authored interpretation remain untouched.
if all(r['task'] is not None for r in rows):
    parent = REPORT.parent / 'README.md'
    text = parent.read_text()
    replacement = []
    for row in rows:
        task, probe = row['task'], row['probe']
        cells = [f"{100*task['metrics']['swe_bench_scorer']['success_rate']:.1f}"]
        cells += [f"{100*probe['accuracy']:.1f}", *[f'{100*v:.1f}' for v in probe['per_role_recall']]] if probe else ['Pending'] * 6
        replacement.append('| SWE-bench Verified | ' + row['model'] + ' | ' + ' | '.join(cells) + ' |')
    updated = []
    inserted = False
    for line in text.splitlines():
        if line.startswith('| SWE-bench Verified |'):
            if not inserted:
                updated.extend(replacement)
                inserted = True
        else:
            updated.append(line)
    if inserted:
        parent.write_text('\n'.join(updated) + '\n')
