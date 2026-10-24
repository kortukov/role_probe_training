"""Refresh only the KL-iteration table; preserve the concurrent SWE report."""
import hashlib
import json
from pathlib import Path

ROOT = Path('experiment_reports/training_against_role_probes')
BENCHMARKS = [('appworld_test', 'AppWorld test_normal', 168, 'appworld_success'),
              ('tau2_retail', 'τ² retail', 114, 'retail_scorer'),
              ('swe_bench', 'SWE-bench Verified', 500, 'swe_bench_scorer')]
VARIANTS = [('agentic800', 'AppWorld + agentic KL (800)'),
            ('same_batch', 'AppWorld + same-batch KL')]


def main():
    comparison = json.loads((ROOT/'post_training/comparison.json').read_text())
    # The canonical comparison follows the other agent's corrected SWE publication.
    if 'models' in comparison:
        comparison = comparison['models']
    table = ['### AppWorld training: KL-data iteration\n',
        'This comparison asks whether changing the KL data preserves task performance '
        'while retaining role recovery. Both new runs keep the original AppWorld role-training '
        'data and optimizer settings. Agentic KL reuses 800 base-model training rollouts '
        'without selecting by outcome, scoring model-authored content over complete histories. '
        'Unlike the original KL run, these histories are not capped at 2,048 tokens. '
        'Same-batch KL scores the exact AppWorld trajectories used for each role-loss update, across the full 2,700-trajectory training corpus. Both variants score the same model-authored roles.\n',
        'All values are percentages; **pending** means no completed evaluation is available. '
        'Base and original AppWorld rows repeat the published comparison above.\n',
        '| Benchmark | Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    coverage = []
    for key, label, expected, metric in BENCHMARKS:
        for model in ['Base', 'AppWorld-trained']:
            row = comparison[model][key]
            values = [row['task']['metrics'][row['task_metric']]['success_rate'],
                      row['probe']['accuracy'], *row['probe']['per_role_recall']]
            table.append('| '+label+' | '+model+' | '+' | '.join(f'{100*x:.1f}' for x in values)+' |')
        for variant, name in VARIANTS:
            root = ROOT/'kl_iteration'/variant
            result_path = root/'post_training/results.json'
            cells = ['pending']*7
            if result_path.exists():
                results = json.loads(result_path.read_text())
                row = results[key]
                task, probe = row['task'], row['probe']
                assert task['samples'] == probe['trajectories'] == expected
                assert task['input_sha256'] == probe['input_sha256']
                assert probe['context_policy'] == 'Whole conversation in one forward pass; no chunking or truncation'
                probe_file = root/'post_training/probes/layer_12.npz'
                assert probe['probe_sha256'] == hashlib.sha256(probe_file.read_bytes()).hexdigest()
                values = [task['metrics'][metric]['success_rate'], probe['accuracy'], *probe['per_role_recall']]
                cells = [f'{100*x:.1f}' for x in values]
                coverage.append(f"{label}, {name}: {probe['scored_trajectories']}/{expected}")
            table.append('| '+label+' | '+name+' | '+' | '.join(cells)+' |')
    table += ['\nThe coverage and generation-setting limitations of the original rows also apply here. '
              'New results will use fresh trajectories and independently fitted standard-data probes. '
              '[Run details](kl_iteration/RUNNING.md).']
    if coverage:
        table.append('\n**New-model role-score coverage:** '+ '; '.join(coverage)+'. Over-limit attempts remain in task denominators. Detailed confusion matrices are in each variant’s appendix.')
    start, end = '<!-- kl-iteration-results -->', '<!-- /kl-iteration-results -->'
    path = ROOT/'README.md'; text = path.read_text()
    block = start+'\n'+'\n'.join(table)+'\n'+end
    if start in text:
        left, rest = text.split(start, 1); _, right = rest.split(end, 1)
        text = left+block+right
    else:
        marker = '<!-- /fresh-rollout-results -->'
        text = text.replace(marker, marker+'\n\n'+block, 1)
    path.write_text(text)


if __name__ == '__main__':
    main()
