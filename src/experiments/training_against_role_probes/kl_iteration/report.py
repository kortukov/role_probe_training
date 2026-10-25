"""Refresh only the KL-iteration table; preserve the concurrent SWE report."""
import hashlib
import json
from pathlib import Path
from validate import validate

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
    complete = all((ROOT/'kl_iteration'/v/'post_training/results.json').exists() for v,_ in VARIANTS)
    table = ['### AppWorld training: KL-data iteration\n',
        'This comparison asks whether changing the KL data preserves task performance '
        'while retaining role recovery. Both new runs keep the original AppWorld role-training '
        'data and optimizer settings. Agentic KL reuses 800 base-model training rollouts '
        'without selecting by outcome, scoring model-authored content over complete histories. '
        'Unlike the original KL run, these histories are not capped at 2,048 tokens. '
        'Same-batch KL scores the exact AppWorld trajectories used for each role-loss update, across the full 2,700-trajectory training corpus. Both variants score the same model-authored roles.\n',
        ('All values are percentages. ' if complete else 'All values are percentages; **pending** means no completed evaluation is available. ') +
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
            row = None
            if result_path.exists():
                row = json.loads(result_path.read_text())[key]
            else:
                task_path=root/'post_training/task_results.json'
                tasks=json.loads(task_path.read_text()) if task_path.exists() else {}
                if key in tasks:
                    cells[0]=f"{100*tasks[key]['metrics'][metric]['success_rate']:.1f}"
                    summary_path=root/f'predictions/post_training/{key}/summary.json'
                    if summary_path.exists():
                        validate(root,[key])
                        row={'task':tasks[key],'probe':json.loads(summary_path.read_text())}
            if row is not None:
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
              'The new models are evaluated on fresh trajectories with independently fitted standard-data probes. '
              '[Run details](kl_iteration/RUNNING.md).']
    if coverage:
        matrix_links=(' Detailed confusion matrices: [fixed-800](kl_iteration/agentic800/APPENDIX.md), [same-batch](kl_iteration/same_batch/APPENDIX.md).' if complete else '')
        table.append('\n**New-model role-score coverage:** '+ '; '.join(coverage)+'. Over-limit attempts remain in task denominators.'+matrix_links)
    retail_summaries=[ROOT/'kl_iteration'/v/'predictions/post_training/tau2_retail/summary.json' for v,_ in VARIANTS]
    if all(p.exists() for p in retail_summaries):
        recalls=[json.loads(p.read_text())['per_role_recall'] for p in retail_summaries]
        old=comparison['AppWorld-trained']['tau2_retail']['probe']['per_role_recall']
        lower=[name for i,name in enumerate(['system','user','reasoning','assistant','tool']) if all(r[i]<old[i] for r in recalls)]
        if lower:
            table.append('\nOn retail, both new variants have lower recall for '+', '.join(lower)+' than the original AppWorld-trained model. Overall accuracy does not establish uniform improvement across roles; each model is scored on its own eligible histories.')
    start, end = '<!-- kl-iteration-results -->', '<!-- /kl-iteration-results -->'
    path = ROOT/'README.md'; text = path.read_text()
    block = start+'\n'+'\n'.join(table)+'\n'+end
    if start in text:
        left, rest = text.split(start, 1); _, right = rest.split(end, 1)
        text = left+block+right
    else:
        marker = '<!-- /fresh-rollout-results -->'
        text = text.replace(marker, marker+'\n\n'+block, 1)
    if complete:
        conclusions=[]; recovered=[]
        for variant,name in VARIANTS:
            results=json.loads((ROOT/'kl_iteration'/variant/'post_training/results.json').read_text())
            task_gains=role_gains=base_matches=0
            for key,_,_,metric in BENCHMARKS:
                task=results[key]['task']['metrics'][metric]['success_rate']
                old=comparison['AppWorld-trained'][key]['task']['metrics'][metric]['success_rate']
                base=comparison['Base'][key]['task']['metrics'][metric]['success_rate']
                task_gains+=task>old
                base_matches+=task>=base
                role_gains+=results[key]['probe']['accuracy']>comparison['Base'][key]['probe']['accuracy']
            conclusions.append(f'{name} has higher observed task success than the original AppWorld-trained model on {task_gains}/3 benchmarks, and higher overall role accuracy than base on {role_gains}/3. It matches or exceeds base task success on {base_matches}/3 benchmarks.')
            if base_matches==3: recovered.append(name)
        if not recovered:
            conclusions.append('The tested variants therefore do not fully retain base task performance across all three benchmarks. Changing KL data and retaining full KL histories are coupled in this comparison; their separate contributions are not isolated.')
        message_start,message_end='<!-- kl-iteration-message -->','<!-- /kl-iteration-message -->'
        message=message_start+'\n'+'\n\n'.join(conclusions)+'\n'+message_end
        if message_start in text:
            left,rest=text.split(message_start,1);_,right=rest.split(message_end,1)
            text=left+message+right
        else:
            text=text.replace('[How to reproduce the experiment](RUNNING.md)',message+'\n\n[How to reproduce the experiment](RUNNING.md)')
    path.write_text(text)


if __name__ == '__main__':
    main()
