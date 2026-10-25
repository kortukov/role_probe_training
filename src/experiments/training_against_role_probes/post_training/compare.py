"""Write the concise four-model comparison from complete, validated cohorts."""
import argparse
import hashlib
import json
from pathlib import Path

from report import BENCHMARKS, ROLE_NAMES, publish, section

MODELS = [('base_model', 'Base'), ('', 'AppWorld-trained'),
          ('astra_sft_1k', 'ASTRA-trained'), ('depthbench', 'DepthBench-trained')]


def compare(root, native_swe=False):
    all_results = {}
    for sub, label in MODELS:
        folder = root / sub
        tasks = json.loads((folder/'post_training/task_results.json').read_text())
        model_results = {}
        for key, (_, expected) in BENCHMARKS.items():
            prediction = folder/f'predictions/post_training/{key}'
            if key == 'tau2_retail':
                prediction = folder/'predictions/post_training_whole/tau2_retail'
            result = json.loads((prediction/'summary.json').read_text())
            task = tasks[key]
            if native_swe and key == 'swe_bench':
                prediction = folder/'predictions/swe_native'
                result = json.loads((prediction/'summary.json').read_text())
                task = json.loads((folder/'post_training/swe_native_task_results.json').read_text())
            assert result['trajectories'] == task['samples'] == expected
            assert result['input_sha256'] == task['input_sha256']
            assert not result.get('segmented_trajectories'), 'Chunked results are not allowed'
            policy = result['context_policy']
            if policy.startswith('Full history when within'):
                # Reuse an earlier completed cohort only if no history was split.
                assert result.get('segmented_trajectories') == []
            else:
                assert policy in {
                    'Whole conversation in one forward pass; no chunking or truncation',
                    'full trajectory, no windows or truncation',
                }
            excluded = result.get('excluded_trajectories', [])
            scored = result.get('scored_trajectories', expected)
            assert scored + len(excluded) == expected
            assert all(x['tokens'] > result['context_limit'] for x in excluded)
            probe_path = (Path('experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz')
                          if sub == 'base_model' else folder/'post_training/probes/layer_12.npz')
            assert result['probe_sha256'] == hashlib.sha256(probe_path.read_bytes()).hexdigest()
            metadata = json.loads(probe_path.with_suffix('.json').read_text())
            assert metadata['model_revision'] == result['model_revision']
            assert metadata['activation'] == 'post_attention_layernorm_output_pre_mlp'
            metric = {'appworld_test':'appworld_success','tau2_retail':'retail_scorer','swe_bench':'swe_bench_scorer'}[key]
            model_results[key] = {'task':task, 'probe':result, 'task_metric':metric}
        all_results[label] = model_results
    if native_swe:
        reference = all_results['Base']['swe_bench']['task']['settings']
        assert all(results['swe_bench']['task']['settings'] == reference for results in all_results.values())
        assert all(results['swe_bench']['probe']['context_limit'] == 131072 for results in all_results.values())
    # Validate every cohort before writing any report.
    for sub, _ in MODELS:
        if sub != 'base_model': publish(root/sub, native_swe=native_swe)
    table = ['### Task performance and role recovery\n',
             'All values are percentages. Role columns are recall within that true role.\n',
             '| Benchmark | Model | Task success | Overall role | System | User | Reasoning | Assistant | Tool |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    appendix = ['## Fresh-trajectory confusions\n',
                'Rows are true roles; columns are predicted roles. Each row sums to 100% '
                '(up to rounding). Only whole histories within native context are scored. '
                'Over-limit histories retain their task outcome but have no role score.\n']
    coverage = ['## Whole-history coverage\n',
                '| Benchmark | Model | Histories scored | Unscorable | Evaluation errors |',
                '|---|---|---:|---:|---:|']
    for key,(benchmark, expected) in BENCHMARKS.items():
        for _, label in MODELS:
            row = all_results[label][key];score=row['probe'];task=row['task']
            task_score = task['metrics'][row['task_metric']]['success_rate']
            values=[task_score,score['accuracy'],*score['per_role_recall']]
            table.append('| '+benchmark+' | '+label+' | '+' | '.join(f'{100*x:.1f}' for x in values)+' |')
            excluded=score.get('excluded_trajectories',[]);scored=score.get('scored_trajectories',expected)
            coverage.append(f'| {benchmark} | {label} | {scored}/{expected} | {len(excluded)} | {task["sample_errors"]} |')
            appendix += [f'\n### {benchmark} — {label}\n',
                         '| True / predicted | '+' | '.join(ROLE_NAMES)+' |',
                         '|---|---:|---:|---:|---:|---:|']
            for name,counts in zip(ROLE_NAMES,score['confusion_counts']):
                total=sum(counts)
                appendix.append('| '+name+' | '+' | '.join(f'{100*x/total:.2f}%' if total else '—' for x in counts)+' |')
    table += ['\n**Evaluation sizes:** AppWorld normal test: 168 tasks; SWE-bench Verified: '
              '500; τ² retail: 114, with one attempt per model per task. Task success includes '
              'all attempts. Role metrics exclude only histories above the native context limit; '
              '[coverage and percentage confusion matrices](APPENDIX.md#whole-history-coverage) are in the appendix.',
              '\nThe base trajectories/task scores are reused from the earlier evaluations; '
              'base probe scores were recomputed on whole histories. Base SWE had 226 API-error '
              'attempts and older generation settings, so its task score is not a controlled '
              'measure of training effects. Base generation also used native MXFP4, versus BF16 '
              'with adapters for trained models. SWE attempts also encountered the 32K serving context ceiling (42/500 AppWorld-trained, 319/500 ASTRA-trained, 133/500 DepthBench-trained); these task results do not measure unconstrained capability. [Failure audit](APPENDIX.md#swe-bench-failure-audit).']
    if native_swe:
        stops = ', '.join(f"{label}: {all_results[label]['swe_bench']['task']['samples_with_model_context_stops']}/500"
                          for _, label in MODELS)
        table[-1] = ('\nBase AppWorld/retail histories and task scores are reused; SWE was regenerated '
                     'for all four models with matched settings, BF16 weights and the native 131K context allowance. '
                     'SWE attempts reaching the model context bound: '+stops+'. All remain in task metrics. '
                     'AppWorld/retail base generation used native MXFP4, versus BF16 with adapters for trained models. '
                     '[SWE evaluation limitations](APPENDIX.md#swe-bench-failure-audit).')
    retail_coverage = []
    for _, label in MODELS:
        score = all_results[label]['tau2_retail']['probe']
        retail_coverage.append(f"{label}: {score.get('scored_trajectories', 114)}/114")
    table.insert(-1, '\n**Retail role-score coverage:** '+', '.join(retail_coverage)+
                 '. Exclusions differ by model, so role scores describe different eligible subsets.')
    if native_swe:
        errors = ', '.join(f"{label}: {all_results[label]['swe_bench']['task']['sample_errors']}/500"
                           for _, label in MODELS)
        table.append('\nSWE evaluation errors, counted as task failures: '+errors+
                     '. These are sandbox runtime errors or scoring/tool timeouts; see the appendix.')
    section(root/'README.md','fresh-rollout-results','\n'.join(table),before='## Message')
    section(root/'APPENDIX.md','fresh-rollout-confusions','\n'.join(appendix))
    section(root/'APPENDIX.md','whole-history-coverage','\n'.join(coverage))
    if native_swe:
        main = root/'README.md'
        text = main.read_text().replace(
            'Base-model trajectories and task scores are reused from the\nexisting evaluations.',
            'Base-model AppWorld/retail trajectories and task scores are reused; SWE is\nregenerated for all four models with matched settings and native 131K context.')
        text = text.replace(
            'Generation settings differ between the reused base run and trained runs, especially\nfor SWE; these results describe observed transfer and behavior, rather than\nisolating every change as a causal effect of training.',
            'AppWorld/retail reuse older base generations, so their comparisons do not isolate\nevery change as a causal effect of training. SWE uses matched generation settings;\nall models are still bounded by native context and the stated task budgets.')
        # Keep repeated reference rows consistent without changing KL experiment results.
        current_rows = {line.split(' | ')[2]: line for line in table
                        if line.startswith('| SWE-bench Verified | ')}
        text = '\n'.join(current_rows.get(line.split(' | ')[2], line)
                         if line.startswith('| SWE-bench Verified | ') else line
                         for line in text.splitlines()) + '\n'
        main.write_text(text)
        for sub in ['astra_sft_1k', 'depthbench']:
            path = root/sub/'README.md'
            path.write_text(path.read_text().replace(
                'SWE task comparisons remain constrained by serving limits and\nthe older base cohort',
                'SWE uses matched base/trained generation settings and native context'))
    (root/'post_training/comparison.json').write_text(json.dumps(all_results,indent=2)+'\n')
    (root/'base_model/post_training/results.json').write_text(json.dumps(all_results['Base'],indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiment_reports/training_against_role_probes'))
    parser.add_argument('--native-swe',action='store_true')
    args=parser.parse_args()
    compare(args.root,args.native_swe)
