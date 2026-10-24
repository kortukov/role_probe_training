"""Replace obsolete chunked baseline summaries with whole-history measurements."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('experiment_reports')
BASE = ROOT/'training_against_role_probes/base_model'
ROLES = ['system','user','cot','assistant','tool']
DATASETS = [('appworld_test','appworld_test_normal','AppWorld normal test',168),
            ('tau2_retail','tau2_retail','τ² retail',114),
            ('swe_bench','swe_bench','SWE-bench Verified',500)]


def main():
    converted={}
    for key,alias,label,n in DATASETS:
        directory=BASE/f'predictions/{"post_training_whole" if key=="tau2_retail" else "post_training"}/{key}'
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['trajectories']==n and 'no chunking' in summary['context_policy']
        rows=[json.loads(x) for x in (directory/'index.jsonl').read_text().splitlines()]
        assert len(rows)==summary['scored_trajectories']
        layers={}
        for layer,score in summary['layers'].items():
            matrix=np.asarray(score['confusion_counts']);support=matrix.sum(1)
            layers[layer]={'trajectories':summary['scored_trajectories'],'available_trajectories':n,
                'scored_tokens':int(matrix.sum()),'accuracy':score['accuracy'],
                'macro_trajectory_accuracy':score['mean_trajectory_accuracy'],
                'majority_role_accuracy':float(support.max()/support.sum()),
                'uniform_five_role_accuracy':0.2,'balanced_role_accuracy':score['macro_recall'],
                'roles':{r:{'tokens':int(support[i]),'recall':score['per_role_recall'][i],
                    'mean_true_role_probability':score['mean_true_role_probability'][i]} for i,r in enumerate(ROLES)},
                'confusion_matrix':matrix.tolist(),'windowed_trajectories':0,
                'excluded_trajectories':summary['excluded_trajectories'],
                'single_window_accuracy':score['accuracy'],
                'trajectory_results':[{'sample_id':x['trajectory_id'],'tokens':int(np.asarray(x['confusion_counts'][layer]).sum()),
                    'accuracy':float(np.trace(x['confusion_counts'][layer])/np.asarray(x['confusion_counts'][layer]).sum()),'windows':1} for x in rows]}
        converted[alias]={'prediction_directory':str(directory),'layers':layers}
    for report,layers in [('probe_transfer',['12']),('probe_layer_comparison',['4','8','12','16','20']),('appworld_probe_transfer',['4','8','12','16','20'])]:
        selected = DATASETS[:1] if report == 'appworld_probe_transfer' else DATASETS
        selected_aliases = {item[1] for item in selected}
        folder=ROOT/'agentic'/report
        path=folder/'metrics.json';previous=json.loads(path.read_text())
        previous['datasets']={name:{'prediction_directory':d['prediction_directory'],'layers':{l:d['layers'][l] for l in layers}} for name,d in converted.items() if name in selected_aliases}
        previous['method']['context_policy']='whole conversation; no chunking or truncation; above 131072 tokens explicitly unscorable'
        previous['method']['replay_precision']='BF16 dequantized original GPT-OSS-20B; flex attention; pre-MLP readout'
        path.write_text(json.dumps(previous,indent=2)+'\n')
        table=['| Dataset | Layer | Overall | System | User | Reasoning | Assistant | Tool |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
        appendix=['# Whole-history confusion matrices\n',
                  'Rows are true content roles; columns are predicted roles. Values are row-normalized percentages. '
                  'Every scored history is processed whole; no chunking or truncation.\n']
        for _,alias,label,n in selected:
            for layer in layers:
                score=converted[alias]['layers'][layer]
                vals=[score['accuracy'],*[score['roles'][r]['recall'] for r in ROLES]]
                table.append(f'| {label} | {layer} | '+' | '.join(f'{x:.2%}' for x in vals)+' |')
                appendix += [f'## {alias} layer {layer}\n',
                    f"Coverage: {score['trajectories']}/{n} histories; {n-score['trajectories']} above native context and unscorable.\n",
                    '| True / predicted | '+' | '.join(ROLES)+' |','|---|---:|---:|---:|---:|---:|']
                for role,row in zip(ROLES,score['confusion_matrix']):
                    total=sum(row);appendix.append('| '+role+' | '+' | '.join(f'{x/total:.2%}' if total else '—' for x in row)+' |')
        (folder/'APPENDIX.md').write_text('\n'.join(appendix)+'\n')
        title='Role-probe transfer on whole agent conversations' if report=='probe_transfer' else 'Comparing role probes across layers on whole conversations'
        text=f'''# {title}

## Motivation & RQ

How well do neutral-text role probes recognize roles in GPT-OSS-20B agent histories?
This evaluation replaces the earlier chunked replay with whole-conversation scoring.

## Evaluation setup

We reuse 168 AppWorld normal-test trajectories, 114 τ² retail trajectories and 500
SWE-bench Verified attempts generated by the unchanged model. Original task scores
and failed attempts are retained. Neutral C4/Dolma-trained linear probes read
post-attention-normalized pre-MLP states; message/channel metadata labels content.
Each scored conversation is processed in one forward pass. The one retail history
above the native 131,072-token limit is unscorable; no text is truncated or chunked.

Overall accuracy weights content tokens equally. Per-role accuracy is recall among
tokens of that true role. No probe fitting or selection uses these benchmarks.

## Factual observations

'''+ '\n'.join(table)+'''

Probe coverage is 168/168 AppWorld, 113/114 retail and 500/500 SWE histories.
[Percentage confusion matrices](APPENDIX.md) give the complete role breakdown.

## Message

Role recognition varies strongly by role and layer; no single aggregate accuracy
establishes reliable recognition of every role. These are base-model measurements,
not evidence about training effects. SWE includes 226 API-error attempts, so its
task score is not a reliable capability estimate. The current
[base-versus-trained comparison](../../training_against_role_probes/README.md)
reports task performance separately from role accuracy.

[Reproduction instructions](RUNNING.md)
'''
        if report == 'appworld_probe_transfer':
            text = text.replace('We reuse 168 AppWorld normal-test trajectories, 114 τ² retail trajectories and 500\nSWE-bench Verified attempts generated by the unchanged model.',
                                'We reuse 168 AppWorld normal-test trajectories generated by the unchanged model.')
            text = text.replace('The one retail history\nabove the native 131,072-token limit is unscorable; no text is truncated or chunked.',
                                'All 168 histories fit the native context; no text is truncated or chunked.')
            text = text.replace('Probe coverage is 168/168 AppWorld, 113/114 retail and 500/500 SWE histories.',
                                'Probe coverage is 168/168 AppWorld histories.')
            text = text.replace('SWE includes 226 API-error attempts, so its\ntask score is not a reliable capability estimate. ', '')
        (folder/'README.md').write_text(text)
        (folder/'RUNNING.md').write_text('''# Reproduction

The original agent trajectories and task scores are reused. Current whole-history
predictions are referenced in [metrics.json](metrics.json), under
`experiment_reports/training_against_role_probes/base_model/predictions/`.
The old chunked viewer/predictions are not the source of these results.

Run `src/experiments/training_against_role_probes/post_training/whole_score.sbatch`
with root `experiment_reports/training_against_role_probes/base_model`, `--base`,
and `--overlength exclude`. Retail output is explicitly selected with `--output`
pointing to `predictions/post_training_whole/tau2_retail`. Read
[the shared reproduction guide](../../training_against_role_probes/RUNNING.md#fresh-rollout-evaluation)
for environments, source paths and job provenance. No new model generations or
probe fits are needed for this baseline rerun.

Regenerate these summaries with:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/post_training/refresh_base_reports.py
```

Inputs retain all recorded reasoning and tool messages in retrospective Harmony;
this is not a reconstruction of each generation-time prompt. The runtime uses
BF16 dequantization and flex attention. All metadata-labeled content tokens in
eligible histories are scored; synthetic tool schemas are excluded.
''')


if __name__=='__main__':main()
