"""Update only the role-results section of the authored AgentDojo report."""
import json
from pathlib import Path

ROOT=Path('experiment_reports/prompt_injection/agentdojo_same_batch_27000')
NAMES=['System','User','Reasoning','Assistant','Tool']
START='<!-- agentdojo-role-results -->'
END='<!-- /agentdojo-role-results -->'


def percent(x):
    return '—' if x is None else f'{100*x:.1f}'


def main():
    table=['### Role perception','',
        'Values are percentages. Overall accuracy weights content tokens; role columns are recall within the true role. Reasoning was not saved and has no scored support. Each model uses its own freshly fitted standard-data layer-12 probe.', '',
        '| Subset | Model | Overall role | System | User | Reasoning | Assistant | Tool |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    appendix=['# AgentDojo role-perception appendix','',
        'Rows are true roles and columns predicted roles. Percentages are normalized within each true role; raw counts follow each percentage matrix. A zero-support row is unavailable, not zero accuracy. Ground truth follows message/channel metadata; attack text never changes its enclosing message role.','',
        'Every saved conversation is replayed whole using retrospective Harmony normalization, including metadata-derived tool identities and unscored tool schemas. This is not byte-identical serving-prompt recovery. AgentDojo discarded reasoning-channel text, so all five classes remain possible predictions but the Reasoning row has no support.','',
        'The expected denominators retain 95 clean and 921 attacked attempts per model. Available partial histories from context-limit errors are included, missing histories are unscorable, and histories above native 131,072-token context are excluded without truncation. Task-success denominators are unchanged.','',
        '| Subset | Model | Scored / assigned | Missing history | Above native context | Saved partial histories |',
        '|---|---|---:|---:|---:|---:|']
    blocks=[];results={}
    for model,label in [('base','Base'),('trained','Same-batch KL (27,000)')]:
        results[model]={}
        for subset in ('clean','attacked'):
            path=ROOT/'role_perception'/model/f'{subset}.json'
            d=json.loads(path.read_text()) if path.exists() else None
            if d is None or not d.get('complete'):
                table.append(f'| {subset.title()} | {label} | Pending | — | — | — | — | — |')
                appendix.append(f'| {subset.title()} | {label} | Pending | — | — | — |')
                continue
            assert d['roles']==['system','user','cot','assistant','tool']
            results[model][subset]=d
            values=[d['accuracy'],*d['per_role_recall']]
            table.append('| '+ ' | '.join([subset.title(),label,*map(percent,values)])+' |')
            appendix.append(f"| {subset.title()} | {label} | {d['scored_trajectories']}/{d['expected_trajectories']} | {len(d['missing_trajectories'])} | {len(d['excluded_trajectories'])} | {len(d['partial_saved_histories'])} |")
            b=[f'## {label}: {subset}','',f"Overall token accuracy: **{percent(d['accuracy'])}%**. Scored content tokens: {d['content_tokens']:,}.",'',
               '| True / predicted | '+' | '.join(NAMES)+' |','|---|'+'---:|'*5]
            for i,row in enumerate(d['confusion_counts']):
                total=sum(row);b.append('| '+NAMES[i]+' | '+' | '.join(f'{100*x/total:.2f}%' if total else '—' for x in row)+' |')
            b+=['','Raw token counts:','', '| True / predicted | '+' | '.join(NAMES)+' |','|---|'+'---:|'*5]
            b+=['| '+NAMES[i]+' | '+' | '.join(map(str,row))+' |' for i,row in enumerate(d['confusion_counts'])]
            b+=['',f"[Metrics and provenance](role_perception/{model}/{subset}.json) · [Fresh probe metadata](role_perception/{model}/post_training/probes/layer_12.json)",'']
            blocks+=b
    table+=['','[Coverage, raw counts and confusion matrices](APPENDIX.md). Missing histories and native-context exclusions are listed in the linked metrics.','']
    if all(len(v)==2 for v in results.values()):
        for subset in ('clean','attacked'):
            delta=100*(results['trained'][subset]['accuracy']-results['base'][subset]['accuracy'])
            table.append(f"On {subset} saved histories, trained overall role accuracy differs from base by {delta:+.1f} percentage points. Each model is scored on its own histories; this is not a paired comparison of identical text.")
        table+=['','These role-label measurements are distinct from task utility and attack success; role recovery alone does not establish robustness.','']
    text=START+'\n'+'\n'.join(table)+END
    path=ROOT/'README.md';readme=path.read_text()
    if START in readme:
        first,rest=readme.split(START,1);_,last=rest.split(END,1);readme=first+text+last
    else:readme=readme.replace('## Message',text+'\n\n## Message')
    path.write_text(readme)
    appendix+=['','## Probe fitting','',
        'Two independent five-class multinomial linear probes are fitted from scratch on the original 250 C4/Dolma passages with their original train/test split, up to 1,024 passage tokens and every content token. Fits use the post-attention-layernorm/pre-MLP site at zero-based layer 12, C=0.005 and cuML max_iter=5,000. The role order, exact model/data hashes, token counts, stopping diagnostics and fit test accuracy are saved with each probe. Iteration count alone does not establish convergence; inspect job logs for cuML line-search warnings.','',*blocks]
    (ROOT/'APPENDIX.md').write_text('\n'.join(appendix)+'\n')
    (ROOT/'role_metrics.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
