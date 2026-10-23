"""Aggregate complete baseline predictions and update the training report."""
import argparse
import json
from pathlib import Path
import numpy as np
from role_probe_training.probes.harmony import ROLES


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,default=Path('experiment_reports/training_against_role_probes'))
    a=p.parse_args(); corpus=json.loads((a.root/'train_corpus.json').read_text())
    layers=(4,8,12,16,20); matrices={k:np.zeros((5,5),dtype=np.int64) for k in layers}
    per_trajectory={k:[] for k in layers}; seen=set()
    for shard in range(2):
        directory=a.root/f'predictions/train2700/shard{shard}'
        manifest=json.loads((directory/'manifest.json').read_text())
        assert manifest['input_sha256']==corpus['sha256']
        summary=json.loads((directory/'summary.json').read_text())
        rows=[json.loads(line) for line in (directory/'index.jsonl').read_text().splitlines()]
        assert len(rows)==summary['trajectories']
        for row in rows:
            assert row['trajectory_id'] not in seen
            assert (directory/row['file']).is_file()
            seen.add(row['trajectory_id'])
            for layer in layers:
                matrix=np.asarray(row['confusion_counts'][str(layer)])
                matrices[layer]+=matrix
                per_trajectory[layer].append(float(matrix.trace()/matrix.sum()))
    assert len(seen)==2700==corpus['trajectories']
    result={'trajectories':2700,'source_sha256':corpus['sha256'],'roles':list(ROLES),'layers':{}}
    table='| Layer | Overall | System | User | Reasoning | Assistant | Tool |\n|---|---:|---:|---:|---:|---:|---:|\n'
    appendix='\n## Training-corpus baseline confusions\n\nRows are ground-truth content roles; columns are predicted roles. Counts include\nall metadata-labeled content tokens from the 2,700 full trajectories.\n'
    for layer in layers:
        matrix=matrices[layer]; support=matrix.sum(1)
        recall=np.divide(matrix.diagonal(),support,out=np.zeros(5),where=support>0)
        accuracy=float(matrix.trace()/matrix.sum())
        result['layers'][str(layer)]=dict(tokens=int(matrix.sum()),accuracy=accuracy,
            macro_recall=float(recall[support>0].mean()),mean_trajectory_accuracy=float(np.mean(per_trajectory[layer])),
            per_role_accuracy=dict(zip(ROLES,recall.tolist())),confusion_counts=matrix.tolist())
        table+=f'| {layer} | '+' | '.join(f'{x:.1%}' for x in [accuracy,*recall])+' |\n'
        appendix+=f'\n### Layer {layer}\n\n| True / predicted | '+' | '.join(ROLES)+' |\n|---|'+'---:|'*5+'\n'
        for role,row in zip(ROLES,matrix): appendix+='| '+role+' | '+' | '.join(str(int(v)) for v in row)+' |\n'
        appendix+='\nRow-normalized percentages:\n\n| True / predicted | '+' | '.join(ROLES)+' |\n|---|'+'---:|'*5+'\n'
        for role,row,count in zip(ROLES,matrix,support):
            appendix+='| '+role+' | '+' | '.join(f'{(v/count if count else 0):.1%}' for v in row)+' |\n'
    (a.root/'baseline_probe_results.json').write_text(json.dumps(result,indent=2)+'\n')
    begin='<!-- corpus-results -->'; end='<!-- /corpus-results -->'
    block=begin+'\n### Training-corpus baseline\n\nThe 2,700 training trajectories are complete. The table evaluates the original\nstandard-data probes on the unchanged base model, using full trajectory history.\nOverall accuracy weights content tokens equally; each role column is recall\namong that ground-truth role’s tokens. This is training-corpus characterization,\nnot held-out generalization.\n\n'+table+'\n[Confusion matrices](APPENDIX.md#training-corpus-baseline-confusions) are in the appendix.\n'+end+'\n\n'
    path=a.root/'README.md'; text=path.read_text()
    if begin in text: text=text[:text.index(begin)]+block+text[text.index(end)+len(end):].lstrip('\n')
    else: text=text.replace('## Message',block+'## Message')
    text=text.replace('is being collected; production training has not started.','is complete; production training has not started.')
    path.write_text(text)
    path=a.root/'APPENDIX.md'; text=path.read_text(); marker='\n## Training-corpus baseline confusions'
    if marker in text: text=text[:text.index(marker)]
    path.write_text(text+appendix)
    print(json.dumps({'trajectories':2700,'layer12_accuracy':result['layers']['12']['accuracy']}))

if __name__=='__main__':main()
