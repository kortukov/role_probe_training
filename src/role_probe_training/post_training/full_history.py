"""Score whole saved conversations, never truncating or resetting their context."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from role_probe_training.probes.harmony import ROLES
from role_probe_training.probes.linear import RoleProbe
from role_probe_training.probes.model import extract_states
from role_probe_training.training.data import trajectory_examples
from role_probe_training.training.model import RoleModel
from role_probe_training.post_training.probes import load


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(root, input_path, output, base=False, overlength='error'):
    if base:
        paths = {i: Path(f'experiment_reports/role_probes/{"gptoss20b_baseline" if i in (12,16) else "gptoss20b_layers"}/probes/layer_{i}.npz') for i in (4,8,12,16,20)}
        runtime = RoleModel('.cache/models/gpt-oss-20b', str(paths[12]), 0, adapters=False)
        tokenizer, model = runtime.tokenizer, runtime.base
        runtime.close()
        revision = '6cee5e81ee83917806bbde320786a8fb61efebee'
    else:
        model_path = root / 'data/post_training/model'
        tokenizer, model = load(model_path)
        paths = {12: root / 'post_training/probes/layer_12.npz'}
        revision = json.loads((model_path / 'export.json').read_text())['adapter_sha256']
    probes = {i: RoleProbe.load(p) for i,p in paths.items()}
    assert all(p.metadata['model_revision'] == revision for p in probes.values())
    examples = trajectory_examples(tokenizer, input_path)
    limit = model.config.max_position_embeddings
    excluded = [{'trajectory_id':x['id'], 'tokens':len(x['ids']), 'reason':'exceeds_native_context'}
                for x in examples if len(x['ids']) > limit]
    if excluded and overlength == 'error':
        raise ValueError(f'{len(excluded)} histories exceed {limit} tokens; explicit exclusion policy required')
    output.mkdir(parents=True, exist_ok=True)
    matrices = {i: np.zeros((5,5),dtype=np.int64) for i in probes}
    probability_sums = {i: np.zeros(5) for i in probes}
    per_trace = {i: [] for i in probes}
    scored = 0
    with (output / 'index.jsonl').open('w') as index:
        for n,x in enumerate(examples):
            if len(x['ids']) > limit:
                continue
            labels = np.asarray(x['labels'],dtype=np.int8)
            valid = labels >= 0
            if not valid.any():
                raise ValueError(f'No labeled content: {x["id"]}')
            states = extract_states(model, x['ids'], list(probes))
            saved = {'input_ids':np.asarray(x['ids'],dtype=np.int32), 'labels':labels,
                     'layers':np.asarray(list(probes),dtype=np.int16)}
            counts_by_layer = {}
            for layer, probe in probes.items():
                probabilities = probe.predict_proba(states.pop(layer))
                predictions = probabilities.argmax(-1).astype(np.int8)
                counts = np.zeros((5,5),dtype=np.int64)
                np.add.at(counts,(labels[valid],predictions[valid]),1)
                matrices[layer] += counts
                per_trace[layer].append(float(counts.trace()/counts.sum()))
                for role in range(5):
                    probability_sums[layer][role] += probabilities[labels == role,role].sum(dtype=np.float64)
                counts_by_layer[str(layer)] = counts.tolist()
                saved[f'probabilities_{layer}'] = probabilities.astype(np.float16)
                saved[f'predictions_{layer}'] = predictions
            name = f'{n:06d}.npz'
            np.savez_compressed(output / name, **saved)
            index.write(json.dumps({'trajectory_id':x['id'], 'file':name,
                                    'confusion_counts':counts_by_layer})+'\n')
            index.flush()
            scored += 1
            if scored % 10 == 0:
                print(f'Scored {scored}/{len(examples)-len(excluded)} whole histories', flush=True)
    layers = {}
    for layer,matrix in matrices.items():
        support = matrix.sum(1)
        recalls = np.divide(matrix.diagonal(),support,out=np.zeros(5),where=support>0)
        layers[str(layer)] = {
            'accuracy':float(matrix.trace()/matrix.sum()),
            'macro_recall':float(recalls[support>0].mean()),
            'per_role_recall':recalls.tolist(),
            'mean_true_role_probability':np.divide(probability_sums[layer],support,out=np.zeros(5),where=support>0).tolist(),
            'mean_trajectory_accuracy':float(np.mean(per_trace[layer])),
            'confusion_counts':matrix.tolist(), 'probe_sha256':digest(paths[layer])}
    result = {**layers['12'], 'layers':layers, 'roles':list(ROLES),
              'trajectories':len(examples), 'scored_trajectories':scored,
              'excluded_trajectories':excluded, 'context_limit':limit,
              'input_sha256':digest(input_path), 'model_revision':revision,
              'context_policy':'Whole conversation in one forward pass; no chunking or truncation',
              'overlength_policy':overlength}
    (output / 'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('layers','excluded_trajectories')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',action='store_true')
    parser.add_argument('--overlength',choices=['error','exclude'],default='error')
    args=parser.parse_args()
    score(args.root,args.input,args.output,args.base,args.overlength)
