"""Fresh standard-data probes and whole-history AgentDojo role evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from role_probe_training.probes.harmony import ROLES
from role_probe_training.probes.linear import RoleProbe

BASE_REVISION='6cee5e81ee83917806bbde320786a8fb61efebee'
SITE='post_attention_layernorm_output_pre_mlp'
DOCUMENTS=Path('experiment_reports/gptoss_role_probes/gptoss20b_baseline/artifacts/inputs/documents.jsonl')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_path(kind):
    root=Path('experiment_reports/training_against_role_probes')
    return root/('base_model/data/swe_native/model' if kind=='base' else 'data_iteration/same_batch_27000/data/post_training/model')


def revision(kind):
    return BASE_REVISION if kind=='base' else json.loads((model_path(kind)/'export.json').read_text())['adapter_sha256']


def load(kind):
    import torch
    if kind=='trained':
        from role_probe_training.post_training.probes import load as trained_load
        return trained_load(model_path(kind))
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch._dynamo.config.recompile_limit=64
    torch._dynamo.config.accumulated_recompile_limit=512
    torch._dynamo.config.fail_on_recompile_limit_hit=True
    path=model_path(kind)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='flex_attention')
    model.requires_grad_(False);model.train()  # zero dropout; routed experts
    return tokenizer,model


def extract(root,kind):
    from role_probe_training.training.data import probe_examples
    from role_probe_training.probes.model import extract_states
    tokenizer,model=load(kind)
    examples=probe_examples(tokenizer,DOCUMENTS)
    cache=root/'role_perception'/kind/'data/post_training/probe_features'
    cache.mkdir(parents=True,exist_ok=True)
    features=[];labels=[];splits=[]
    for i,x in enumerate(examples):
        y=np.asarray(x['labels']);positions=np.flatnonzero(y>=0)
        features.append(extract_states(model,x['ids'],[12],positions)[12].astype(np.float16))
        labels.append(y[positions]);splits.extend([x['split']]*len(positions))
        if (i+1)%100==0:print(f'Extracted {i+1}/{len(examples)} passage-role examples',flush=True)
    states=np.concatenate(features);y=np.concatenate(labels);train=np.asarray(splits)=='train'
    np.save(cache/'states.npy',states);np.save(cache/'labels.npy',y);np.save(cache/'train.npy',train)
    metadata={'model_revision':revision(kind),'base_model_revision':BASE_REVISION,'model_export':str(model_path(kind)),
        'layer':12,'layer_indexing':'zero_based','activation':SITE,'fresh_fit':True,
        'initialization':'from scratch; no training-loop or previous evaluation probe weights',
        'documents_sha256':digest(DOCUMENTS),'passages':250,'train_tokens':int(train.sum()),
        'test_tokens':int((~train).sum()),'token_stride':1,'train_split':'original passage split',
        'sequence_length':1024,'precision':'BF16 model; trained attention adapter in FP32',
        'feature_cache_dtype':'float16'}
    (cache/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


def metrics(matrix):
    support=matrix.sum(1);total=int(matrix.sum())
    recalls=[float(matrix[i,i]/support[i]) if support[i] else None for i in range(5)]
    return {'accuracy':float(matrix.trace()/total) if total else None,
        'per_role_recall':recalls,'support':support.tolist(),'content_tokens':total,
        'confusion_counts':matrix.tolist()}


def score(root,kind,limit_count=None):
    from role_probe_training.training.data import trajectory_examples
    from role_probe_training.probes.model import extract_states
    tokenizer,model=load(kind)
    probe_path=root/'role_perception'/kind/'post_training/probes/layer_12.npz'
    probe=RoleProbe.load(probe_path)
    assert probe.metadata['model_revision']==revision(kind)
    assert probe.metadata['activation']==SITE and probe.metadata['layer']==12
    assert probe.metadata['roles']==list(ROLES)
    inputs=json.loads((root/'role_inputs.json').read_text())['models'][kind]
    for subset in ('clean','attacked'):
        info=inputs[subset];input_path=Path(info['input'])
        assert digest(input_path)==info['input_sha256']
        examples=trajectory_examples(tokenizer,input_path)
        if limit_count is not None:examples=examples[:limit_count]
        output=root/'predictions/role_perception'/kind/subset
        output.mkdir(parents=True,exist_ok=True)
        fingerprint={'probe_sha256':digest(probe_path),'input_sha256':digest(input_path),
            'model_revision':revision(kind),'activation':SITE,'layer':12}
        meta=output/'provenance.json'
        if meta.exists():assert json.loads(meta.read_text())==fingerprint,'Refusing stale predictions'
        else:meta.write_text(json.dumps(fingerprint,indent=2)+'\n')
        matrix=np.zeros((5,5),dtype=np.int64);excluded=[];per_trace=[];scored=0
        for i,x in enumerate(examples):
            path=output/f'{i:06d}.json'
            if len(x['ids'])>model.config.max_position_embeddings:
                excluded.append({'trajectory_id':x['id'],'tokens':len(x['ids']),'reason':'exceeds_native_context'})
                continue
            if path.exists():
                row=json.loads(path.read_text());assert row['trajectory_id']==x['id']
                counts=np.asarray(row['confusion_counts'],dtype=np.int64)
            else:
                labels=np.asarray(x['labels'],dtype=np.int8);positions=np.flatnonzero(labels>=0)
                if not len(positions):raise ValueError(f'No content labels: {x["id"]}')
                states=extract_states(model,x['ids'],[12],positions)[12]
                pred=probe.predict_proba(states).argmax(-1).astype(np.int8)
                counts=np.zeros((5,5),dtype=np.int64)
                np.add.at(counts,(labels[positions],pred),1)
                np.savez_compressed(output/f'{i:06d}.npz',positions=positions.astype(np.int32),
                    labels=labels[positions],predictions=pred)
                path.write_text(json.dumps({'trajectory_id':x['id'],'tokens':len(x['ids']),
                    'confusion_counts':counts.tolist()})+'\n')
            matrix+=counts;per_trace.append(float(counts.trace()/counts.sum()));scored+=1
            if (i+1)%20==0:print(f'{kind}/{subset}: {i+1}/{len(examples)} whole histories',flush=True)
        result={**metrics(matrix),**fingerprint,'roles':list(ROLES),'expected_trajectories':info['expected'],
            'saved_trajectories':info['saved'],'scored_trajectories':scored,'excluded_trajectories':excluded,
            'missing_trajectories':info['missing'],'partial_saved_histories':info['partial_histories'],
            'context_limit':model.config.max_position_embeddings,'context_policy':'whole saved history; no chunking or truncation',
            'reasoning_policy':'not recorded by AgentDojo; zero-support recall is null',
            'mean_trajectory_accuracy':float(np.mean(per_trace)) if per_trace else None,
            'complete':limit_count is None}
        target=root/'role_perception'/kind/f'{subset}.json'
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('accuracy','per_role_recall','scored_trajectories')}),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['extract','score'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--model',choices=['base','trained'],required=True)
    parser.add_argument('--limit-count',type=int)
    a=parser.parse_args()
    if a.stage=='extract':extract(a.root,a.model)
    else:score(a.root,a.model,a.limit_count)

if __name__=='__main__':main()
