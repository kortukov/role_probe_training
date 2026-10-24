"""Fit fresh standard-data probes and score fresh agent trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from role_probe_training.probes.harmony import ROLES
from role_probe_training.probes.linear import RoleProbe, fit_probe
from role_probe_training.probes.model import extract_states
from role_probe_training.probes.evaluation import metrics
from role_probe_training.training.data import probe_examples, trajectory_examples


def load(path):
    torch._dynamo.config.recompile_limit=64
    torch._dynamo.config.accumulated_recompile_limit=512
    torch._dynamo.config.fail_on_recompile_limit_hit=True
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='flex_attention')
    PeftModel.from_pretrained(model,Path(path)/'attention_adapter',is_trainable=False,autocast_adapter_dtype=True)
    model.requires_grad_(False)
    model.train()  # zero dropout; use memory-efficient routed experts
    return tokenizer,model


def fit(root):
    model_path=root/'data/post_training/model'; tokenizer,model=load(model_path)
    documents=Path('experiment_reports/role_probes/gptoss20b_baseline/artifacts/inputs/documents.jsonl')
    examples=probe_examples(tokenizer,documents)
    features=[]; labels=[]; splits=[]
    for i,example in enumerate(examples):
        y=np.asarray(example['labels']); positions=np.flatnonzero(y>=0)
        features.append(extract_states(model,example['ids'],[12],positions)[12].astype(np.float16))
        labels.append(y[positions]); splits.extend([example['split']]*len(positions))
        if (i+1)%100==0:print(f'Extracted {i+1}/{len(examples)} standard passage-role examples',flush=True)
    x=np.concatenate(features); del features
    y=np.concatenate(labels); mask=np.asarray(splits)=='train'
    export=json.loads((model_path/'export.json').read_text())
    probe=fit_probe(x[mask],y[mask],C=0.005,max_iter=5000,backend='cuml',metadata={
        'model_revision':export['adapter_sha256'],'model_export':str(model_path),
        'base_model_revision':export['base_revision'],'layer':12,'layer_indexing':'zero_based',
        'activation':'post_attention_layernorm_output_pre_mlp','fresh_fit':True,
        'initialization':'from scratch; no training-loop probe weights',
        'documents_sha256':hashlib.sha256(documents.read_bytes()).hexdigest(),
        'passages':250,'train_tokens':int(mask.sum()),'test_tokens':int((~mask).sum()),
        'token_stride':1,'train_split':'original passage split','sequence_length':1024})
    output=root/'post_training'; probe.save(output/'probes/layer_12.npz')
    result=metrics(y[~mask],probe.predict_proba(x[~mask]))
    (output/'probe_fit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'heldout_standard_accuracy':result['accuracy']}),flush=True)


def score(root,input_path,output):
    model_path=root/'data/post_training/model';tokenizer,model=load(model_path)
    probe=RoleProbe.load(root/'post_training/probes/layer_12.npz')
    export=json.loads((model_path/'export.json').read_text())
    assert probe.metadata['model_revision']==export['adapter_sha256']
    examples=trajectory_examples(tokenizer,input_path)
    output.mkdir(parents=True,exist_ok=True)
    matrix=np.zeros((5,5),dtype=np.int64); per_trace=[]
    with (output/'index.jsonl').open('w') as index:
        for i,x in enumerate(examples):
            if len(x['ids'])>model.config.max_position_embeddings:raise ValueError('Trajectory exceeds full context')
            states=extract_states(model,x['ids'],[12])[12]
            probabilities=probe.predict_proba(states)
            pred=probabilities.argmax(-1).astype(np.int8);labels=np.asarray(x['labels'],dtype=np.int8)
            valid=labels>=0; counts=np.zeros((5,5),dtype=np.int64)
            np.add.at(counts,(labels[valid],pred[valid]),1);matrix+=counts
            per_trace.append(float(counts.trace()/counts.sum()))
            name=f'{i:06d}.npz'
            np.savez_compressed(output/name,input_ids=np.asarray(x['ids'],dtype=np.int32),labels=labels,
                predictions=pred,probabilities=probabilities.astype(np.float16))
            index.write(json.dumps({'trajectory_id':x['id'],'file':name,'confusion_counts':counts.tolist()})+'\n');index.flush()
            if (i+1)%10==0:print(f'Scored {i+1}/{len(examples)} trajectories',flush=True)
    support=matrix.sum(1); recalls=np.divide(matrix.diagonal(),support,out=np.zeros(5),where=support>0)
    result={'trajectories':len(examples),'roles':list(ROLES),'accuracy':float(matrix.trace()/matrix.sum()),
        'macro_recall':float(recalls[support>0].mean()),'per_role_recall':recalls.tolist(),
        'mean_trajectory_accuracy':float(np.mean(per_trace)),'confusion_counts':matrix.tolist(),
        'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'probe_sha256':hashlib.sha256((root/'post_training/probes/layer_12.npz').read_bytes()).hexdigest(),
        'model_revision':export['adapter_sha256'],'context_policy':'full trajectory, no windows or truncation'}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['fit','score'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.stage=='fit':fit(a.root)
    else:
        if not a.input or not a.output:p.error('Scoring requires input and output')
        score(a.root,a.input,a.output)

if __name__=='__main__':main()
