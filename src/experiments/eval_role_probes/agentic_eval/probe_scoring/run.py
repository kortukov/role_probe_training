"""Score all content tokens, capturing multiple pre-MLP probes in one forward per window."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from role_probe_training.probes.harmony import ROLES
from role_probe_training.probes.linear import RoleProbe
from role_probe_training.probes.model import load_model
from role_probe_training.trajectories.replay import encode_record


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probes',type=Path,nargs='+',required=True)
    p.add_argument('--model',default='.cache/models/gpt-oss-20b')
    p.add_argument('--context-window',type=int,default=16384,help='Independent context windows; keep and score every token')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=1)
    a=p.parse_args()
    if not 0 <= a.shard_index < a.shard_count: p.error('invalid shard')
    a.output.mkdir(parents=True,exist_ok=True)
    probes={}
    for path in a.probes:
        probe=RoleProbe.load(path)
        assert probe.metadata['activation']=='post_attention_layernorm_output_pre_mlp'
        assert probe.metadata['model_revision']=='6cee5e81ee83917806bbde320786a8fb61efebee'
        assert probe.metadata['roles']==list(ROLES)
        probes[probe.metadata['layer']]=probe
    manifest={'schema_version':1,'input':str(a.input),'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'probe_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.probes},'probe_metadata':{str(k):v.metadata for k,v in probes.items()},'context_window':a.context_window,'context_policy':'nonoverlapping windows; no token dropping; positions restart at each window; original eager attention, single model forward per window','shard_index':a.shard_index,'shard_count':a.shard_count,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'rendering':'retrospective Harmony; preserves all reasoning; tool schemas added as system context; not exact serving prompts','labels':'content tokens only; special/header/boundary tokens excluded; synthetic tool-schema context separately identified by source_message_index=-1'}
    mp=a.output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())['input_sha256']!=manifest['input_sha256']:
        raise ValueError('Cannot resume on different input')
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    tokenizer,model=load_model(a.model)
    weights={k:torch.as_tensor(v.weight,device=model.device,dtype=torch.float32) for k,v in probes.items()}
    biases={k:torch.as_tensor(v.bias,device=model.device,dtype=torch.float32) for k,v in probes.items()}
    captured={}
    def hook(layer):
        def fn(module,args,output):
            hidden=output[0] if isinstance(output,tuple) else output
            captured[layer]=torch.softmax(hidden[0].float()@weights[layer].T+biases[layer],dim=-1).cpu().numpy()
        return fn
    handles=[model.model.layers[k].post_attention_layernorm.register_forward_hook(hook(k)) for k in probes]
    opener=gzip.open if a.input.suffix=='.gz' else open
    try:
      with opener(a.input,'rt') as source, torch.inference_mode():
        for index,line in enumerate(source):
            if index%a.shard_count!=a.shard_index: continue
            record=json.loads(line)
            destination=a.output/f'{index:06d}.json.gz'
            if destination.exists(): continue
            start=time.monotonic()
            try:
                enc=encode_record(tokenizer,record)
                ids=enc['input_ids']; n=len(ids)
                if not 0 < a.context_window <= model.config.max_position_embeddings:
                    raise ValueError('Context window must fit model')
                chunks={k:[] for k in probes}
                for lo in range(0,n,a.context_window):
                    hi=min(n,lo+a.context_window)
                    result=model.model(input_ids=torch.as_tensor(ids[lo:hi],device=model.device)[None,:], attention_mask=torch.ones((1,hi-lo),device=model.device,dtype=torch.long),use_cache=False)
                    for k in probes: chunks[k].append(captured[k])
                    del result
                probabilities={str(k):np.concatenate(v) for k,v in chunks.items()}
                labels=enc['labels']; valid=labels>=0
                # Separate utterance tokens from synthetic schema context.
                origin=np.asarray([enc['source_message_indices'][i] if i>=0 else -2 for i in enc['message_ids']])
                content_mask=valid & (origin>=0)
                layers={}
                for k,prob in probabilities.items():
                    pred=prob.argmax(-1)
                    matrix=np.zeros((len(ROLES),len(ROLES)),dtype=np.int64)
                    np.add.at(matrix,(labels[content_mask],pred[content_mask]),1)
                    layers[k]={'predictions':pred.tolist(),'probabilities':prob.round(6).tolist(),'confusion_matrix':matrix.tolist(),'correct':int(np.trace(matrix)),'tokens':int(matrix.sum()),'accuracy':float(np.trace(matrix)/matrix.sum()) if matrix.sum() else None}
                out={'schema_version':1,'benchmark':record['benchmark'],'sample_id':record['sample_id'],'source_index':index,'roles':ROLES,'trajectory':record,'rendered_messages':enc['messages'],'source_message_indices':enc['source_message_indices'],'text':enc['text'],'input_ids':ids.tolist(),'offsets':enc['offsets'].tolist(),'labels':labels.tolist(),'message_ids':enc['message_ids'].tolist(),'score_mask':content_mask.tolist(),'layers':layers,'context_window_starts':list(range(0,n,a.context_window)),'seconds':time.monotonic()-start}
                temporary=destination.with_suffix('.tmp')
                with gzip.open(temporary,'wt') as target: json.dump(out,target,ensure_ascii=False,separators=(',',':'))
                temporary.replace(destination)
                print(f'{index}: {record["sample_id"]} {n} tokens {time.monotonic()-start:.1f}s',flush=True)
            except Exception as error:
                with (a.output/'errors.jsonl').open('a') as target: target.write(json.dumps({'index':index,'sample_id':record.get('sample_id'),'error':repr(error)})+'\n')
                raise  # Stop rather than silently dropping difficult/long trajectories.
    finally:
        for handle in handles: handle.remove()

if __name__=='__main__': main()
