"""Full-context probe evaluation, with reusable per-token arrays and source IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from role_probe_training.probes.harmony import ROLES
from role_probe_training.probes.linear import RoleProbe
from .data import read_jsonl,trajectory_examples
from .model import RoleModel


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=1)
    a=p.parse_args()
    paths={12:'experiment_reports/role_probes/gptoss20b_baseline/probes/layer_12.npz',
           16:'experiment_reports/role_probes/gptoss20b_baseline/probes/layer_16.npz',
           **{i:f'experiment_reports/role_probes/gptoss20b_layers/probes/layer_{i}.npz' for i in (4,8,20)}}
    runtime=RoleModel('.cache/models/gpt-oss-20b',paths[12],0,adapters=False)
    examples=trajectory_examples(runtime.tokenizer,a.input)
    a.output.mkdir(parents=True,exist_ok=True)
    metadata=dict(input=str(a.input),input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        roles=list(ROLES),layers=sorted(paths),context_policy='full trajectory; no truncation or independent windows',
        activation='zero-based post_attention_layernorm output, pre-MLP',
        shard_index=a.shard_index,shard_count=a.shard_count,
        array_schema='input_ids[int32,N], labels[int8,N], layers[int8,K], predictions[int8,K,N], probabilities[float16,K,N,5]',
        probe_sha256={str(k):hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items()})
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    outputs={}; handles=[]
    for layer,path in paths.items():
        artifact=RoleProbe.load(path)
        assert artifact.metadata['roles']==list(ROLES)
        assert artifact.metadata['layer']==layer
        assert artifact.metadata['activation']=='post_attention_layernorm_output_pre_mlp'
        weight=torch.as_tensor(artifact.weight,device=0,dtype=torch.float32)
        bias=torch.as_tensor(artifact.bias,device=0,dtype=torch.float32)
        def hook(module,args,hidden,layer=layer,weight=weight,bias=bias):
            outputs[layer]=torch.softmax(hidden[0].float()@weight.T+bias,dim=-1).cpu().numpy()
        handles.append(runtime.base.model.layers[layer].post_attention_layernorm.register_forward_hook(hook))
    matrices={k:np.zeros((5,5),dtype=np.int64) for k in paths}; scored=0
    try:
        with torch.no_grad(),(a.output/'index.jsonl').open('w') as index_file:
            for index,example in enumerate(examples):
                if index%a.shard_count!=a.shard_index: continue
                destination=a.output/f'{index:06d}.npz'
                runtime.states([example])
                layers=sorted(paths); labels=np.asarray(example['labels'],dtype=np.int8)
                valid=labels>=0; probabilities=np.stack([outputs[k] for k in layers])
                predictions=probabilities.argmax(-1).astype(np.int8)
                counts={}
                for i,k in enumerate(layers):
                    matrix=np.zeros((5,5),dtype=np.int64)
                    np.add.at(matrix,(labels[valid],predictions[i,valid]),1)
                    matrices[k]+=matrix; counts[str(k)]=matrix.tolist()
                with destination.open('wb') as f:
                    np.savez_compressed(f,input_ids=np.asarray(example['ids'],dtype=np.int32),labels=labels,
                        layers=np.asarray(layers,dtype=np.int8),predictions=predictions,
                        probabilities=probabilities.astype(np.float16))
                index_file.write(json.dumps(dict(source_index=index,trajectory_id=example['id'],
                    file=destination.name,confusion_counts=counts))+'\n'); index_file.flush()
                scored+=1
                if scored%10==0: print(f'Scored {scored} trajectories',flush=True)
        (a.output/'summary.json').write_text(json.dumps(dict(trajectories=scored,
            confusion_counts={str(k):v.tolist() for k,v in matrices.items()}),indent=2)+'\n')
    finally:
        for h in handles: h.remove()
        runtime.close()

if __name__=='__main__':main()
