"""Fixed trajectory, standard probe and base-completion datasets."""
import json
from pathlib import Path
import numpy as np
import torch
from role_probe_training.probes.harmony import ROLES, tokenize_messages
from role_probe_training.trajectories.replay import encode_record


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def trajectory_examples(tokenizer, path):
    examples=[]
    for row in read_jsonl(path):
        enc=encode_record(tokenizer,row)
        labels=enc['labels'].copy()
        for i,mid in enumerate(enc['message_ids']):
            if mid<0 or enc['source_message_indices'][int(mid)]<0:
                labels[i]=-1
        examples.append(dict(id=row.get('trajectory_id',row['sample_id']),
            ids=enc['input_ids'].tolist(),labels=labels.tolist()))
    if len({x['id'] for x in examples})!=len(examples):
        raise ValueError('Trajectory identifiers must be unique')
    return examples


def probe_examples(tokenizer,path):
    docs=read_jsonl(path)
    if len(docs)!=250:
        raise ValueError('Standard probe training requires the original 250 passages')
    examples=[]
    for doc in docs:
        for role in ROLES:
            enc=tokenize_messages(tokenizer,[{'role':role,'content':doc['text']}])
            examples.append(dict(id=f"{doc['id']}/{role}",ids=enc['input_ids'].tolist(),
                                 labels=enc['labels'].tolist(),split=doc['split']))
    return examples


def kl_examples(path, max_length=2048):
    examples=[]
    for row in read_jsonl(path):
        prefix=row['prompt_ids']; ids=(prefix+row['completion_ids'])[:max_length]
        # Upstream scores completion positions without shifting this mask.
        mask=[i>=len(prefix) for i in range(len(ids))]
        examples.append(dict(id=str(row['id']),ids=ids,completion_mask=mask))
    if not examples or not any(any(x['completion_mask']) for x in examples):
        raise ValueError('KL anchor contains no scored completion tokens')
    return examples


def collate(examples, device, pad_id):
    length=max(len(x['ids']) for x in examples)
    ids=torch.full((len(examples),length),pad_id,device=device,dtype=torch.long)
    labels=torch.full_like(ids,-1); attention=torch.zeros_like(ids)
    completion=torch.zeros_like(ids,dtype=torch.bool)
    for i,x in enumerate(examples):
        n=len(x['ids']); ids[i,:n]=torch.as_tensor(x['ids'],device=device)
        attention[i,:n]=1
        if 'labels' in x: labels[i,:n]=torch.as_tensor(x['labels'],device=device)
        if 'completion_mask' in x: completion[i,:n]=torch.as_tensor(x['completion_mask'],device=device)
    return ids,attention,labels,completion
