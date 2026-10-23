"""Prepare upstream-honesty KL prompts; no Tool-Use training data is used."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from datasets import load_dataset
from huggingface_hub import HfApi


def first_user(ex):
    # Adapted from training_against_probes/src/utils/finetune_utils.py.
    messages = ex.get('messages') or ex.get('conversations') or ex.get('conversation')
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        first = messages[0]
        role = first.get('role') or first.get('from')
        content = first.get('content') or first.get('value')
        if content and (role is None or str(role).lower() in ('user', 'human')):
            return content
    for key in ('prompt', 'instruction', 'question', 'query', 'text', 'input'):
        value = ex.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    source='allenai/Dolci-Instruct-SFT'
    revision=HfApi().dataset_info(source).sha
    dataset=load_dataset(source, split='train', revision=revision, streaming=True)
    prompts=[]
    for row in dataset:
        prompt=first_user(row)
        if prompt:
            prompts.append(prompt)
        if len(prompts)==5000:
            break
    if len(prompts)!=5000:
        raise ValueError('Expected 5000 valid source prompts')
    random.Random(42).shuffle(prompts)
    a.output.mkdir(parents=True, exist_ok=True)
    path=a.output/'prompts.json'
    path.write_text(json.dumps(prompts[:800],ensure_ascii=False)+'\n')
    (a.output/'source.json').write_text(json.dumps(dict(dataset=source,revision=revision,
        split='train',pool=5000,selected=800,seed=42,prompts_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        method='First 5000 valid first-user prompts; Python shuffle seed42; retain800; upstream honesty selection'),indent=2)+'\n')

if __name__=='__main__': main()
