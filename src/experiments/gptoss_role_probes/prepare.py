"""Prepare pinned public-data samples for the upstream training recipe. No model generation or fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    documents, seen = [], set()
    for source_index, source in enumerate(config['sources']):
        target = (int(config['n_documents'] * source['fraction'])
                  if source_index < len(config['sources']) - 1
                  else config['n_documents'] - len(documents))
        stream = load_dataset(source['id'], source['name'], split=source['split'],
                              revision=source['revision'], streaming=True, token=False)
        stream = stream.shuffle(seed=config['seed'], buffer_size=config['shuffle_buffer'])
        added = 0
        for index, row in enumerate(stream):
            ids = tokenizer(row['text'], add_special_tokens=False, truncation=True,
                            max_length=config['sequence_length'])['input_ids']
            if not ids:
                continue
            text = tokenizer.decode(ids)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen or any(t in text for t in tokenizer.all_special_tokens):
                continue
            seen.add(digest)
            documents.append({'id': digest, 'source': source['id'], 'stream_index': index,
                              'text': text, 'n_tokens': len(ids)})
            added += 1
            if added == target:
                break
        if added != target:
            raise RuntimeError(f'Insufficient records from {source["id"]}')
        print(f'Prepared {added} documents from {source["id"]}', flush=True)
    rng = np.random.default_rng(config['seed'])
    # Source-stratified, document-grouped split: every role variant stays together.
    for source in config['sources']:
        indices = [i for i,d in enumerate(documents) if d['source'] == source['id']]
        held = set(rng.permutation(indices)[:max(1, round(len(indices)*config['test_fraction']))])
        for i in indices:
            documents[i]['split'] = 'test' if i in held else 'train'
    write_jsonl(output / 'documents.jsonl', documents)
    source = config['conversation_source']
    ds = load_dataset(source['id'], source['name'], split=source['split'],
                      revision=source['revision'], token=False).shuffle(seed=config['seed'])
    prompts = []
    for row in ds:
        # Benign conversations only; no injection experiments in this baseline.
        if row['toxicity'] or row['jailbreaking'] or not 100 <= len(row['user_input']) <= 500:
            continue
        prompts.append({'id': str(row['conv_id']), 'content': row['user_input']})
        if len(prompts) == config['n_conversations']:
            break
    if len(prompts) != config['n_conversations']:
        raise RuntimeError('Insufficient evaluation prompts')
    write_jsonl(output / 'prompts.jsonl', prompts)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Prepared {len(prompts)} benign conversational prompts', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/experiments/gptoss_role_probes/config.json')
    parser.add_argument('--output', default='experiment_reports/role_probes/gptoss20b_baseline/artifacts/inputs')
    args = parser.parse_args()
    prepare(json.loads(Path(args.config).read_text()), Path(args.output))
