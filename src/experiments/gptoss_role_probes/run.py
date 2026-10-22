"""Train reusable probes and evaluate a small clean conversational baseline."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from role_probe_training.probes.harmony import ROLES, SYSTEM_PROMPT, parse_generated, render, tokenize_messages
from role_probe_training.probes.model import extract_states, load_model
from role_probe_training.probes.linear import RoleProbe, fit_probe
from role_probe_training.probes.evaluation import metrics, conversation_summary, plot_summary


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def train(config, tokenizer, model, documents, output, metadata):
    if config['token_stride'] != 1:
        raise ValueError('Reproduction requires all content tokens (token_stride=1)')
    layers = config['layers']
    features = {layer: [] for layer in layers}
    labels, splits = [], []
    for index, document in enumerate(documents):
        for role in ROLES:
            enc = tokenize_messages(tokenizer, [{'role': role, 'content': document['text']}])
            positions = np.flatnonzero(enc['labels'] >= 0)
            states = extract_states(model, enc['input_ids'], layers, positions)
            for layer in layers:
                features[layer].append(states[layer].astype(np.float16))
            labels.extend(enc['labels'][positions])
            splits.extend([document['split']] * len(positions))
        if (index+1) % 10 == 0:
            print(f'Extracted {index+1}/{len(documents)} training documents', flush=True)
    labels = np.asarray(labels)
    mask = np.asarray(splits) == 'train'
    results = {}
    for layer in layers:
        x = np.concatenate(features.pop(layer))
        print(f'Fitting block {layer}: {mask.sum()} train / {(~mask).sum()} test tokens', flush=True)
        probe = fit_probe(x[mask], labels[mask], C=config['C'], metadata={
            **metadata, 'layer': layer, 'activation': 'post_attention_layernorm_output_pre_mlp',
            'layer_indexing': 'zero_based', 'hidden_size': x.shape[1],
            'training_config': config,
        })
        path = output / 'probes' / f'layer_{layer}.npz'
        probe.save(path)
        loaded = RoleProbe.load(path)
        np.testing.assert_allclose(loaded.predict_proba(x[:32]), probe.predict_proba(x[:32]))
        results[str(layer)] = metrics(labels[~mask], loaded.predict_proba(x[~mask]))
        results[str(layer)]['converged'] = probe.metadata['converged']
        print(f'Block {layer}: held-out accuracy {results[str(layer)]["accuracy"]:.4f}', flush=True)
        del x
    write_json(output / 'metrics' / 'synthetic.json', results)


@torch.inference_mode()
def generate_conversations(config, tokenizer, model, prompts, path):
    """Persist native assistant analysis/final messages, retaining failure counts."""
    conversations, failures = [], []
    with path.open('w') as handle:
        for index, prompt in enumerate(prompts):
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': prompt['content']}]
            prefix, _ = render(messages, bos=True)
            prefix += '<|start|>assistant'
            ids = tokenizer(prefix, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
            generated = model.generate(ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=config['max_new_tokens'], do_sample=True, temperature=1., top_p=1., top_k=0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            continuation = '<|start|>assistant' + tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=False)
            try:
                assistant = parse_generated(continuation)
            except ValueError as error:
                failures.append({'id': prompt['id'], 'reason': str(error), 'raw': continuation})
                print(f'Conversation {index+1}: incomplete, excluded', flush=True)
                continue
            record = {'id': prompt['id'], 'dataset': 'toxic-chat', 'messages': messages + assistant}
            conversations.append(record)
            handle.write(json.dumps(record) + '\n'); handle.flush()
            print(f'Generated conversation {index+1}/{len(prompts)}', flush=True)
    write_json(path.with_name('generation_failures.json'), failures)
    if len(conversations) < max(2, len(prompts)//2):
        raise RuntimeError('Too few completed conversations; inspect generation_failures.json')
    return conversations


def evaluate(config, tokenizer, model, conversations, output):
    probes = {layer: RoleProbe.load(output/'probes'/f'layer_{layer}.npz') for layer in config['layers']}
    summaries = {layer: [] for layer in probes}
    prediction_path = output/'artifacts'/'token_predictions.jsonl'
    with prediction_path.open('w') as handle:
        for conv in conversations:
            enc = tokenize_messages(tokenizer, conv['messages'], bos=True)
            positions = np.flatnonzero(enc['labels'] >= 0)
            states = extract_states(model, enc['input_ids'], list(probes), positions)
            labels = enc['labels'][positions]
            for layer, probe in probes.items():
                probabilities = probe.predict_proba(states[layer])
                summaries[layer].append({'conversation_id': conv['id'], 'labels': labels,
                                         'probabilities': probabilities})
                for index, position in enumerate(positions):
                    handle.write(json.dumps({
                        'conversation_id': conv['id'], 'layer': layer, 'token_index': int(position),
                        'token_id': int(enc['input_ids'][position]),
                        'message_index': int(enc['message_ids'][position]),
                        'character_offset': enc['offsets'][position].tolist(),
                        'true_role': ROLES[labels[index]],
                        'predicted_role': ROLES[probabilities[index].argmax()],
                        'probabilities': probabilities[index].tolist(),
                    }) + '\n')
    results = {}
    for layer, rows in summaries.items():
        summary = conversation_summary(rows)
        results[str(layer)] = {
            'token_metrics': metrics(np.concatenate([r['labels'] for r in rows]),
                                     np.concatenate([r['probabilities'] for r in rows])),
            'conversation_balanced': summary,
        }
        plot_summary(summary, output/'plots'/f'clean_chat_layer_{layer}.png',
                     f'GPT-OSS-20B clean conversations — block {layer}')
    write_json(output/'metrics'/'conversations.json', results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/experiments/gptoss_role_probes/config.json')
    parser.add_argument('--output', default='experiment_reports/role_probes/gptoss20b_baseline')
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate', 'audit'], default='all')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    for folder in ['artifacts', 'probes', 'metrics', 'plots']:
        (output/folder).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    tokenizer, model = load_model(config['model_path'])
    input_dir = output/'artifacts'/'inputs'
    prepared = json.loads((input_dir/'config.json').read_text())
    # Generation length does not affect the prepared training texts or user prompts.
    if {k:v for k,v in prepared.items() if k != 'max_new_tokens'} != {k:v for k,v in config.items() if k != 'max_new_tokens'}:
        raise ValueError('Prepared inputs do not match the requested configuration')
    metadata = {
        'stage': args.stage, 'generation_max_new_tokens': config['max_new_tokens'],
        'model_id': config['model_id'], 'model_revision': config['model_revision'],
        'precision': 'MXFP4 checkpoint dequantized to BF16', 'attention': 'eager',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        'upstream_commit': subprocess.check_output(['git', '-C', 'submodules/prompt-injection-as-role-confusion', 'rev-parse', 'HEAD'], text=True).strip(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(),
        'versions': {p: importlib.metadata.version(p) for p in ['torch','transformers','scikit-learn','cuml-cu13','numpy','datasets']},
        'input_sha256': hashlib.sha256((input_dir/'documents.jsonl').read_bytes()).hexdigest(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for root in ['src/role_probe_training/probes','src/experiments/gptoss_role_probes']
                          for p in Path(root).glob('*.py')},
    }
    write_json(output/'metrics'/f'run_metadata_{args.stage}.json', metadata)
    if args.stage == 'audit':
        from role_probe_training.probes.audit import audit_objective
        result = audit_objective(tokenizer, model, read_jsonl(input_dir/'documents.jsonl'),
            {layer: output/'probes'/f'layer_{layer}.npz' for layer in config['layers']})
        write_json(output/'metrics'/'optimizer_audit.json', result)
    if args.stage in ('all','train'):
        train(config, tokenizer, model, read_jsonl(input_dir/'documents.jsonl'), output, metadata)
    if args.stage in ('all','evaluate'):
        path = output/'artifacts'/'conversations.jsonl'
        if path.exists():
            if not path.with_name('generation_failures.json').exists():
                raise RuntimeError('Incomplete generation artifact; move it aside before retrying evaluation')
            conversations = read_jsonl(path)
            if len(conversations) < max(2, config['n_conversations']//2):
                raise RuntimeError('Too few completed conversations in saved evaluation data')
        else:
            conversations = generate_conversations(config, tokenizer, model,
                read_jsonl(input_dir/'prompts.jsonl'), path)
        evaluate(config, tokenizer, model, conversations, output)
    print('Completed', args.stage, flush=True)


if __name__ == '__main__':
    main()
