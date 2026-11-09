"""Train reusable probes and evaluate a small clean conversational baseline.

`--backend` selects how probes are fitted from scratch: `cuml` (default, the role-probe
paper's logistic regression) or `adamw` (the training_against_probes step-0 fit). Several
backends in one `train` run share one activation extraction. cuML results keep their
paths (probes/, metrics/synthetic.json, ...); every other backend adds its name as a
suffix (probes_adamw/, metrics/synthetic_adamw.json, ...), so both can be compared.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from role_probe_training.harmony import (LIBRARY, LIBRARY_VERSION, ROLES, RenderConfig, completion_prompt, encode_record,
                                         encoding, render_document, render_turns, request_record, tokens, walk_completion)
from role_probe_training.probes.model import extract_states, load_model
from role_probe_training.probes.linear import RoleProbe, fit_probe
from role_probe_training.probes.evaluation import metrics, conversation_summary, plot_summary


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def suffix(backend):
    return '' if backend == 'cuml' else f'_{backend}'


def train(config, tokenizer, model, documents, output, metadata, backends=('cuml',), fit_steps=5000):
    if config['token_stride'] != 1:
        raise ValueError('Reproduction requires all content tokens (token_stride=1)')
    layers = config['layers']
    # Which sources' training documents fit the probe; held-out documents of every
    # source are always scored, so all arms share identical test sets.
    train_sources = set(config.get('train_sources') or [s['id'] for s in config['sources']])
    documents = [d for d in documents if d['split'] != 'train' or d['source'] in train_sources]
    features = {layer: [] for layer in layers}
    labels, splits, collections, groups = [], [], [], []
    for index, document in enumerate(documents):
        headers = document.get('header', {})
        for role in ROLES:
            # Toucan documents carry native header fields for their own role (replay.py
            # convention); C4/Dolma documents have none and render exactly as before.
            enc = render_document(role, document['text'], headers.get(role)).encoded()
            positions = np.flatnonzero(enc['labels'] >= 0)
            states = extract_states(model, enc['input_ids'], layers, positions)
            for layer in layers:
                features[layer].append(states[layer].astype(np.float16))
            labels.extend(enc['labels'][positions])
            splits.extend([document['split']] * len(positions))
            collections.extend([document.get('collection', 'neutral')] * len(positions))
            groups.extend([len(ROLES)*index + ROLES.index(role)] * len(positions))  # one rendered message
        if (index+1) % 10 == 0:
            print(f'Extracted {index+1}/{len(documents)} training documents', flush=True)
    labels = np.asarray(labels)
    mask = np.asarray(splits) == 'train'
    collections = np.asarray(collections)
    groups = np.asarray(groups)
    results = {backend: {} for backend in backends}
    for layer in layers:
        x = np.concatenate(features.pop(layer))
        for backend in backends:
            print(f'Fitting block {layer} with {backend}: {mask.sum()} train / {(~mask).sum()} test tokens', flush=True)
            fit_arguments = (dict(C=config['C']) if backend == 'cuml' else
                             dict(max_iter=fit_steps, groups=groups[mask], seed=config['seed']))
            probe = fit_probe(x[mask], labels[mask], backend=backend, **fit_arguments, metadata={
                **metadata, 'layer': layer, 'activation': 'post_attention_layernorm_output_pre_mlp',
                'layer_indexing': 'zero_based', 'hidden_size': x.shape[1],
                'training_config': config,
            })
            path = output / f'probes{suffix(backend)}' / f'layer_{layer}.npz'
            probe.save(path)
            loaded = RoleProbe.load(path)
            np.testing.assert_allclose(loaded.predict_proba(x[:32]), probe.predict_proba(x[:32]))
            probabilities = loaded.predict_proba(x[~mask])
            result = results[backend][str(layer)] = metrics(labels[~mask], probabilities)
            result['converged'] = probe.metadata['converged']
            result['train_sources'] = sorted(train_sources)
            result['by_collection'] = {
                name: metrics(labels[~mask][collections[~mask] == name], probabilities[collections[~mask] == name])
                for name in sorted(set(collections[~mask]))}
            print(f'Block {layer} ({backend}): held-out accuracy {result["accuracy"]:.4f}; by collection ' +
                  ', '.join(f'{k} {v["accuracy"]:.4f}' for k, v in result['by_collection'].items()), flush=True)
        del x
    for backend in backends:
        write_json(output / 'metrics' / f'synthetic{suffix(backend)}.json', results[backend])


def render_config(config):
    return RenderConfig(config.get('reasoning_effort', 'medium'), config.get('conversation_start_date', '2025-12-31'))


def check_chat_completion(completion):
    """Reject anything but complete analysis/final messages ending with <|return|>."""
    entries, error = walk_completion(completion)
    if error is not None:
        return entries, f'unparseable: {error}'
    if not completion or completion[-1] != tokens().ret or not entries or not entries[-1]['complete']:
        return entries, 'generation did not finish a final answer'
    for entry in entries:
        if entry['role'] != 'assistant' or entry['recipient'] or entry['channel'] not in ('analysis', 'final'):
            return entries, f"unexpected generated message: channel {entry['channel']!r} recipient {entry['recipient']!r}"
    if entries[-1]['channel'] != 'final':
        return entries, 'last message is not final'
    return entries, None


@torch.inference_mode()
def generate_conversations(config, tokenizer, model, prompts, path):
    """Persist each generation as a token record (prompt and completion token IDs with
    message spans) plus readable messages, retaining failure counts."""
    conversations, failures = [], []
    rendering_config = render_config(config)
    with path.open('w') as handle:
        for index, prompt in enumerate(prompts):
            turns = [{'role': 'user', 'content': prompt['content']}]
            rendering = render_turns(turns, rendering_config)
            ids = torch.tensor([completion_prompt(rendering)], device=model.device)
            generated = model.generate(ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=config['max_new_tokens'], do_sample=True, temperature=1., top_p=1., top_k=0,
                eos_token_id=list(tokens().stop),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            completion = generated[0, ids.shape[1]:].tolist()
            entries, reason = check_chat_completion(completion)
            if reason is not None:
                failures.append({'id': prompt['id'], 'reason': reason, 'raw': encoding().decode(completion)})
                print(f'Conversation {index+1}: {reason}, excluded', flush=True)
                continue
            record = request_record(rendering, completion, rendering_config, id=prompt['id'], dataset='toxic-chat',
                completion_source=len(turns), messages=turns + [
                    {'role': 'assistant', 'channel': e['channel'], 'content': e['content']} for e in entries])
            conversations.append(record)
            handle.write(json.dumps(record) + '\n'); handle.flush()
            print(f'Generated conversation {index+1}/{len(prompts)}', flush=True)
    write_json(path.with_name('generation_failures.json'), failures)
    if len(conversations) < max(2, len(prompts)//2):
        raise RuntimeError('Too few completed conversations; inspect generation_failures.json')
    return conversations


def evaluate(config, tokenizer, model, conversations, output, backend='cuml'):
    probes = {layer: RoleProbe.load(output/f'probes{suffix(backend)}'/f'layer_{layer}.npz') for layer in config['layers']}
    summaries = {layer: [] for layer in probes}
    prediction_path = output/'artifacts'/f'token_predictions{suffix(backend)}.jsonl'
    with prediction_path.open('w') as handle:
        for conv in conversations:
            enc = encode_record(conv)
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
        plot_summary(summary, output/'plots'/f'clean_chat_layer_{layer}{suffix(backend)}.png',
                     f'GPT-OSS-20B clean conversations — block {layer}' + (f' ({backend} probe)' if suffix(backend) else ''))
    write_json(output/'metrics'/f'conversations{suffix(backend)}.json', results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/experiments/gptoss_role_probes/config.json')
    parser.add_argument('--output', default='experiment_reports/gptoss_role_probes/gptoss20b_baseline')
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate', 'audit'], default='all')
    parser.add_argument('--backend', nargs='+', choices=['cuml', 'adamw'], default=['cuml'],
                        help='probe fitter(s); see the module docstring for the output paths')
    parser.add_argument('--fit-steps', type=int, default=5000,
                        help='optimizer steps of the adamw backend (upstream probe_training.max_steps)')
    args = parser.parse_args()
    if args.stage == 'audit' and args.backend != ['cuml']:
        parser.error('The optimizer audit checks the cuML logistic-regression objective only')
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    for folder in ['artifacts', 'metrics', 'plots'] + [f'probes{suffix(backend)}' for backend in args.backend]:
        (output/folder).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    if 'cuml' in args.backend and args.stage in ('all', 'train') and importlib.util.find_spec('cuml') is None:
        raise SystemExit('Probe fitting with the cuml backend needs cuML: run with `uv run --project src/experiments/gptoss_role_probes/fit_env --frozen --no-sync python ...`')
    tokenizer, model = load_model(config['model_path'])
    input_dir = output/'artifacts'/'inputs'
    prepared = json.loads((input_dir/'config.json').read_text())
    # Generation length, the fitted-source selection and the Harmony system-message settings
    # do not affect the prepared texts or prompts.
    ignored = {'max_new_tokens', 'train_sources', 'reasoning_effort', 'conversation_start_date'}
    if {k:v for k,v in prepared.items() if k not in ignored} != {k:v for k,v in config.items() if k not in ignored}:
        raise ValueError('Prepared inputs do not match the requested configuration')
    metadata = {
        'stage': args.stage, 'generation_max_new_tokens': config['max_new_tokens'],
        'probe_backends': args.backend, 'adamw_fit_steps': args.fit_steps if 'adamw' in args.backend else None,
        'model_id': config['model_id'], 'model_revision': config['model_revision'],
        'precision': 'MXFP4 checkpoint dequantized to BF16', 'attention': 'eager',
        'rendering': render_config(config).metadata(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        'upstream_commit': subprocess.check_output(['git', '-C', 'submodules/prompt-injection-as-role-confusion', 'rev-parse', 'HEAD'], text=True).strip(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(),
        'versions': {p: importlib.metadata.version(p) for p in ['torch','transformers','scikit-learn','cuml-cu13','numpy','datasets',LIBRARY]},
        'input_sha256': hashlib.sha256((input_dir/'documents.jsonl').read_bytes()).hexdigest(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [q for root in ['src/role_probe_training/probes','src/role_probe_training/harmony','src/experiments/gptoss_role_probes']
                                    for q in Path(root).glob('*.py')] + [Path('src/role_probe_training/training/probe_fit.py')]},
    }
    tag = '' if args.backend == ['cuml'] else '_' + '_'.join(args.backend)
    write_json(output/'metrics'/f'run_metadata_{args.stage}{tag}.json', metadata)
    if args.stage == 'audit':
        from role_probe_training.probes.audit import audit_objective
        result = audit_objective(tokenizer, model, read_jsonl(input_dir/'documents.jsonl'),
            {layer: output/'probes'/f'layer_{layer}.npz' for layer in config['layers']},
            train_sources=config.get('train_sources'))
        write_json(output/'metrics'/'optimizer_audit.json', result)
    if args.stage in ('all','train'):
        train(config, tokenizer, model, read_jsonl(input_dir/'documents.jsonl'), output, metadata,
              args.backend, args.fit_steps)
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
        for backend in args.backend:
            evaluate(config, tokenizer, model, conversations, output, backend)
    print('Completed', args.stage, flush=True)


if __name__ == '__main__':
    main()
