"""Prepare pinned public-data samples for the upstream training recipe. No model generation or fitting.

Text sources (C4/Dolma) are sampled exactly as in the baseline. A `toucan` source adds
tool-use messages from Agent-Ark/Toucan-1.5M: every message of a sample becomes one
document, tool calls/responses carry the native GPT-OSS header metadata used by
`trajectories/replay.py`, samples never share an MCP server, and the held-out split is
assigned per sample so all its messages stay on one side.
"""

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

TOUCAN_ROLES = ('user', 'assistant', 'tool_call', 'tool_response')


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def truncate(tokenizer, config, text):
    """The baseline cap for every source: tokenize, cut at sequence_length, decode."""
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=config['sequence_length'])['input_ids']
    return ids, (tokenizer.decode(ids) if ids else '')


def unusable(tokenizer, seen, digest, text):
    return digest in seen or any(t in text for t in tokenizer.all_special_tokens)


def text_sources(config):
    return [s for s in config['sources'] if s.get('kind', 'text') == 'text']


def prepare_text_documents(config, tokenizer, seen):
    """Baseline sampling, byte-identical to the original recipe for the same config."""
    documents = []
    sources = text_sources(config)
    for source_index, source in enumerate(sources):
        target = (int(config['n_documents'] * source['fraction'])
                  if source_index < len(sources) - 1
                  else config['n_documents'] - len(documents))
        stream = load_dataset(source['id'], source['name'], split=source['split'],
                              revision=source['revision'], streaming=True, token=False)
        stream = stream.shuffle(seed=config['seed'], buffer_size=config['shuffle_buffer'])
        added = 0
        for index, row in enumerate(stream):
            ids, text = truncate(tokenizer, config, row['text'])
            if not ids:
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if unusable(tokenizer, seen, digest, text):
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
    return documents


def parse_tool_call(content):
    try:
        call = json.loads(content)
    except ValueError:
        call = ast.literal_eval(content)
    arguments = call['arguments']
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return call['name'], arguments


def toucan_server(function, bare_names, known_servers):
    """Server prefix of a Toucan tool name, e.g. lyrical-mcp-find_rhymes -> lyrical-mcp."""
    for bare in sorted(bare_names, key=len, reverse=True):
        for separator in ('-', '_'):
            if function.endswith(separator + bare) and len(function) > len(bare) + 1:
                return function[:-len(bare) - 1]
    for server in sorted(known_servers, key=len, reverse=True):
        if function.startswith(server + '-') or function.startswith(server + '_'):
            return server
    raise ValueError(f'Cannot resolve the server of {function!r}')


def toucan_messages(row):
    """Normalize one Toucan row; raise ValueError when its structure is not usable."""
    messages = json.loads(row['messages'])
    # Multi-server rows list targets as 'Server Name::tool'; keep only the tool part.
    bare_names = [t.split('::')[-1].strip() for t in row['target_tools'].split(',') if t.strip()]
    pending, out, servers = [], [], []
    for message in messages:
        role, content = message['role'], message.get('content')
        if role not in TOUCAN_ROLES:
            raise ValueError(f'Unsupported Toucan role {role!r}')
        if role == 'tool_call':
            name, arguments = parse_tool_call(content)
            server = toucan_server(name, bare_names, servers)
            if server not in servers:
                servers.append(server)
            pending.append(name)
            out.append({'role': role, 'text': arguments, 'function': name})
        elif role == 'tool_response':
            if not pending:
                raise ValueError('Tool response without a pending call')
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            out.append({'role': role, 'text': text, 'function': pending.pop(0)})
        else:
            if not isinstance(content, str):
                raise ValueError('Non-string message content')
            out.append({'role': role, 'text': content, 'function': None})
    if pending:
        raise ValueError('Unanswered tool calls')
    if not any(m['role'] == 'tool_call' for m in out):
        raise ValueError('No tool calls')
    for index, message in enumerate(out):
        following = out[index + 1]['role'] if index + 1 < len(out) else None
        message['followed_by_call'] = following == 'tool_call'
    return out, servers


def toucan_headers(message, first_function):
    """Per-role header fields added when this message is wrapped in that role.

    Only the wrap matching the message's native role (and the tool-role function
    name) differs from the generic C4/Dolma headers; see trajectories/replay.py.
    """
    headers = {'tool': {'name': message['function'] or first_function}}
    if message['role'] == 'tool_call':
        headers['assistant'] = {'channel': 'commentary', 'recipient': 'functions.' + message['function'],
                                'ending': '<|call|>'}
    elif message['role'] == 'assistant':
        headers['assistant'] = ({'channel': 'commentary', 'ending': '<|end|>'} if message['followed_by_call']
                                else {'channel': 'final', 'ending': '<|return|>'})
    return headers


def prepare_toucan_documents(config, source, tokenizer, seen, rng):
    stream = load_dataset(source['id'], source['name'], split=source['split'],
                          revision=source['revision'], streaming=True, token=False)
    stream = stream.shuffle(seed=config['seed'], buffer_size=config['shuffle_buffer'])
    samples, used_servers = [], set()
    skipped, dropped, kept, tokens = Counter(), Counter(), Counter(), Counter()
    for index, row in enumerate(stream):
        if row['subset_name'] != source['subset']:
            continue
        try:
            messages, servers = toucan_messages(row)
        except (ValueError, KeyError, SyntaxError, TypeError) as error:
            skipped[f'unusable: {type(error).__name__}'] += 1
            continue
        if used_servers.intersection(servers):
            skipped['server_overlap'] += 1
            continue
        first_function = next(m['function'] for m in messages if m['role'] == 'tool_call')
        documents = []
        for message_index, message in enumerate(messages):
            ids, text = truncate(tokenizer, config, message['text'])
            if len(ids) < source['min_tokens']:
                dropped[f'{message["role"]}: below min_tokens'] += 1
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if unusable(tokenizer, seen, digest, text):
                dropped[f'{message["role"]}: duplicate or special token'] += 1
                continue
            seen.add(digest)
            kept[message['role']] += 1
            tokens[message['role']] += len(ids)
            documents.append({
                'id': digest, 'source': source['id'], 'collection': 'toucan', 'stream_index': index,
                'text': text, 'n_tokens': len(ids), 'group': row['uuid'], 'servers': servers,
                'message_index': message_index, 'message_role': message['role'],
                'function': message['function'], 'header': toucan_headers(message, first_function)})
        if not documents:
            skipped['no_documents'] += 1
            continue
        used_servers.update(servers)
        samples.append(documents)
        if len(samples) == source['n_samples']:
            break
    if len(samples) != source['n_samples']:
        raise RuntimeError(f'Insufficient usable samples from {source["id"]}: {len(samples)}')
    # Sample-grouped split: every message of a held-out sample, in all roles, is test data.
    held = set(rng.permutation(len(samples))[:max(1, round(len(samples) * config['test_fraction']))])
    documents = []
    for sample_index, sample in enumerate(samples):
        for document in sample:
            document['split'] = 'test' if sample_index in held else 'train'
            documents.append(document)
    manifest = {
        'source': source, 'n_samples': len(samples), 'n_test_samples': len(held),
        'n_servers': len(used_servers), 'n_documents': len(documents),
        'documents_by_split': dict(Counter(d['split'] for d in documents)),
        'kept_messages_by_role': dict(kept), 'tokens_by_role': dict(tokens),
        'dropped_messages': dict(dropped), 'skipped_samples': dict(skipped),
        'sample_uuids': [s[0]['group'] for s in samples],
    }
    print(f'Prepared {len(documents)} Toucan documents from {len(samples)} samples; '
          f'kept {dict(kept)}; dropped {dict(dropped)}; skipped {dict(skipped)}', flush=True)
    return documents, manifest


def prepare(config, output):
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    seen = set()
    documents = prepare_text_documents(config, tokenizer, seen)
    rng = np.random.default_rng(config['seed'])
    # Source-stratified, document-grouped split: every role variant stays together.
    for source in text_sources(config):
        indices = [i for i,d in enumerate(documents) if d['source'] == source['id']]
        held = set(rng.permutation(indices)[:max(1, round(len(indices)*config['test_fraction']))])
        for i in indices:
            documents[i]['split'] = 'test' if i in held else 'train'
    manifests = {}
    for source in config['sources']:
        if source.get('kind', 'text') == 'toucan':
            added, manifests[source['id']] = prepare_toucan_documents(config, source, tokenizer, seen, rng)
            documents.extend(added)
    write_jsonl(output / 'documents.jsonl', documents)
    if manifests:
        (output / 'toucan_manifest.json').write_text(json.dumps(manifests, indent=2) + '\n')
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
    parser.add_argument('--output', default='experiment_reports/gptoss_role_probes/gptoss20b_baseline/artifacts/inputs')
    args = parser.parse_args()
    prepare(json.loads(Path(args.config).read_text()), Path(args.output))
