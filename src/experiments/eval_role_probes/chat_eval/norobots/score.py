"""Score several probe arms on whole No Robots conversations, one forward pass each.

Per arm and layer it reports overall token accuracy, per-role recall, per-turn accuracy
(system, user k, reasoning k, assistant k), mean per-conversation accuracy, and the
system-turn accuracy split by whether the sample supplied its own system message.
`--mode replay` renders the dataset's own answers with the Harmony library; `--mode
generated` reads the token records written by generate.py without re-rendering.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from role_probe_training.harmony import ROLES, RenderConfig, history, render_turns
from role_probe_training.probes.linear import RoleProbe
from role_probe_training.probes.model import extract_states, load_model


def read_jsonl(path):
    # Split on newlines only: json.dumps(ensure_ascii=False) leaves U+2028-style separators inside strings.
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').split('\n') if line.strip()]


def turn_labels(rendering):
    """One label per rendered message: 'system' (Harmony system message, developer message and
    tool schemas), 'user 1', 'reasoning 1', 'assistant 1', 'user 2', ..."""
    labels, seen = [], defaultdict(int)
    for entry in rendering.messages:
        if entry['kind'] in ('system', 'developer', 'tools'):
            labels.append('system')
            continue
        if entry['kind'] == 'assistant':
            name = 'reasoning' if entry['channel'] == 'analysis' else 'assistant'
        else:
            name = entry['kind']
        seen[name] += 1
        labels.append(f'{name} {seen[name]}')
    return labels


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--mode', choices=['replay', 'generated'], required=True)
    parser.add_argument('--reasoning-effort', default='medium')
    parser.add_argument('--date', default='2025-12-31')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', type=Path, nargs='+', required=True, help='Arm directories holding probes/layer_L.npz')
    parser.add_argument('--layers', type=int, nargs='+', default=[4, 8, 12, 16, 20])
    parser.add_argument('--model', default='.cache/models/gpt-oss-20b')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'predictions').mkdir(exist_ok=True)
    probes = {arm.name: {L: RoleProbe.load(arm / 'probes' / f'layer_{L}.npz') for L in args.layers} for arm in args.arms}
    tokenizer, model = load_model(args.model)
    rows = read_jsonl(args.input)[:args.limit]
    keys = [(a, L) for a in probes for L in args.layers]
    confusion = {k: np.zeros((5, 5), dtype=np.int64) for k in keys}
    turns = {k: defaultdict(lambda: [0, 0]) for k in keys}
    per_sample = {k: [] for k in keys}
    system_split = {k: {'with_sample_system': [0, 0], 'without_sample_system': [0, 0]} for k in keys}
    config = RenderConfig(args.reasoning_effort, args.date)
    for n, row in enumerate(rows):
        rendering = render_turns(row['replay_turns'], config) if args.mode == 'replay' else history(row)
        labels_by_turn = turn_labels(rendering)
        enc = rendering.encoded()
        positions = np.flatnonzero(enc['labels'] >= 0)
        labels = enc['labels'][positions]
        message_ids = enc['message_ids'][positions]
        states = extract_states(model, enc['input_ids'], args.layers, positions)
        saved = {'labels': labels.astype(np.int8), 'message_ids': message_ids.astype(np.int16),
                 'turn_labels': np.asarray(labels_by_turn),
                 'message_kinds': np.asarray([m['kind'] for m in rendering.messages])}
        for arm, layers in probes.items():
            for L, probe in layers.items():
                predictions = probe.predict_proba(states[L]).argmax(1)
                key = (arm, L)
                np.add.at(confusion[key], (labels, predictions), 1)
                correct = predictions == labels
                per_sample[key].append(float(correct.mean()))
                for mid in np.unique(message_ids):
                    mask = message_ids == mid
                    entry = turns[key][labels_by_turn[int(mid)]]
                    entry[0] += int(correct[mask].sum()); entry[1] += int(mask.sum())
                    if labels_by_turn[int(mid)] == 'system':
                        bucket = system_split[key]['with_sample_system' if row.get('has_system') else 'without_sample_system']
                        bucket[0] += int(correct[mask].sum()); bucket[1] += int(mask.sum())
                saved[f'predictions_{arm}_{L}'] = predictions.astype(np.int8)
        np.savez_compressed(args.output / 'predictions' / f'{row["id"]}.npz', **saved)
        if (n + 1) % 50 == 0:
            print(f'Scored {n + 1}/{len(rows)} conversations', flush=True)
    result = {'n_conversations': len(rows), 'input': str(args.input), 'mode': args.mode, 'rendering': config.metadata(), 'roles': list(ROLES), 'arms': {}}
    for arm in probes:
        result['arms'][arm] = {}
        for L in args.layers:
            key = (arm, L); matrix = confusion[key]
            support = matrix.sum(1)
            recall = np.divide(matrix.diagonal(), support, out=np.zeros(5), where=support > 0)
            result['arms'][arm][str(L)] = {
                'accuracy': float(matrix.trace() / matrix.sum()),
                'macro_recall': float(recall[support > 0].mean()),
                'per_role_recall': {ROLES[i]: (float(recall[i]) if support[i] else None) for i in range(5)},
                'role_support': {ROLES[i]: int(support[i]) for i in range(5)},
                'confusion_counts': matrix.tolist(),
                'mean_conversation_accuracy': float(np.mean(per_sample[key])),
                'per_turn': {name: {'accuracy': c / t, 'tokens': t} for name, (c, t) in sorted(turns[key].items())},
                'system_turn_by_source': {name: {'accuracy': (c / t if t else None), 'tokens': t}
                                          for name, (c, t) in system_split[key].items()},
            }
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=1) + '\n')
    for arm in probes:
        r = result['arms'][arm]['12']
        print(arm, 'layer 12 accuracy', round(r['accuracy'], 4), 'macro', round(r['macro_recall'], 4), flush=True)


if __name__ == '__main__':
    main()
