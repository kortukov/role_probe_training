"""Secondary analyses from saved per-token predictions (no GPU): context effects on user
turns, what later user turns are mislabelled as, and the version-2 system turn split
into the fixed GPT-OSS preamble and the sample's own instruction text."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from role_probe_training.probes.harmony import ROLES, SYSTEM_PROMPT, tokenize_messages

ARMS = ('c4_dolma', 'toucan', 'combined')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').split('\n') if line.strip()]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiment_reports/eval_role_probes/chat_eval/norobots'))
    parser.add_argument('--layers', type=int, nargs='+', default=[12, 16])
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b', local_files_only=True)
    out = {}
    # Version 1: first user turn with/without a system message; later user turns' predicted roles.
    rows = {r['id']: r for r in read_jsonl(args.root / 'inputs' / 'samples.jsonl')}
    first = {L: {a: defaultdict(lambda: [0, 0]) for a in ARMS} for L in args.layers}
    later = {L: {a: np.zeros(5, dtype=np.int64) for a in ARMS} for L in args.layers}
    for path in sorted((args.root / 'metrics' / 'version1_replay' / 'predictions').glob('*.npz')):
        saved = np.load(path); row = rows[path.stem]
        turns = list(saved['turn_labels']); mids = saved['message_ids']
        first_user = turns.index('user 1')
        later_user = [i for i, t in enumerate(turns) if t.startswith('user ') and t != 'user 1']
        for L in args.layers:
            for a in ARMS:
                pred = saved[f'predictions_{a}_{L}']
                mask = mids == first_user
                bucket = first[L][a]['with_system' if row['has_system'] else 'without_system']
                bucket[0] += int((pred[mask] == ROLES.index('user')).sum()); bucket[1] += int(mask.sum())
                if later_user:
                    mask = np.isin(mids, later_user)
                    later[L][a] += np.bincount(pred[mask], minlength=5)
    out['version1_first_user_turn_accuracy'] = {str(L): {a: {k: {'accuracy': c / t, 'tokens': t} for k, (c, t) in first[L][a].items()} for a in ARMS} for L in args.layers}
    out['version1_later_user_turns_predicted_as'] = {str(L): {a: {ROLES[i]: float(later[L][a][i] / later[L][a].sum()) for i in range(5)} for a in ARMS} for L in args.layers}
    # Version 2: split the system turn into the fixed preamble and the sample's instruction text.
    generated = {r['id']: r for r in read_jsonl(args.root / 'artifacts' / 'generated.jsonl')}
    preamble = {L: {a: [0, 0] for a in ARMS} for L in args.layers}
    instruction = {L: {a: [0, 0] for a in ARMS} for L in args.layers}
    for path in sorted((args.root / 'metrics' / 'version2_generated' / 'predictions').glob('*.npz')):
        saved = np.load(path); row = generated[path.stem]
        enc = tokenize_messages(tokenizer, row['messages'], bos=True)
        positions = np.flatnonzero(enc['labels'] >= 0)
        assert len(positions) == len(saved['labels'])
        system_start = enc['text'].index(SYSTEM_PROMPT)
        boundary = system_start + len(SYSTEM_PROMPT)
        offsets = enc['offsets'][positions]
        in_system = saved['message_ids'] == 0
        is_preamble = in_system & (offsets[:, 1] <= boundary)
        is_instruction = in_system & (offsets[:, 0] >= boundary)
        for L in args.layers:
            for a in ARMS:
                pred = saved[f'predictions_{a}_{L}']
                for mask, store in ((is_preamble, preamble), (is_instruction, instruction)):
                    store[L][a][0] += int((pred[mask] == 0).sum()); store[L][a][1] += int(mask.sum())
    out['version2_system_turn_split'] = {str(L): {a: {'preamble': {'accuracy': preamble[L][a][0] / preamble[L][a][1], 'tokens': preamble[L][a][1]},
                                                       'sample_instruction': {'accuracy': instruction[L][a][0] / max(1, instruction[L][a][1]), 'tokens': instruction[L][a][1]}}
                                                   for a in ARMS} for L in args.layers}
    (args.root / 'metrics' / 'analysis.json').write_text(json.dumps(out, indent=1) + '\n')
    for L in args.layers:
        print(f'--- layer {L}: version 1 first user turn accuracy, with system / without system')
        for a in ARMS:
            d = out['version1_first_user_turn_accuracy'][str(L)][a]
            print(f'  {a:9s} {100*d["with_system"]["accuracy"]:5.1f} ({d["with_system"]["tokens"]})  {100*d["without_system"]["accuracy"]:5.1f} ({d["without_system"]["tokens"]})')
        print(f'--- layer {L}: version 1 later user turns predicted as (system,user,cot,assistant,tool) %')
        for a in ARMS:
            print(f'  {a:9s}', [round(100 * v, 1) for v in out['version1_later_user_turns_predicted_as'][str(L)][a].values()])
        print(f'--- layer {L}: version 2 system turn: preamble vs sample instruction accuracy')
        for a in ARMS:
            d = out['version2_system_turn_split'][str(L)][a]
            print(f'  {a:9s} preamble {100*d["preamble"]["accuracy"]:5.1f} ({d["preamble"]["tokens"]})  instruction {100*d["sample_instruction"]["accuracy"]:5.1f} ({d["sample_instruction"]["tokens"]})')


if __name__ == '__main__':
    main()
