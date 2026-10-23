"""Measure full-trajectory lengths and padding costs without loading model weights."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics

from transformers import AutoTokenizer
from role_probe_training.trajectories.replay import encode_record


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b', local_files_only=True)
    lengths, messages, roles = [], [], Counter()
    for line in args.data.open():
        row = json.loads(line)
        encoded = encode_record(tokenizer, row)
        lengths.append(len(encoded['input_ids']))
        messages.append(len(row['messages']))
        roles.update(int(label) for label in encoded['labels'] if label >= 0)
    ordered = sorted(lengths)
    shuffled = lengths.copy()
    random.Random(123).shuffle(shuffled)
    costs = {}
    for name, sequence in [('random', shuffled), ('length_sorted', ordered)]:
        for size in (1, 2, 4, 8, 16):
            batches = [sequence[i:i+size] for i in range(0, len(sequence), size)]
            slots = sum(len(batch) * max(batch) for batch in batches)
            # Dense attention's quadratic length proxy, not a timing prediction.
            attention = sum(len(batch) * max(batch)**2 for batch in batches)
            costs[f'{name}_batch{size}'] = {
                'token_slots': slots, 'padding_fraction': 1-sum(lengths)/slots,
                'token_cost_ratio': slots/sum(lengths),
                'quadratic_attention_ratio': attention/sum(n*n for n in lengths)}
    result = {
        'source': str(args.data), 'trajectories': len(lengths),
        'total_tokens': sum(lengths), 'labeled_tokens_by_index': dict(sorted(roles.items())),
        'mean_tokens': statistics.mean(lengths), 'median_tokens': statistics.median(lengths),
        'min_tokens': min(lengths), 'p90_tokens': ordered[math.ceil(.9*len(ordered))-1],
        'p95_tokens': ordered[math.ceil(.95*len(ordered))-1], 'max_tokens': max(lengths),
        'median_messages': statistics.median(messages),
        'over_context': {str(n): sum(x > n for x in lengths) for n in (16384,32768,65536,131072)},
        'optimizer_updates_per_epoch_at_16_trajectories': math.ceil(len(lengths)/16),
        'padding': costs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
