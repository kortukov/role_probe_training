"""Same-batch KL masks for off-policy training: one row per training rollout.

The KL term compares the current and the base model on the tokens the model itself wrote
(reasoning, answers and tool-call arguments) of the very rollouts in the role-loss batch.
Each row holds the whole history as tokens and a mask of those positions. Rollouts need the
token records of the Harmony server; the trainer reads the same records for the role loss.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from role_probe_training.harmony import ROLES
from role_probe_training.training.data import read_jsonl, trajectory_examples

MODEL_ROLES = {ROLES.index('cot'), ROLES.index('assistant')}
NATIVE_CONTEXT = 131072


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='folder for anchors.jsonl and anchors_manifest.json')
    parser.add_argument('--expected', type=int, default=2700)
    args = parser.parse_args()
    target = args.output/'anchors.jsonl'
    if target.exists():
        raise FileExistsError(target)
    rows = read_jsonl(args.trajectories)
    examples = trajectory_examples(args.trajectories)
    if len(rows) != args.expected or len(examples) != len(rows):
        raise ValueError(f'{len(rows)} rollouts, {len(examples)} with token records, {args.expected} expected; '
                         'a rollout without a model call cannot be trained on')
    validation = trajectory_examples(args.validation)
    if not validation or {r['sample_id'] for r in rows} & {r['sample_id'] for r in read_jsonl(args.validation)}:
        raise ValueError('Validation rollouts are missing or share tasks with training')
    args.output.mkdir(parents=True, exist_ok=True)
    lengths, scored, role_counts = [], [], Counter()
    with target.open('w') as stream:
        for example in examples:
            mask = [label in MODEL_ROLES for label in example['labels']]
            if not any(mask) or len(mask) != len(example['ids']) or len(mask) > NATIVE_CONTEXT:
                raise ValueError(f"Rollout {example['id']}: no model-written tokens or above the native context")
            stream.write(json.dumps(dict(id=example['id'], input_ids=example['ids'], completion_mask=mask)) + '\n')
            lengths.append(len(mask)); scored.append(sum(mask))
            role_counts.update(ROLES[label] for label, keep in zip(example['labels'], mask) if keep)
    manifest = dict(source=str(args.trajectories), source_sha256=hashlib.sha256(args.trajectories.read_bytes()).hexdigest(),
        anchors=str(target), anchors_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        selection='every training rollout; KL uses the exact rollouts of the current role-loss batch',
        examples=len(examples), validation_examples=len(validation),
        mask='content tokens the model wrote (cot, assistant); headers, delimiters and other roles excluded',
        scored_tokens_by_role=dict(role_counts), context='whole history; no truncation or chunking',
        total_context_tokens=sum(lengths), max_context_tokens=max(lengths),
        total_kl_tokens=sum(scored), min_kl_tokens=min(scored))
    (args.output/'anchors_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
