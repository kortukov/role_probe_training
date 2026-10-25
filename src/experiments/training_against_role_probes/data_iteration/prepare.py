"""Build fixed full-history KL pools from saved training data, never test tasks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from role_probe_training.training.data import trajectory_examples


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    output = args.root
    target = output/'data/anchors.jsonl'
    if target.exists():
        raise FileExistsError(target)
    tokenizer = AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b', local_files_only=True)
    source = output/'data/train27000/trajectories.jsonl'
    examples = trajectory_examples(tokenizer, source)
    assert len(examples) == 27000
    label_set = {2, 3}
    selection = 'All 27000 AppWorld training trajectories; exact current role-loss batch'
    target.parent.mkdir(parents=True, exist_ok=True)
    lengths = []; scored = []; role_counts = Counter()
    with target.open('w') as f:
        for x in examples:
            mask = [label in label_set for label in x['labels']]
            assert any(mask) and len(mask) == len(x['ids']) <= 131072
            row = dict(id=x['id'], input_ids=x['ids'], completion_mask=mask)
            f.write(json.dumps(row)+'\n')
            lengths.append(len(mask)); scored.append(sum(mask))
            role_counts.update(y for y, keep in zip(x['labels'], mask) if keep)
    manifest = dict(variant='same_batch_27000', source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        anchors=str(target), anchors_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        selection=selection, examples=len(examples), selected_ids=[x['id'] for x in examples],
        mask='Unshifted content positions; headers, delimiters and synthetic tool schemas excluded',
        roles_scored=sorted(label_set), scored_tokens_by_role=dict(role_counts),
        context='Complete history; no truncation or chunking',
        total_context_tokens=sum(lengths), max_context_tokens=max(lengths),
        total_kl_tokens=sum(scored), min_kl_tokens=min(scored),
        model_revision='6cee5e81ee83917806bbde320786a8fb61efebee')
    (output/'anchors_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'selected_ids'}), flush=True)


if __name__ == '__main__':
    main()
