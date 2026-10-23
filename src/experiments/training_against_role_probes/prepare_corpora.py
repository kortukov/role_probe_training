"""Prepare all ASTRA rows and the exact DepthBench train-task partitions."""
import argparse
import json
from pathlib import Path
import subprocess
import pyarrow.parquet as pq
from role_probe_training.trajectories.corpora import (
    astra_record, corpus_summary, depthbench_training, sha256, write_jsonl,
)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path('experiment_reports/training_against_role_probes'))
    a = p.parse_args()
    astra = a.root/'astra_sft_1k'
    source = astra/'data/source/data.parquet'
    revision = (astra/'data/source/.cache/huggingface/download/data.parquet.metadata').read_text().splitlines()[0]
    rows = [astra_record(r, i, revision) for i, r in enumerate(pq.read_table(source).to_pylist())]
    assert len(rows) == 1000
    output = astra/'data/trajectories.jsonl'
    write_jsonl(output, rows)
    manifest = corpus_summary(rows) | {'source_revision': revision, 'source_sha256': sha256(source),
        'trajectory_sha256': sha256(output), 'dataset_split': 'train',
        'source': 'https://huggingface.co/datasets/Emperorizzis/ASTRA-SFT-1k',
        'conversion': 'All 1000 rows, all messages, reasoning_content -> thinking; function -> tool; no truncation',
        'tools': 'mcp_info.base_info.tool_list; synthetic context excluded from probe metrics',
        'adapter_sha256': sha256('src/role_probe_training/trajectories/corpora.py')}
    (astra/'data_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('ASTRA', json.dumps(manifest))
    depth = a.root/'depthbench'
    upstream = depth/'data/upstream'
    tasks, functions = depthbench_training(upstream)
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    counts = {}
    for n in range(7):
        group = [dict(task_id=k, **v) for k, v in tasks.items() if v['num_calls'] == n]
        write_jsonl(depth/f'data/tasks/calls_{n}.jsonl', group)
        counts[n] = len(group)
    manifest = {'upstream_url': 'https://github.com/anaykulkarni/TIER', 'revision': revision,
        'source_sha256': sha256(upstream/'data/depthbench/full_dataset.json'),
        'loader_sha256': sha256(upstream/'src/tier/data/dataset_loader.py'),
        'split': 'upstream create_seen_unseen_dataset(chain=True), training partition only',
        'training_functions': sorted(functions), 'prompts_by_required_calls': counts,
        'attempts_per_prompt': 20, 'expected_rollouts': 20*len(tasks),
        'model_update_groups': [1, 2, 3],
        'expected_training_rollouts': 20*sum(counts[n] for n in (1, 2, 3)),
        'task_ids': sorted(tasks, key=int)}
    (depth/'data_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('DepthBench', counts)


if __name__ == '__main__':
    main()
