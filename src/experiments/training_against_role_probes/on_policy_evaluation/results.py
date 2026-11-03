"""Materialize each completed benchmark independently, without rerunning models."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--benchmark',choices=['appworld_test','tau2_retail','swe_bench'],required=True)
    a=p.parse_args()
    if a.benchmark=='swe_bench':
        task_path=a.root/'post_training/swe_inspect_generous_task_results.json'
        score_path=a.root/'predictions/swe_inspect_generous/summary.json'
    else:
        task_path=a.root/f'data/post_training/{a.benchmark}/manifest.json'
        score_path=a.root/f'predictions/post_training/{a.benchmark}/summary.json'
    task=json.loads(task_path.read_text()); score=json.loads(score_path.read_text())
    expected={'appworld_test':168,'tau2_retail':114,'swe_bench':500}[a.benchmark]
    assert task['samples']==score['trajectories']==expected
    assert task['input_sha256']==score['input_sha256']
    export=json.loads((a.root/'data/post_training/model/export.json').read_text())
    assert export['adapter_sha256']==score['model_revision']
    result=dict(benchmark=a.benchmark,task=task,probe=score,checkpoint=export['checkpoint'],
                source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (task_path,score_path)})
    output=a.root/'post_training/results';output.mkdir(parents=True,exist_ok=True)
    (output/f'{a.benchmark}.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'# {a.benchmark}: final-checkpoint evaluation','',
           'Task success includes all assigned attempts. Role accuracy is content-token-weighted argmax accuracy from a freshly fitted layer-12 probe, replaying whole histories.','',
           '| Task metric | Successes / assigned | Success rate |','| --- | ---: | ---: |']
    for name,m in task['metrics'].items():
        lines.append(f"| {name} | {m['successes']}/{m['total']} | {100*m['success_rate']:.2f}% |")
    lines += ['',f"Role accuracy: **{100*score['accuracy']:.2f}%**. Scored histories: **{score['scored_trajectories']}/{expected}**.",'',
              'Histories above native context are excluded only from role metrics; task denominators retain every attempt. No histories are chunked or truncated.', '',
              'These measurements do not alone establish improvement over a baseline. SWE uses the recorded Inspect generous protocol, not a model-card reproduction.', '',
              f'[Detailed counts and provenance]({a.benchmark}.json).', '',
              '[Reproduction](../../RUNNING.md)']
    (output/f'{a.benchmark}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(benchmark=a.benchmark,task_metrics=task['metrics'],role_accuracy=score['accuracy'],coverage=f"{score['scored_trajectories']}/{expected}")))


if __name__=='__main__':
    main()
