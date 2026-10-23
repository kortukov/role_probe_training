"""Validate paired two-replica microbatch runs and save small timing summaries."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    results, reference = [], None
    for microbatch in (1, 2, 4):
        ranks = [[json.loads(line) for line in
                  (args.runs / str(microbatch) / f'rank{rank}.jsonl').read_text().splitlines()]
                 for rank in (0, 1)]
        assert all(any(r['event'] == 'completed' for r in rows) for rows in ranks)
        assert all(any(r['event'] == 'padding_check' and r['passed'] for r in rows) for rows in ranks)
        updates = [[r for r in rows if r['event'] == 'update'] for rows in ranks]
        assert len(updates[0]) == len(updates[1]) == 5
        signature = [(r['trajectory_ids'], r['tokens'], r['labeled_tokens']) for r in updates[0]]
        if reference is None:
            reference = signature
        assert signature == reference, 'Microbatch conditions used different data'
        times = []
        for left, right in zip(*updates):
            assert left['step'] == right['step']
            assert left['role_loss'] == right['role_loss']
            if not left['warmup']:
                times.append(max(left['seconds'], right['seconds']))
        measured = [r for r in updates[0] if not r['warmup']]
        allocated = max(r['peak_allocated_gib'] for rows in updates for r in rows)
        stress = [r for rows in ranks for r in rows if r['event'] == 'longest_backward']
        results.append(dict(
            microbatch_per_gpu=microbatch, accumulation_steps=8//microbatch,
            padding_checks=[r for rows in ranks for r in rows if r['event']=='padding_check'],
            update_seconds=times, mean_update_seconds=statistics.mean(times),
            median_update_seconds=statistics.median(times),
            peak_update_allocated_gib=allocated,
            peak_stress_allocated_gib=max(r['peak_allocated_gib'] for r in stress),
            stress_sequence_tokens=stress[0]['tokens'],
            padding_fraction=1-sum(r['tokens'] for r in measured)/sum(r['padded_tokens'] for r in measured),
            real_tokens_per_second=sum(r['tokens'] for r in measured)/sum(times),
            role_loss_by_update=[r['role_loss'] for r in updates[0]],
            role_accuracy_by_update=[r['role_accuracy'] for r in updates[0]],
        ))
    for row in results:
        row['speedup_vs_microbatch1'] = results[0]['mean_update_seconds']/row['mean_update_seconds']
    output = dict(job_id=args.runs.name, scope='role-only updates; fixed standard-data probe; no KL/refitting',
                  gpus=2, gpu='NVIDIA B200', effective_batch=16, context='full trajectories',
                  ordering='seed123 shuffle; sort by length within each batch16; alternate ranks',
                  warmup_updates=1, measured_updates=4, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
