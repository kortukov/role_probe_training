"""Collect one evaluated checkpoint's task success and role accuracy into eval/results.json."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    result = dict(name=args.name, checkpoint=args.checkpoint,
                  probe_fit={layer: values['accuracy'] for layer, values in
                             json.loads((args.root/'post_training/probe_fit.json').read_text())['layers'].items()},
                  benchmarks={})
    for benchmark in ('appworld_test', 'tau2_retail'):
        summary = json.loads((args.root/'runs'/benchmark/'summary.json').read_text())
        (scorer, task), = summary['metrics'].items()
        roles = json.loads((args.root/'predictions'/benchmark/'summary.json').read_text())
        result['benchmarks'][benchmark] = dict(
            tasks=summary['samples'], successes=task['successes'],
            task_success=task['success_rate_observed_errors_as_failures'],
            sample_errors=summary['sample_errors'], generation_token_caps=summary['generation_token_caps'],
            scored_histories=roles['scored_trajectories'], excluded_histories=len(roles['excluded_trajectories']),
            roles=roles['roles'],
            role_accuracy={layer: dict(overall=values['accuracy'], balanced=values['macro_recall'],
                                       per_role=values['per_role_recall']) for layer, values in roles['layers'].items()})
    (args.root/'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
