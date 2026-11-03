"""Require two fresh updates and a successful checkpoint restart before scaling."""
import argparse
import json
import math
from pathlib import Path
from role_probe_training.on_policy.data import sha256, validate_requests

ROOT = Path('experiment_reports/training_against_role_probes/on_policy/appworld2700')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/epoch1')
    args=parser.parse_args()
    run=args.run
    manifest = json.loads((run/'manifest.json').read_text())
    config = manifest['config']
    schedule = json.loads((run/'schedule.json').read_text())
    records = []
    for rank in range(2):
        events = [json.loads(line) for line in (run/f'rank{rank}.jsonl').read_text().splitlines()]
        updates = [x for x in events if x['event'] == 'update']
        assert [x['step'] for x in updates] == [1, 2]
        assert any(x['event'] == 'ready' and x['start_step'] == 1 for x in events)
        assert events[-1]['event'] == 'pilot_completed' and events[-1]['updates'] == 2
        assert abs(updates[0]['reverse_kl']) < 1e-5
        for update in updates:
            assert update['refit_steps'] == 2000 and update['probe_training_tokens'] > 0
            assert update['collection_attempts'] == 16
            assert all(math.isfinite(update[key]) for key in
                       ('role_loss', 'reverse_kl', 'gradient_norm', 'probe_heldout_accuracy'))
            assert update['reverse_kl'] >= -1e-6 and update['gradient_norm'] > 0
            step = update['step']
            folder = run/'rollouts'/f'step{step:05d}'/f'rank{rank}'
            assert not (folder/'server_errors.json').exists()
            requests = [json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
            validate_requests(requests, schedule[step-1][rank::2], step-1)
            complete = json.loads((folder/'collection_complete.json').read_text())
            assert complete['attempts'] == 8 and complete['policy_step'] == step-1
        if config.get('serving_backend') == 'colocated_vllm':
            comparisons=[x for x in events if x['event']=='serving_distribution']
            assert [x['step'] for x in comparisons]==[1,2]
            for comparison in comparisons:
                value=comparison['comparison']
                assert value['policy_step']==comparison['step']-1
                assert -1e-5<=value['hf_to_vllm_kl']<0.1
                assert value['hf_argmax']==value['vllm_argmax']
        records.append(updates[-1])
    for path, digest in manifest['source_sha256'].items():
        assert sha256(path) == digest, f'Source changed: {path}'
    checkpoint = run/'checkpoints/step00002'
    assert json.loads((checkpoint/'layer_12.json').read_text())['model_update_step'] == 2
    assert sha256(checkpoint/'adapter/adapter_model.safetensors') != sha256(
        run/'checkpoints/step00001/adapter/adapter_model.safetensors')
    # Do not knowingly submit a production continuation that exceeds walltime.
    projected_hours = (169-2)*max(x['total_seconds'] for x in records)*1.2/3600
    result = dict(two_updates_validated=True, checkpoint_resume_validated=True,
        projected_remaining_hours=projected_hours, configuration=config)
    (run/'pilot_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    assert projected_hours < 68, 'Pilot throughput exceeds one allocation; plan segmented continuation'
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
