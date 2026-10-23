"""Workload arithmetic, not measured throughput or a GPU memory benchmark."""
import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-tokens-per-second-per-gpu', type=float, default=2000)
    p.add_argument('--extraction-tokens-per-second-per-gpu', type=float, default=8000)
    args = p.parse_args()
    stats = json.loads(args.stats.read_text())
    train_rate = args.training_tokens_per_second_per_gpu
    extract_rate = args.extraction_tokens_per_second_per_gpu
    assert train_rate > 0 and extract_rate > 0
    steps = math.ceil(27000 / 16)
    result = {
        'status': 'illustrative throughput assumptions, not measured GPU performance',
        'training_trajectories': 27000, 'effective_batch': 16, 'epoch_updates': steps,
        'training_tokens_per_second_per_gpu': train_rate,
        'extraction_tokens_per_second_per_gpu': extract_rate,
        'scaling_assumption': 'ideal data parallelism, no padding',
        'kl_tokens_per_update_upper_bound': 16 * 2048,
        'excluded': ['probe optimizer time', 'validation', 'checkpoint IO',
                     'communication inefficiency', 'initial model load', 'trajectory generation'],
        'variants': {},
    }
    for variant in ('4096', '8192', '16384', 'full'):
        mean = stats['mean_tokens'] if variant == 'full' else stats['prefixes'][variant]['mean_tokens']
        max_length = stats['max_tokens'] if variant == 'full' else int(variant)
        tokens = 27000 * mean
        row = {'estimated_epoch_tokens': tokens,
               'max_trajectory_tokens': max_length,
               'bf16_all_token_layer12_cache_gib': tokens * 2880 * 2 / 2**30,
               'checkpoint_layer_inputs_gib': 24 * max_length * 2880 * 2 / 2**30,
               'one_eager_attention_score_tensor_gib': 64 * max_length**2 * 2 / 2**30,
               'gpus': {}}
        for gpus in (2, 4):
            update_h = tokens / (gpus * train_rate) / 3600
            kl_h = steps * 16 * 2048 * (1/train_rate + 1/extract_rate) / gpus / 3600
            refit_h = {str(pool): (steps+1) * pool * mean / (gpus * extract_rate) / 3600
                       for pool in (300, 27000)}
            row['gpus'][str(gpus)] = {
                'role_model_update_hours': update_h,
                'kl_hours_upper_length_assumption': kl_h,
                'refit_extraction_hours_by_pool_size': refit_h,
                'total_hours_with_hypothetical_300_trajectory_fit_pool': update_h + kl_h + refit_h['300'],
                'total_hours_with_full_27000_trajectory_fit_pool': update_h + kl_h + refit_h['27000']}
        result['variants'][variant] = row
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
