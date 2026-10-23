"""Prepare selected pinned SWE images on a CPU node before allocating GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('sample_id', nargs='*')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('.cache/swe-images'))
    args = parser.parse_args()
    images = json.loads(Path('experiment_reports/agentic/swe_bench_full/image_manifest.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('Invalid shard index/count')
    if args.all == bool(args.sample_id):
        parser.error('Choose --all or explicit sample IDs')
    selected = sorted(images) if args.all else args.sample_id
    for sample_id in selected[args.shard_index::args.shard_count]:
        source = images[sample_id]['image']
        target = args.output / f'{sample_id}.sif'
        if target.exists():
            metadata = json.loads(target.with_suffix('.json').read_text())
            with target.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if metadata != {'source_image': source, 'sha256': digest}:
                raise ValueError(f'Cached image provenance mismatch: {target}')
            print(sample_id, 'verified existing cache', flush=True)
            continue
        temporary = target.with_suffix('.partial.sif')
        threads = os.environ.get('SLURM_CPUS_PER_TASK', '1')
        subprocess.run(['apptainer', 'build', '--mksquashfs-args',
                        f'-comp zstd -Xcompression-level 1 -processors {int(threads)}',
                        str(temporary), 'docker://' + source], check=True, timeout=1800)
        temporary.replace(target)
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        target.with_suffix('.json').write_text(json.dumps({'source_image': source, 'sha256': digest}, indent=2)+'\n')
        print(sample_id, digest)


if __name__ == '__main__':
    main()
