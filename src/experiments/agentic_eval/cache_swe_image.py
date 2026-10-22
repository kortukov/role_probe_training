"""Prepare selected pinned SWE images on a CPU node before allocating GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('sample_id', nargs='+')
    parser.add_argument('--output', type=Path, default=Path('.cache/swe-images'))
    args = parser.parse_args()
    images = json.loads(Path('experiment_reports/agentic/swe_bench_full/image_manifest.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for sample_id in args.sample_id:
        source = images[sample_id]['image']
        target = args.output / f'{sample_id}.sif'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        threads = os.environ.get('SLURM_CPUS_PER_TASK', '1')
        subprocess.run(['apptainer', 'build', '--mksquashfs-args',
                        f'-comp zstd -Xcompression-level 1 -processors {int(threads)}',
                        str(target), 'docker://' + source], check=True, timeout=1800)
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        target.with_suffix('.json').write_text(json.dumps({'source_image': source, 'sha256': digest}, indent=2)+'\n')
        print(sample_id, digest)


if __name__ == '__main__':
    main()
