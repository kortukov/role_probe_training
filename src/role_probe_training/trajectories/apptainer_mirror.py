"""Same sandbox/harness, with digest-pinned official images from Google's cache."""
import json
import hashlib
import os
from pathlib import Path
from inspect_ai.util import SandboxEnvironmentSpec
from .apptainer import ApptainerSandbox  # register the validated provider

MANIFEST=Path('experiment_reports/agentic/swe_bench_full/image_manifest.json')
IMAGES=json.loads(MANIFEST.read_text())

def sandbox_config(kind,sample):
    image=IMAGES[str(sample.id)]
    if image['status']!=200 or not image['digest']:
        raise ValueError(f'Missing pinned image for {sample.id}')
    sample.metadata['original_image_name']=sample.metadata['image_name']
    sample.metadata['image_name']=image['image']
    sample.metadata['image_digest']=image['digest']
    source = image['image']
    if os.environ.get('SWE_IMAGE_CACHE'):
        local = Path(os.environ['SWE_IMAGE_CACHE']) / f'{sample.id}.sif'
        if local.exists():
            metadata = json.loads(local.with_suffix('.json').read_text())
            digest = hashlib.sha256()
            with local.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if metadata['source_image'] != source or metadata['sha256'] != digest.hexdigest():
                raise ValueError(f'Cached image provenance mismatch: {local}')
            sample.metadata['local_image_sha256'] = metadata['sha256']
            source = str(local.resolve())
    return SandboxEnvironmentSpec(type='apptainer',config=source)
