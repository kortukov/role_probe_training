"""Same sandbox/harness, with digest-pinned official images from Google's cache."""
import json
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
    return SandboxEnvironmentSpec(type='apptainer',config=image['image'])
