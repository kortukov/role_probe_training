"""Compare Python in a cached image with a fresh writable evaluation sandbox."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import tempfile
from inspect_ai.util import subprocess
from role_probe_training.trajectories.apptainer import ApptainerSandbox

async def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--sample-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    image=Path('.cache/swe-images')/(args.sample_id+'.sif')
    metadata=json.loads(image.with_suffix('.json').read_text())
    check=['python','-c','import sys,encodings,pyexpat; print(sys.executable); print("Python runtime healthy")']
    result={}
    with tempfile.TemporaryDirectory(prefix='swe-runtime-check-',dir='/tmp') as scratch:
        os.environ['TMPDIR']=scratch
        direct=await subprocess(['apptainer','exec','--containall','--cleanenv','--no-mount','hostfs,bind-paths','--pwd','/testbed',str(image.resolve()),*check],timeout=120)
        result['image']={'success':direct.success,'stdout':direct.stdout,'stderr':direct.stderr}
        environments=await ApptainerSandbox.sample_init('runtime-check',str(image.resolve()),{'local_image_sha256':metadata['sha256']})
        try:
            box=environments['default']
            probe=await box.exec(check)
            result['writable_sandbox']={'success':probe.success,'stdout':probe.stdout,'stderr':probe.stderr}
            result['build_log']=(box.directory/'image-build.log').read_text()[-12000:]
        finally:
            await ApptainerSandbox.sample_cleanup('runtime-check',str(image),environments,False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='build_log'}))
    assert result['image']['success'] and result['writable_sandbox']['success']

asyncio.run(main())
