"""Isolated missing-dependency control for Pylint gold validation."""
import os
import shutil
from pathlib import Path
from inspect_ai.util import SandboxEnvironmentSpec, sandboxenv
from .swe_snapshot import SnapshotSandbox
from .swe_testbed_python import RepositoryPythonSandbox
from .apptainer_mirror import sandbox_config as pinned_config

@sandboxenv(name="apptainer_appdirs")
class AppdirsSandbox(SnapshotSandbox):
    @classmethod
    async def sample_init(cls, task_name, config, metadata):
        environments = await super().sample_init(task_name, config, metadata)
        try:
            if metadata['repo'] == 'pylint-dev/pylint':
                for box in environments.values():
                    target = box.directory / 'rootfs/opt/miniconda3/envs/testbed/lib/python3.9/site-packages'
                    assert target.is_dir()
                    for source in Path(os.environ['SWE_APPDIRS_FIXTURE']).iterdir():
                        destination = target / source.name
                        if destination.exists():
                            raise RuntimeError('Refusing to overwrite dependency: ' + str(destination))
                        if source.is_dir():
                            shutil.copytree(source, destination)
                        else:
                            shutil.copy2(source, destination)
            return environments
        except BaseException:
            await cls.sample_cleanup(task_name, config, environments, True)
            raise

@sandboxenv(name='apptainer_repository_appdirs')
class RepositoryAppdirsSandbox(RepositoryPythonSandbox, AppdirsSandbox):
    pass

def sandbox_config(kind, sample):
    original = pinned_config(kind, sample)
    return SandboxEnvironmentSpec(type='apptainer_repository_appdirs', config=original.config)
