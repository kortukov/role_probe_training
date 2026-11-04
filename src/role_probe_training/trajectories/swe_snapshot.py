"""Diagnostic baseline snapshot excludes pre-existing image build files from patches."""
from inspect_ai.util import SandboxEnvironmentSpec, sandboxenv
from .apptainer_mirror import sandbox_config as pinned_config
from .swe_localhost import LocalhostSandbox
from .swe_testbed_python import RepositoryPythonSandbox


@sandboxenv(name='apptainer_snapshot')
class SnapshotSandbox(LocalhostSandbox):
    @classmethod
    async def sample_init(cls, task_name, config, metadata):
        environments = await super().sample_init(task_name, config, metadata)
        try:
            for box in environments.values():
                result = await box.exec(['bash', '-c',
                    'git add -A && git -c user.name=SWEEnvironment -c user.email=environment@localhost '
                    'commit --allow-empty --no-verify -m "Record initial image files before model edits"'], cwd='/testbed')
                if not result.success:
                    raise RuntimeError('Initial image snapshot failed: ' + result.stderr)
            return environments
        except BaseException:
            await cls.sample_cleanup(task_name, config, environments, True)
            raise


@sandboxenv(name='apptainer_repository_snapshot')
class RepositorySnapshotSandbox(RepositoryPythonSandbox, SnapshotSandbox):
    """Repository interpreter plus explicit initial image snapshot."""


def sandbox_config(kind, sample):
    original = pinned_config(kind, sample)
    return SandboxEnvironmentSpec(type='apptainer_repository_snapshot', config=original.config)
