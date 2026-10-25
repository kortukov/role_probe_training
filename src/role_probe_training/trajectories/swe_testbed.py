"""Opt-in SWE sandbox that activates repository dependencies for every tool call.

Kept separate from the original provider so running experiments retain their
recorded environment. Use only for new, explicitly labeled evaluations.
"""
from inspect_ai.util import SandboxEnvironmentSpec, sandboxenv
from .apptainer import ApptainerSandbox
from .apptainer_mirror import sandbox_config as original_config


@sandboxenv(name='apptainer_testbed')
class TestbedSandbox(ApptainerSandbox):
    async def exec(self, cmd, **kwargs):
        # Activation is scoped to the command; it also reaches child processes
        # started by Inspect's persistent shell and Python tools.
        return await super().exec(
            ['bash', '-c', 'source /opt/miniconda3/bin/activate testbed && exec "$@"',
             'bash', *cmd], **kwargs)


def sandbox_config(kind, sample):
    original = original_config(kind, sample)
    return SandboxEnvironmentSpec(type='apptainer_testbed', config=original.config)
