"""Versioned SWE sandbox fix for Inspect's hard-coded Python 3 tool command.

Some pinned repositories require Python 2. Their testbed has no python3 binary,
so Inspect's default command falls through to an unrelated base interpreter.
Keep the agent tool schema unchanged but use the activated repository interpreter.
The original provider remains unchanged for reproducibility of saved cohorts.
"""
from inspect_ai.util import SandboxEnvironmentSpec, sandboxenv
from .swe_testbed import TestbedSandbox, sandbox_config as previous_config


@sandboxenv(name='apptainer_testbed_python')
class RepositoryPythonSandbox(TestbedSandbox):
    async def exec(self, cmd, **kwargs):
        if cmd == ['bash', '--login', '-c', 'python3 -']:
            cmd = ['python', '-']
        return await super().exec(cmd, **kwargs)


def sandbox_config(kind, sample):
    original = previous_config(kind, sample)
    return SandboxEnvironmentSpec(type='apptainer_testbed_python', config=original.config)
