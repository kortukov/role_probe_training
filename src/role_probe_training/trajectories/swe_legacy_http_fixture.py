"""Opt-in local HTTP fixture for Requests diagnostics; no external networking."""
import asyncio
import os
from pathlib import Path
import shutil
from inspect_ai.util import SandboxEnvironmentSpec, sandboxenv
from .apptainer import ApptainerSandbox
from .apptainer_mirror import sandbox_config as pinned_config
from .swe_snapshot import SnapshotSandbox as LocalhostSandbox
from .swe_testbed_python import RepositoryPythonSandbox


@sandboxenv(name='apptainer_legacy_http_fixture')
class LegacyHttpFixtureSandbox(LocalhostSandbox):
    @classmethod
    async def sample_init(cls, task_name, config, metadata):
        environments = await super().sample_init(task_name, config, metadata)
        if metadata['repo'] != 'psf/requests':
            return environments
        try:
            for box in environments.values():
                source = Path(os.environ['SWE_HTTP_FIXTURE'])
                destination = box.directory / 'rootfs/opt/swe-http-fixture'
                destination.mkdir(parents=True, exist_ok=False)
                for version in ['py311', 'py39']:
                    shutil.copytree(source / version, destination / version)
                script = destination / 'server.py'
                script.write_text('from httpbin import app\nfrom wsgiref.simple_server import make_server\nmake_server("127.0.0.1", 8765, app).serve_forever()\n')
                result = await ApptainerSandbox.exec(box, ['bash', '-c',
                    'nohup /opt/miniconda3/bin/python /opt/swe-http-fixture/server.py >/tmp/swe-http-fixture.log 2>&1 </dev/null &'],
                    env={'PYTHONPATH': '/opt/swe-http-fixture/py311'})
                if not result.success:
                    raise RuntimeError(result.stderr)
                for _ in range(30):
                    ready = await ApptainerSandbox.exec(box, ['/opt/miniconda3/bin/python', '-c',
                        'import urllib.request; assert urllib.request.urlopen("http://127.0.0.1:8765/get",timeout=1).status==200'])
                    if ready.success:
                        break
                    await asyncio.sleep(1)
                else:
                    log = await ApptainerSandbox.exec(box, ['cat', '/tmp/swe-http-fixture.log'])
                    raise RuntimeError('Local HTTP fixture did not start: ' + log.stdout + log.stderr)
                version = await ApptainerSandbox.exec(box, ['bash', '-c',
                    'source /opt/miniconda3/bin/activate testbed; python -c "import sys; print(str(sys.version_info[0])+\'.\'+str(sys.version_info[1]))"'])
                if not version.success:
                    raise RuntimeError(version.stderr)
                box.fixture_environment = {'HTTPBIN_URL': 'http://127.0.0.1:8765/'}
                if version.stdout.strip() == '3.9':
                    # Put the repository first: fixture dependencies must never shadow Requests.
                    box.fixture_environment['PYTHONPATH'] = '/testbed:/opt/swe-http-fixture/py39'
                elif version.stdout.strip() != '2.7':
                    raise RuntimeError('Unvalidated Requests interpreter: ' + version.stdout)
            return environments
        except BaseException:
            await cls.sample_cleanup(task_name, config, environments, True)
            raise

    async def exec(self, cmd, env=None, **kwargs):
        merged = {**getattr(self, 'fixture_environment', {}), **(env or {})}
        return await super().exec(cmd, env=merged, **kwargs)


@sandboxenv(name='apptainer_repository_legacy_http_fixture')
class RepositoryLegacyHttpFixtureSandbox(RepositoryPythonSandbox, LegacyHttpFixtureSandbox):
    """Repository tool interpreter with pinned local HTTP fixtures."""


def sandbox_config(kind, sample):
    original = pinned_config(kind, sample)
    return SandboxEnvironmentSpec(type='apptainer_repository_legacy_http_fixture', config=original.config)
