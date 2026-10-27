"""Versioned SWE sandbox with localhost resolution and unchanged network isolation."""
import base64
from inspect_ai.util import SandboxEnvironmentSpec, sandboxenv
from .apptainer import ApptainerSandbox
from .apptainer_mirror import sandbox_config as pinned_config
from .swe_testbed_python import RepositoryPythonSandbox


@sandboxenv(name='apptainer_localhost')
class LocalhostSandbox(ApptainerSandbox):
    @classmethod
    async def sample_init(cls, task_name, config, metadata):
        environments = await super().sample_init(task_name, config, metadata)
        try:
            for box in environments.values():
                root = (box.directory / 'rootfs').resolve()
                hosts = (root / 'etc/hosts').resolve()
                if not hosts.is_relative_to(root):
                    raise RuntimeError('Container hosts file escapes its root')
                # Image-owned file only; never modify a host /etc/hosts mount.
                hosts.write_text('127.0.0.1 localhost\n::1 localhost ip6-localhost ip6-loopback\n')
                check = await ApptainerSandbox.exec(box, ['/opt/miniconda3/bin/python', '-c',
                    'import socket; assert socket.gethostbyname("localhost")=="127.0.0.1"; '
                    's=socket.socket(); s.bind(("localhost",0)); s.listen(1); '
                    'c=socket.socket(); c.settimeout(2); c.connect(s.getsockname()); print("localhost connected")'])
                if not check.success:
                    raise RuntimeError('Localhost control failed: ' + check.stderr)
            return environments
        except BaseException:
            await cls.sample_cleanup(task_name, config, environments, True)
            raise

    async def write_file(self, file, contents):
        payload = contents.encode() if isinstance(contents, str) else contents
        result = await ApptainerSandbox.exec(self, ['/opt/miniconda3/bin/python', '-c',
            'import pathlib,sys,base64; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(base64.b64decode(sys.stdin.buffer.read()))', file],
            input=base64.b64encode(payload).decode())
        if not result.success:
            raise RuntimeError(result.stderr)

    async def read_file(self, file, text=True):
        result = await ApptainerSandbox.exec(self, ['/opt/miniconda3/bin/python', '-c',
            'import pathlib,sys,base64; sys.stdout.write(base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode())', file])
        if not result.success:
            if 'FileNotFoundError' in result.stderr:
                raise FileNotFoundError(file)
            raise RuntimeError(result.stderr)
        raw = base64.b64decode(result.stdout)
        return raw.decode() if text else raw


@sandboxenv(name='apptainer_repository_localhost')
class RepositoryLocalhostSandbox(RepositoryPythonSandbox, LocalhostSandbox):
    """Repository Python for tools; base Python for sandbox file transfer."""


def sandbox_config(kind, sample):
    original = pinned_config(kind, sample)
    return SandboxEnvironmentSpec(type='apptainer_repository_localhost', config=original.config)
