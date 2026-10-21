"""Inspect sandbox backed by an isolated, writable Apptainer instance.

Images are unpacked into job-local scratch and removed after each sample.
No host project/home mounts or network access enter the instance.
"""
import base64
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from inspect_ai.util import SandboxEnvironment, SandboxEnvironmentSpec, sandboxenv, subprocess
from inspect_ai.util._sandbox.environment import SandboxUserUnsupportedError


def sandbox_config(kind, sample):
    return SandboxEnvironmentSpec(type="apptainer", config=sample.metadata["image_name"])


@sandboxenv(name="apptainer")
class ApptainerSandbox(SandboxEnvironment):
    def __init__(self, directory, instance):
        super().__init__()
        self.directory, self.instance = directory, instance

    @classmethod
    async def sample_init(cls, task_name, config, metadata):
        directory = Path(tempfile.mkdtemp(prefix="swe-", dir=os.environ.get("TMPDIR")))
        instance = "swe-" + uuid.uuid4().hex[:16]
        obj = cls(directory, instance)
        try:
            result = await subprocess(["apptainer", "build", "--fix-perms", "--sandbox", str(directory / "rootfs"), "docker://" + str(config)], timeout=1800)
            if not result.success:
                raise RuntimeError("Image preparation failed: " + result.stderr[-4000:])
            (directory / "image-build.log").write_text(result.stdout + result.stderr)
            result = await subprocess(["apptainer", "instance", "start", "--writable", "--containall", "--cleanenv", "--no-mount", "hostfs,bind-paths", "--net", "--network", "none", str(directory / "rootfs"), instance], timeout=120)
            if not result.success:
                raise RuntimeError("Sandbox startup failed: " + result.stderr)
            check = await obj.exec(["bash", "-c", 'test -d /testbed/.git && test ! -e /data/cluster/users/kortukov/role_probe_training && test -z "$AZURE_OPENAI_API_KEY" && git rev-parse HEAD'])
            if not check.success:
                raise RuntimeError("Sandbox isolation/repository check failed: " + check.stderr)
            return {"default": obj}
        except BaseException:
            await cls.sample_cleanup(task_name, config, {"default": obj}, True)
            raise

    @classmethod
    async def sample_cleanup(cls, task_name, config, environments, interrupted):
        for obj in environments.values():
            await subprocess(["apptainer", "instance", "stop", "--force", obj.instance], timeout=60)
            shutil.rmtree(obj.directory, ignore_errors=True)

    async def exec(self, cmd, input=None, cwd=None, env=None, user=None, timeout=None, timeout_retry=True, concurrency=True):
        if user is not None and user not in (str(os.getuid()), os.environ.get("USER")):
            raise SandboxUserUnsupportedError("Apptainer runs as the unprivileged job user")
        args = ["apptainer", "exec", "--cleanenv", "--pwd", cwd or "/testbed"]
        for key, value in (env or {}).items():
            args += ["--env", f"{key}={value}"]
        args += ["instance://" + self.instance]
        if timeout:
            args += ["timeout", "--signal=TERM", "--kill-after=5", str(timeout)]
        result = await subprocess(args + cmd, input=input, timeout=(timeout + 15 if timeout else 1800), output_limit=10 * 1024 * 1024, concurrency=concurrency)
        if result.returncode == 124:
            raise TimeoutError("Sandbox command exceeded timeout")
        return result

    async def write_file(self, file, contents):
        payload = contents.encode() if isinstance(contents, str) else contents
        result = await self.exec(["python", "-c", "import pathlib,sys,base64; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(base64.b64decode(sys.stdin.buffer.read()))", file], input=base64.b64encode(payload).decode())
        if not result.success:
            raise RuntimeError(result.stderr)

    async def read_file(self, file, text=True):
        result = await self.exec(["python", "-c", "import pathlib,sys,base64; sys.stdout.write(base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode())", file])
        if not result.success:
            if "FileNotFoundError" in result.stderr:
                raise FileNotFoundError(file)
            raise RuntimeError(result.stderr)
        raw = base64.b64decode(result.stdout)
        return raw.decode() if text else raw
