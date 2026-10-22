"""Inspect adapter for AppWorld's process-isolated interactive Python worlds."""
import asyncio
import json
from pathlib import Path
import time

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score, accuracy, scorer
from inspect_ai.solver import solver
from inspect_ai.tool import tool

PROMPT = """You are an autonomous assistant completing tasks in a simulated world of apps.
Use the python tool to interact with the apps through the provided `apis` object.
Python variables persist between calls. Print values to inspect results. Discover apps and APIs with:
print(apis.api_docs.show_app_descriptions())
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))
Read API documentation before using an API. Use the supervisor app to obtain account credentials,
addresses and payment cards. References to files mean the file_system app, never the operating system.
Only use the provided APIs to interact with apps. Inspect all pages of paginated results.
Use the world's current date, not your prior knowledge; temporal requests use full day boundaries.
References to friends and family refer to the phone contacts. Work in small Python steps and verify results.
Once done, call apis.supervisor.complete_task(). For information requests, pass answer=<answer>,
using the requested entity or number rather than a full sentence. You may use status='fail' if unable
to complete the task. Make decisions autonomously without asking for clarification.
"""


class WorldWorker:
    def __init__(self, image, data, outputs, task_id, experiment, oracle=False):
        self.output = Path(outputs) / task_id
        self.output.mkdir(parents=True, exist_ok=True)
        worker = Path(__file__).resolve().parents[2] / 'experiments/agentic_eval/appworld_worker.py'
        self.command = ['apptainer', 'exec', '--containall', '--cleanenv', '--no-mount',
                        'hostfs,bind-paths', '--net', '--network', 'none', '--pwd', '/run',
                        '--bind', f'{Path(data).resolve()}:/run/data:ro',
                        '--bind', f'{self.output.resolve()}:/run/experiments/outputs:rw',
                        '--bind', f'{worker}:/worker.py:ro', '--env', 'APPWORLD_ROOT=/run',
                        str(Path(image).resolve()), '/opt/venv/bin/python', '/worker.py', task_id, experiment]
        if oracle:
            self.command.append('--oracle')

    async def __aenter__(self):
        self.stderr = (self.output / 'worker.log').open('w')
        self.process = await asyncio.create_subprocess_exec(*self.command,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.stderr,
            limit=10 * 1024 * 1024)
        self.info = await self.read()
        return self

    async def read(self):
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout=180)
        if not line:
            raise RuntimeError(f'AppWorld worker exited; see {self.output / "worker.log"}')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    async def request(self, method, **kwargs):
        self.process.stdin.write((json.dumps({'method': method, **kwargs})+'\n').encode())
        await self.process.stdin.drain()
        return await self.read()

    async def __aexit__(self, *exc):
        self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=15)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()
        self.stderr.close()


@scorer(metrics=[accuracy()])
def appworld_success():
    async def score(state, target):
        return Score(value='C' if state.metadata['appworld']['success'] else 'I')
    return score


def appworld_task(task_ids, split, image, data, outputs, experiment, max_steps=50):
    @solver
    def run_world():
        async def solve(state, generate):
            started = time.perf_counter()
            async with WorldWorker(image, data, outputs, str(state.sample_id), experiment) as world:
                loaded = time.perf_counter()
                steps = 0
                tool_seconds = 0.0
                completed = False
                @tool
                def python():
                    async def execute(code: str) -> str:
                        """Execute Python in the stateful AppWorld environment.

                        Args:
                            code: Python code using the provided apis object.
                        """
                        nonlocal steps, tool_seconds, completed
                        before = time.perf_counter()
                        result = await world.request('execute', code=code)
                        tool_seconds += time.perf_counter() - before
                        steps += 1
                        completed = result['completed']
                        if completed or steps >= max_steps:
                            state.completed = True
                        return result['output']
                    return execute
                state.messages = [ChatMessageSystem(content=PROMPT), ChatMessageUser(content=
                    'Supervisor: '+json.dumps(world.info['supervisor'])+'\n\nTask: '+world.info['instruction'])]
                state.tools = [python()]
                await generate(state, tool_calls='loop')
                result = await world.request('evaluate')
                state.metadata['appworld'] = {**result, 'split': split, 'steps': steps,
                    'completed': completed, 'step_limit': steps >= max_steps and not completed,
                    'load_seconds': loaded-started, 'tool_seconds': tool_seconds,
                    'rollout_seconds': time.perf_counter()-loaded,
                    'total_seconds': time.perf_counter()-started}
            return state
        return solve
    return Task(name='appworld_'+split, dataset=[Sample(id=i, input=i, metadata={}) for i in task_ids],
                solver=run_world(), scorer=appworld_success(), message_limit=2*max_steps+4)
