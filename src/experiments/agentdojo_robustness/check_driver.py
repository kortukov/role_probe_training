"""CPU integration check: real benchmark tasks with a mocked inference boundary."""
import runpy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from openai.types.chat import ChatCompletion

calls = []
def complete(self, **kwargs):
    assert kwargs['model'] == 'role-agentdojo'
    assert kwargs['max_tokens'] == 8192
    assert kwargs['reasoning_effort'] == 'medium'
    assert kwargs['temperature'] == 1.0
    calls.append(kwargs)
    return ChatCompletion(id='mock', created=0, model='role-agentdojo', object='chat.completion',
        choices=[{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'I cannot complete this task.', 'tool_calls': [{'id':'bad-call','type':'function','function':{'name':'get_balance','arguments':'broken json'}}] if len(calls)==1 else None}}])

with tempfile.TemporaryDirectory() as tmp:
    sys.argv = ['run.py', '--port', '1', '--suite', 'banking', '--output', tmp, '--smoke']
    with patch('openai.resources.chat.completions.Completions.create', complete):
        runpy.run_path('src/experiments/agentdojo_robustness/run.py', run_name='__main__')
    assert (Path(tmp)/'banking.complete.json').exists()
    files = list((Path(tmp)/'vllm_parsed').glob('*/*/*/*.json'))
    assert len(files) == 3, files
    assert len(calls) == 4
    assert (Path(tmp)/'banking.invalid_calls.jsonl').exists()
    print(f'PASS: clean, seed, attacked trajectories saved; {len(calls)} requests validated')
