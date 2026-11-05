"""Exercise real Inspect/OpenAI serialization with a deterministic CPU model."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import urllib.request

import torch
from transformers import AutoTokenizer
from role_probe_training.on_policy.serving import serve


def main():
    tokenizer = AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b', local_files_only=True)
    completion = tokenizer.encode('<|channel|>analysis<|message|>Check the sum.<|end|>'
        '<|start|>assistant<|channel|>final<|message|>Two.<|return|>', add_special_tokens=False)
    tool_completion = tokenizer.encode('<|channel|>analysis<|message|>Check with Python.<|end|>'
        '<|start|>assistant to=functions.python<|channel|>commentary json<|message|>'
        '{"code":"print(1+1)"}<|call|>', add_special_tokens=False)

    class Model:
        calls = 0
        def gradient_checkpointing_disable(self):
            pass

        def gradient_checkpointing_enable(self, **kwargs):
            pass

        def generate(self, input_ids, **kwargs):
            assert kwargs['top_k'] == 0 and kwargs['temperature'] == 1
            self.calls += 1
            tokens = tool_completion if self.calls == 1 else completion
            return torch.cat([input_ids, torch.tensor([tokens])], dim=1)

    runtime = SimpleNamespace(model=Model(), tokenizer=tokenizer, device='cpu',
                              pad_id=tokenizer.pad_token_id, hidden=None)
    with TemporaryDirectory() as directory, patch('torch.random.fork_rng', return_value=nullcontext()):
        with serve(runtime, directory, 7, {'max_tokens': 128, 'context_limit': 32768}) as (url, state):
            request = urllib.request.Request(url+'/episode', data=b'{"id":"test"}',
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request) as response:
                assert json.load(response)['ok']
            code = '''import asyncio, sys
from inspect_ai.model import get_model, GenerateConfig, ChatMessageUser, ChatMessageTool
from inspect_ai.tool import ToolInfo
async def main():
    model = get_model('vllm/on-policy-role-model', base_url=sys.argv[1]+'/v1',
                      api_key='local', stream=False,
                      config=GenerateConfig(temperature=1, top_p=1, seed=1,
                          max_tokens=128, reasoning_effort='high', max_retries=0))
    tools = [ToolInfo(name='python', description='Execute Python', parameters={
        'type':'object', 'properties':{'code':{'type':'string'}}, 'required':['code']})]
    user = ChatMessageUser(content='Use Python to check one plus one.')
    first = await model.generate([user], tools=tools)
    assert first.message.tool_calls[0].arguments == {'code':'print(1+1)'}
    answer = ChatMessageTool(content='2', tool_call_id=first.message.tool_calls[0].id, function='python')
    output = await model.generate([user, first.message, answer], tools=tools)
    assert output.completion == 'Two.', output
    assert any(b.type == 'reasoning' and b.reasoning == 'Check the sum.'
               for b in output.message.content), output
asyncio.run(main())
'''
            subprocess.run(['.tools/bin/uv', 'run', '--project', 'src/experiments/eval_role_probes/agentic_eval/swe_bench/env',
                '--frozen', '--no-sync', 'python', '-c', code, url], check=True, timeout=60)
            assert not state['errors']
        records = [json.loads(line) for line in (Path(directory)/'requests.jsonl').read_text().splitlines()]
        assert len(records) == 2 and all(x['policy_step'] == 7 for x in records)
        assert records[0]['completion_ids'] == tool_completion
        assert records[1]['completion_ids'] == completion
        assert records[0]['prompt_ids']
    print('PASS: real Inspect tool round trip, reasoning retained, context discovery and exact sampled-token capture.')


if __name__ == '__main__':
    main()
