"""Real HTTP/Inspect context-stop test, without GPU generation or truncation."""
import json,subprocess,tempfile,urllib.request
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from role_probe_training.on_policy_recovery.serving import serve

t=AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b',local_files_only=True)
model=SimpleNamespace(gradient_checkpointing_disable=lambda:None,gradient_checkpointing_enable=lambda **kwargs:None)
runtime=SimpleNamespace(tokenizer=t,device='cpu',hidden=None,model=model)
class Sampler:
    def sample(self,*args):raise AssertionError('No sampling at context exhaustion')
    def parse_completion(self,*args):raise AssertionError('No invented model completion')

with tempfile.TemporaryDirectory(prefix='context-check-') as tmp:
    with serve(runtime,tmp,0,dict(context_limit=1,max_tokens=8192),sampler=Sampler().sample) as (url,state):
        def post(route,body):
            r=urllib.request.Request(url+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            return json.load(urllib.request.urlopen(r))
        for name in ['one','two']:
            post('/episode',{'id':name})
            result=post('/v1/chat/completions',dict(model='test',messages=[{'role':'user','content':'Test'}],seed=123))
            assert result['choices'][0]['finish_reason']=='stop'
            assert result['usage']['completion_tokens']==0
            assert result['usage']['prompt_tokens']>1
        assert not state['errors']
        assert not (Path(tmp)/'requests.jsonl').exists()
        # Exercise the same Inspect generate(tool_calls='loop') path, not just HTTP.
        post('/episode',{'id':'inspect'})
        script=Path('src/experiments/training_against_role_probes/on_policy_recovery/check_inspect_stop.py')
        subprocess.run(['.tools/bin/uv','run','--project','src/experiments/eval_role_probes/agentic_eval/swe_bench/env','--frozen','--no-sync','python',str(script),url],check=True)
    events=[json.loads(x) for x in (Path(tmp)/'episode_stops.jsonl').read_text().splitlines()]
    assert len(events)==3 and all(e['reason']=='serving_context_limit' for e in events)
print('Context limit: no generation, no truncation, explicit stop metadata, next episode healthy, Inspect loop terminates.')
