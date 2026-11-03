"""CPU HTTP regression: invalid episode must not poison subsequent episodes."""
import json,tempfile,urllib.request
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from role_probe_training.on_policy_harmony.serving import serve

t=AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b',local_files_only=True)
fixtures=json.loads(Path('/tmp/onpolicy-parser-fixtures.json').read_text())
rows=[json.loads(x) for x in Path('experiment_reports/training_against_role_probes/on_policy/appworld270_batch64/runs/epoch1_repaired/rollouts/step00001/rank0/requests.jsonl').read_text().splitlines()]
class Sampler:
    def sample(self,*args):return self.tokens
    def parse_completion(self,*args):return self.parsed
sampler=Sampler()
runtime=SimpleNamespace(tokenizer=t,device='cpu',hidden=None,model=SimpleNamespace(gradient_checkpointing_disable=lambda:None,gradient_checkpointing_enable=lambda **kwargs:None))
with tempfile.TemporaryDirectory() as tmp:
    with serve(runtime,tmp,0,dict(context_limit=32768,max_tokens=8192),sampler=sampler.sample) as (url,state):
        def post(route,body):
            request=urllib.request.Request(url+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            try:return json.load(urllib.request.urlopen(request))
            except Exception:
                print(state['errors'])
                raise
        for index,kind in enumerate(('failure','control')):
            sampler.parsed=next(x for x in fixtures if x['case']=='batch64' and x['label']==kind)
            sampler.tokens=rows[-1 if index==0 else 0]['completion_ids']
            post('/episode',{'id':str(index)})
            response=post('/v1/chat/completions',dict(model='audit',messages=[{'role':'user','content':'Test'}],tools=[{'type':'function','function':{'name':'python','description':'Execute Python','parameters':{'type':'object','properties':{'code':{'type':'string'}}}}}],temperature=1,top_p=1,seed=123))
            choice=response['choices'][0]
            assert choice['finish_reason']==('stop' if index==0 else 'tool_calls')
        assert not state['errors']
    requests=[json.loads(x) for x in (Path(tmp)/'requests.jsonl').read_text().splitlines()]
    assert requests[0]['completion_ids']==rows[-1]['completion_ids']
    assert requests[1]['completion_ids']==rows[0]['completion_ids']
print('HTTP invalid-episode isolation, subsequent tool call, and raw-token retention passed.')
