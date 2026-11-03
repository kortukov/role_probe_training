import sys,tempfile
from inspect_ai import Task,eval
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model,GenerateConfig
from inspect_ai.solver import generate
model=get_model('vllm/context-check',base_url=sys.argv[1]+'/v1',api_key='local',stream=False,config=GenerateConfig(seed=123,temperature=1,top_p=1,max_retries=0))
with tempfile.TemporaryDirectory() as logs:
    result=eval(Task(dataset=[Sample(input='Test')],solver=generate(tool_calls='loop')),model=model,log_dir=logs,display='none')[0]
    assert result.status=='success' and not result.samples[0].error
