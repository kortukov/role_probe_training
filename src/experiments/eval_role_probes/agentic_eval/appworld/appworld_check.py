"""CPU-only checks for parallel worlds and scoring after Inspect execution limits."""
import asyncio
from pathlib import Path
from inspect_ai import eval
from inspect_ai.model import get_model, ModelOutput, ChatCompletionChoice, ChatMessageAssistant
from inspect_ai.tool import ToolCall
from role_probe_training.trajectories.appworld import WorldWorker, appworld_task

ROOT=Path('.cache/appworld')
IDS=['82e2fac_1','82e2fac_2']

async def verify():
    async def one(task_id):
        async with WorldWorker(ROOT/'appworld.sif',ROOT/'data',ROOT/'verify',task_id,'oracle',True) as world:
            assert (await world.request('oracle'))['success']
    await asyncio.gather(*(one(i) for i in IDS))


def main():
    asyncio.run(verify())
    output=ModelOutput(model='mock',choices=[ChatCompletionChoice(stop_reason='tool_calls',
        message=ChatMessageAssistant(content='',tool_calls=[ToolCall('call1','python',{'code':'print(1)'})]))])
    task=appworld_task(IDS[:1],'train',ROOT/'appworld.sif',ROOT/'data',ROOT/'limit-check','limit-check')
    logs=eval(task,model=get_model('mockllm/model',custom_outputs=[output]*5),message_limit=3,
              log_dir=str(ROOT/'limit-check-logs'),display='plain')
    sample=logs[0].samples[0]
    assert sample.error is None, sample.error
    assert sample.scores['appworld_success'].value=='I'
    assert 'appworld' in sample.metadata
    assert sample.limit is not None
    print('PASS: parallel reference solutions and scoring after an Inspect message limit')

if __name__=='__main__':main()
