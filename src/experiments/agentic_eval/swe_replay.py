"""Replay failed SWE API requests without executing tools; diagnostic, not task scores."""
import argparse
import asyncio
import json
from pathlib import Path
import time
from inspect_ai.log import read_eval_log
from openai import AsyncOpenAI
from jsonschema import Draft202012Validator

async def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--base-url',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=4)
    a=p.parse_args()
    root=Path('experiment_reports/agentic/swe_bench_full/runs/1461197/logs')
    log=read_eval_log(next(root.glob('*.eval')),resolve_attachments='full')
    requests=[]
    for sample in sorted(log.samples,key=lambda s:str(s.id)):
        for event in sample.events:
            if event.event=='model' and event.error and 'unexpected tokens remaining' in event.error:
                request=dict(event.call.request)
                request.pop('extra_headers',None)
                requests.append((sample.id,request))
                break
        if len(requests)>=a.samples:break
    client=AsyncOpenAI(base_url=a.base_url,api_key='local',max_retries=0,timeout=300)
    async def run(i,request,temperature):
        started=time.perf_counter()
        try:
            r=await client.chat.completions.create(**(request|{'temperature':temperature}))
            validation_errors=[]
            schemas={t['function']['name']:t['function']['parameters'] for t in request['tools']}
            for call in r.choices[0].message.tool_calls or []:
                if call.function.name not in schemas:
                    validation_errors.append('Unknown function: '+call.function.name)
                    continue
                try:
                    arguments=json.loads(call.function.arguments)
                    validation_errors.extend(error.message for error in Draft202012Validator(schemas[call.function.name]).iter_errors(arguments))
                except json.JSONDecodeError:
                    validation_errors.append('Invalid JSON arguments')
            result={'error':None,'tool_validation_errors':validation_errors,'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump(),
                    'tool_calls':len(r.choices[0].message.tool_calls or [])}
        except Exception as exc:
            result={'error':str(exc)}
        return {'sample_id':i,'temperature':temperature,'seconds':time.perf_counter()-started,**result}
    results=await asyncio.gather(*(run(i,r,t) for i,r in requests for t in [0,1]))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='error'}|{'error':bool(r['error'])} for r in results]))

if __name__=='__main__':asyncio.run(main())
