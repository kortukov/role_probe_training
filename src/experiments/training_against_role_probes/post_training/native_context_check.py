"""Verify the served limit and a real request above the old 32K ceiling."""
import argparse
import json
from openai import OpenAI
p=argparse.ArgumentParser(__doc__)
p.add_argument('--base-url',required=True)
p.add_argument('--model',required=True)
a=p.parse_args()
client=OpenAI(base_url=a.base_url,api_key='local',timeout=300,max_retries=0)
info=next(m for m in client.models.list().data if m.id==a.model)
assert info.model_dump().get('max_model_len')==131072,info
response=client.chat.completions.create(model=a.model,
    messages=[{'role':'user','content':'Read this data: '+ 'x '*40000+'\nReply OK.'}],
    max_tokens=1,temperature=0)
assert response.usage.prompt_tokens>32768,response.usage
print(json.dumps({'server_context_limit':131072,'prompt_tokens':response.usage.prompt_tokens,
                  'completion_tokens':response.usage.completion_tokens,'request_above_32k_passed':True}))
