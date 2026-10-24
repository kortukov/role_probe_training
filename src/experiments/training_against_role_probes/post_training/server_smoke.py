"""Check protocol and finite output, without requiring benchmark-like task success."""
import argparse
import json
from openai import OpenAI
p=argparse.ArgumentParser(__doc__);p.add_argument('--base-url',required=True);p.add_argument('--model',required=True)
a=p.parse_args(); client=OpenAI(base_url=a.base_url,api_key='local',max_retries=0,timeout=180)
r=client.chat.completions.create(model=a.model,messages=[{'role':'user','content':'Say hello briefly.'}],
    max_tokens=256,temperature=0,reasoning_effort='high',seed=123)
assert r.choices and r.usage and r.usage.completion_tokens>0
print(r.model_dump_json())
