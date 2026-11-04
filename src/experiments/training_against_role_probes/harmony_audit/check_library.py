"""Read-only parser audit of retained on-policy failures; no generation."""
import importlib.metadata
import json
from pathlib import Path
from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Role

enc=load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
versions={}
for name in ('openai-harmony','oss-harmony','vllm'):
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: pass
results=[]
for batch,step,rank in ((64,1,0),(16,4,1)):
    path=Path(f'experiment_reports/training_against_role_probes/on_policy/appworld270_batch{batch}/runs/epoch1_repaired/rollouts/step{step:05d}/rank{rank}/requests.jsonl')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    for label,row in (('ordinary_control',rows[0]),('failure',rows[-1])):
        for strict in (True,False):
            result=dict(batch=batch,case=label,strict=strict,source=str(path),episode=row['episode_id'],generation_index=row['generation_index'])
            try:
                messages=enc.parse_messages_from_completion_tokens(row['completion_ids'],Role.ASSISTANT,strict=strict)
                result['messages']=[dict(role=str(m.author.role),channel=m.channel,recipient=m.recipient,content_type=m.content_type) for m in messages]
            except Exception as error:
                result['error']=f'{type(error).__name__}: {error}'
            results.append(result)
print(json.dumps(dict(versions=versions,results=results),indent=2))
