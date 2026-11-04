"""Normalize saved AgentDojo logs; never generate or modify benchmark attempts."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai
from agentdojo.task_suite.load_suites import get_suite

ROOT = Path('experiment_reports/prompt_injection/agentdojo_same_batch_27000')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(messages):
    result, names = [], {}
    for message in messages:
        content = message.get('content')
        if isinstance(content, list):
            if any(b['type'] != 'text' for b in content):
                raise ValueError('Unsupported content block; do not discard it')
            content = ''.join(b['content'] or '' for b in content)
        out = {'role':message['role'], 'content':content or ''}
        for field in ('thinking','reasoning','reasoning_content'):
            if message.get(field):
                raise ValueError('New reasoning-bearing logs require explicit normalization')
        if message.get('tool_calls'):
            out['tool_calls'] = []
            for call in message['tool_calls']:
                names[call['id']] = call['function']
                out['tool_calls'].append({'id':call['id'], 'type':'function',
                    'function':{'name':call['function'], 'arguments':call['args']}})
        if message['role'] == 'tool':
            name = names.get(message.get('tool_call_id')) or (message.get('tool_call') or {}).get('function')
            if not name:
                raise ValueError('Tool result has no metadata-derived tool identity')
            out.update(name=name, tool_call_id=message.get('tool_call_id'))
            if message.get('error'):
                out['content'] = str(message['error'])
        result.append(out)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    a=parser.parse_args(); root=a.root
    archive=root/'data/historical_base.tar.gz'
    with tarfile.open(archive) as tar:
        keys=sorted(m.name.split('/vllm_parsed/')[1] for m in tar.getmembers()
            if m.isfile() and m.name.endswith('.json') and '/vllm_parsed/' in m.name)
    keys=[k for k in keys if k.split('/')[1].startswith('user_task_')
        and not (k.startswith('workspace/') and k.split('/')[1] in {'user_task_35','user_task_38'})]
    tools={s:[_function_to_openai(t) for t in get_suite('v1.2.2',s).tools]
           for s in ('workspace','slack','travel','banking')}
    manifest={'historical_archive_sha256':digest(archive), 'reasoning_policy':'Absent from saved logs; never synthesized',
              'replay_policy':'metadata-labelled retrospective Harmony replay; not exact serving-prompt recovery', 'models':{}}
    for model in ('base','trained'):
        failure={}
        for p in (root/'runs'/model).glob('*.failures.jsonl'):
            for line in p.read_text().splitlines():
                d=json.loads(line);failure[d['key']]=d['error']
        manifest['models'][model]={}
        for subset in ('clean','attacked'):
            selected=[k for k in keys if ('/none/none.json' in k)==(subset=='clean')]
            records=[]; missing=[]; sources={}; partial=[]
            for key in selected:
                p=root/'runs'/model/'vllm_parsed'/key
                if not p.exists():
                    missing.append({'trajectory_id':key, 'reason':failure.get(key,'missing_saved_log')});continue
                data=json.loads(p.read_text());sources[key]=digest(p)
                if not data.get('messages'):
                    missing.append({'trajectory_id':key,'reason':'empty_saved_history'});continue
                if data.get('utility') is None:
                    partial.append(key)
                records.append({'sample_id':key, 'trajectory_id':key, 'messages':normalize(data['messages']),
                    'tools':tools[key.split('/')[0]], 'source_sha256':sources[key],
                    'task_verdict_available':data.get('utility') is not None})
            out=root/'data/role_perception'/model/subset
            out.mkdir(parents=True,exist_ok=True)
            path=out/'trajectories.jsonl'
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            manifest['models'][model][subset]={'expected':len(selected),'saved':len(records),'missing':missing,
                'partial_histories':partial,'sources':sources,'input_sha256':digest(path),'input':str(path)}
    assert all(manifest['models'][m]['clean']['expected']==95 and manifest['models'][m]['attacked']['expected']==921 for m in ('base','trained'))
    (root/'role_inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({m:{s:{k:v for k,v in d.items() if k not in ('sources','partial_histories')} for s,d in ss.items()} for m,ss in manifest['models'].items()},indent=2))

if __name__=='__main__':
    main()
