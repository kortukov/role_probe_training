"""Audit saved retail histories and API stops without generating or truncating text."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from role_probe_training.probes.harmony import render
from role_probe_training.trajectories.replay import replay_messages

root = Path('experiment_reports/training_against_role_probes')
tokenizer = AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b',local_files_only=True)
report = {}
for sub in ['base_model', '', 'astra_sft_1k', 'depthbench']:
    rows = [json.loads(line) for line in (root/sub/'data/post_training/tau2_retail/trajectories.jsonl').read_text().splitlines()]
    records = []
    for row in rows:
        text,_ = render(replay_messages(row)[0],bos=True)
        length = len(tokenizer(text,add_special_tokens=False)['input_ids'])
        assistants = [m for m in row['messages'] if m['role']=='assistant']
        empty_turns = sum(bool(m.get('thinking')) and not (m.get('content') or m.get('tool_calls')) for m in assistants)
        records.append({'sample_id':str(row['sample_id']),'tokens':length,
                        'success':row['scores']['retail_scorer']['value']=='C',
                        'generation_token_caps':row['generation_token_caps'],
                        'assistant_turns':len(assistants),'reasoning_only_turns':empty_turns,
                        'tool_calls':sum(len(m.get('tool_calls') or []) for m in assistants),
                        'limit':row.get('limit'),'error':bool(row.get('error')),
                        'simulator_stop_seen':row.get('simulator_stop_seen')})
    long = [r for r in records if r['tokens']>131072]
    generation_files = {Path(r['source_log']).parent.parent/'generations.jsonl' for r in rows}
    inputs,outputs,stops,call_examples = [],[],Counter(),[]
    ids = {r['sample_id'] for r in long}
    for path in sorted(generation_files):
        if not path.exists(): continue
        for line in path.open():
            event=json.loads(line)
            usage=event.get('output',{}).get('usage') or {}
            if 'input_tokens' in usage:inputs.append(usage['input_tokens'])
            if 'output_tokens' in usage:outputs.append(usage['output_tokens'])
            for choice in event.get('output',{}).get('choices',[]):stops[choice['stop_reason']]+=1
            if str(event['sample_id']) in ids:
                call_examples.append({'sample_id':str(event['sample_id']), 'usage':usage,
                    'stops':[c['stop_reason'] for c in event.get('output',{}).get('choices',[])]})
    result={'histories':len(rows),'median_replay_tokens':float(np.median([r['tokens'] for r in records])),
            'max_replay_tokens':max(r['tokens'] for r in records),
            'over_native_context':len(long),'over_native_successes':sum(r['success'] for r in long),
            'histories_with_reasoning_only_turns':sum(r['reasoning_only_turns']>0 for r in records),
            'reasoning_only_turns':sum(r['reasoning_only_turns'] for r in records),
            'assistant_turns':sum(r['assistant_turns'] for r in records),
            'histories_with_generation_caps':sum(r['generation_token_caps']>0 for r in records),
            'generation_token_caps':sum(r['generation_token_caps'] for r in records),
            'max_api_input_tokens':max(inputs) if inputs else None,
            'max_api_output_tokens':max(outputs) if outputs else None,'api_stop_reasons':dict(stops),
            'long_history_records':long,'all_history_records':records}
    report[sub or 'appworld_trained']=result
    local=root/sub/'data/post_training';local.mkdir(parents=True,exist_ok=True)
    (local/'long_history_calls.json').write_text(json.dumps(call_examples,indent=2)+'\n')
    print(sub or 'appworld_trained',json.dumps({k:v for k,v in result.items() if not k.endswith('records')}),flush=True)
(root/'post_training/retail_length_audit.json').write_text(json.dumps(report,indent=2)+'\n')
