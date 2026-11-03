"""CPU regression against all three retained failures and ordinary controls."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from role_probe_training.on_policy_harmony.parser import parse_tokens

t = AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b',local_files_only=True)
base = Path('experiment_reports/training_against_role_probes/on_policy')
cases = [('batch64',base/'appworld270_batch64/runs/epoch1_repaired/rollouts/step00001/rank0'),
         ('batch16',base/'appworld270_batch16/runs/epoch1_repaired/rollouts/step00004/rank1'),
         ('full2700',base/'appworld2700/runs/colocated_epoch1/rollouts/step00003/rank1')]
results=[]
for name,folder in cases:
    rows=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    for label,row in [('control',rows[0]),('failure',rows[-1])]:
        parsed=parse_tokens(t,row['completion_ids'],[{'type':'function','function':{'name':'python'}}])
        if label=='control':assert not parsed['invalid'] and parsed['message']['tool_calls']
        if name in ('batch64','batch16') and label=='failure':
            assert parsed['invalid'] and not parsed['message'].get('tool_calls')
        results.append(dict(case=name,label=label,**parsed))
Path('/tmp/onpolicy-parser-fixtures.json').write_text(json.dumps(results))
print(json.dumps([dict(case=x['case'],label=x['label'],invalid=x['invalid'],issues=x['issues']) for x in results]))
