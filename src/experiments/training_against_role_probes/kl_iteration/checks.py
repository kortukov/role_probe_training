"""Regression checks: legacy cap unchanged and agentic masks never truncated."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from role_probe_training.training.data import kl_examples

with TemporaryDirectory() as temp:
    path = Path(temp)/'anchors.jsonl'
    full = dict(id='full', input_ids=list(range(3000)), completion_mask=[False]*2500+[True]*500)
    old = dict(id='old', prompt_ids=[1]*1000, completion_ids=[2]*2000)
    path.write_text(json.dumps(full)+'\n'+json.dumps(old)+'\n')
    a,b = kl_examples(path)
    assert len(a['ids']) == 3000 and sum(a['completion_mask']) == 500
    assert len(b['ids']) == 2048 and sum(b['completion_mask']) == 1048
    full['completion_mask'].pop()
    path.write_text(json.dumps(full)+'\n')
    try:
        kl_examples(path)
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed mask accepted')
print('Full agentic histories, legacy caps, and malformed-mask checks passed.')
