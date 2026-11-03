"""Exercise real collection from cached episodes; no environment re-execution."""
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
from role_probe_training.on_policy_recovery.collection import collect
from role_probe_training.on_policy_recovery.recovery import stage_cache
from role_probe_training.on_policy_recovery.data import schedule
from role_probe_training.on_policy.data import validate_requests

base=Path('experiment_reports/training_against_role_probes/on_policy')
code=Path('src/experiments/training_against_role_probes/on_policy_single_gpu')
t=AutoTokenizer.from_pretrained('.cache/models/gpt-oss-20b',local_files_only=True)
runtime=SimpleNamespace(tokenizer=t,device='cpu',hidden=None,
    model=SimpleNamespace(gradient_checkpointing_disable=lambda:None,gradient_checkpointing_enable=lambda **kwargs:None),
    base=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=131072)))
for name in ('appworld270_batch64','appworld270_batch16'):
    config=json.loads((code/f'{name}.json').read_text())
    spec=json.loads((Path(config['output'])/'recovery.json').read_text())
    batches=schedule(config['trajectories'],config['validation'],config['seed'],episodes_per_task=config['episodes_per_task'],batch_size=config['batch_size'])
    migration=json.loads((code/f'{name}_migration.json').read_text())
    old=Path(migration['checkpoint']).parent.parent
    assert batches==json.loads((old/'schedule.json').read_text())
    step=spec['step']
    with tempfile.TemporaryDirectory(prefix='recover-check-') as tmp:
        root=Path(tmp);(root/'recovery.json').write_text(json.dumps(spec))
        for rank in (0,):
            source=Path(spec['cache'])/f'rank{rank}'
            plan=json.loads((source/'plan.json').read_text())
            if name=='appworld270_batch64':
                folder=root/f'rank{rank}';folder.mkdir()
                stage_cache(spec,rank,plan,folder)
                assert (folder/'cached_trajectories.jsonl').read_bytes()==(source/'trajectories.jsonl').read_bytes()
                assert (folder/'requests.jsonl').read_bytes()==(source/'requests.jsonl').read_bytes()
            else:
                examples,requests,records=collect(runtime,config,batches[step-1],step,root,rank)
                assert len(examples)==len(records)==16
                assert requests==[json.loads(line) for line in (source/'requests.jsonl').read_text().splitlines()]
                assert records==[json.loads(line) for line in (source/'trajectories.jsonl').read_text().splitlines()]
                assert sum(bool(r.get('recovered_terminal')) for r in records)==1
                validate_requests(requests,batches[step-1],step-1)
    print(name,'same schedule, cache provenance, task coverage and exact tokens passed',flush=True)
# A changed cache must fail; no unchecked fallback to generation.
with tempfile.TemporaryDirectory() as tmp:
    broken=json.loads(json.dumps(spec));broken['ranks']['0']['sha256']['requests.jsonl']='0'*64
    try:stage_cache(broken,0,json.loads((Path(spec['cache'])/'rank0/plan.json').read_text()),Path(tmp))
    except ValueError:pass
    else:raise AssertionError('Corrupt-cache guard did not reject')
print('Cached batches collected without any sampler/world execution; malformed and context-stopped episodes retained. Corruption guard passed.')
