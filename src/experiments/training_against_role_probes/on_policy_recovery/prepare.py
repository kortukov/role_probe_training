"""Audit and freeze same-policy episode caches without executing any tools."""
import hashlib,json
from pathlib import Path
from inspect_ai.log import read_eval_log

BASE=Path('experiment_reports/training_against_role_probes/on_policy')
CODE=Path('src/experiments/training_against_role_probes/on_policy_recovery')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

for name,run,step,reason in [('appworld270_batch64','epoch1_repaired',1,'invalid_model_output'),('appworld270_batch16','epoch1_repaired',4,'invalid_model_output'),('appworld2700','epoch1_harmony',3,'serving_context_limit')]:
    root=BASE/name; output=root/'runs/epoch1_recovery'
    output.mkdir(exist_ok=True)
    cache=output/'episode_cache';cache.mkdir(exist_ok=True)
    spec=dict(step=step,cache=str(cache),ranks={})
    for rank in (0,1):
        source=root/f'runs/{run}/rollouts/step{step:05d}/rank{rank}'
        target=cache/f'rank{rank}';target.mkdir(exist_ok=True)
        rows=[json.loads(x) for x in (source/'trajectories.jsonl').read_text().splitlines()]
        requests=[json.loads(x) for x in (source/'requests.jsonl').read_text().splitlines()]
        source_hashes={str(source/f):digest(source/f) for f in ('plan.json','trajectories.jsonl','requests.jsonl')}
        for row in rows:
            if row['export_errors']:raise ValueError('Export error: '+row['trajectory_id'])
            if row['error']:
                errors=json.loads((source/'server_errors.json').read_text())
                marker=('Full prompt reached serving context limit' if reason=='serving_context_limit' else
                        'Unsupported assistant channel' if name.endswith('64') else 'Unsupported generated recipient: assistant')
                if not any(marker in error for error in errors):raise ValueError('Unrecognized failure')
                log=read_eval_log(row['source_log'])
                samples=[s for s in log.samples if str(s.id)==str(row['sample_id'])]
                assert len(samples)==1
                state=samples[0].metadata['appworld']
                assert isinstance(state['success'],bool)
                # Preserve original errors, log references and the raw sampled
                # tokens; restore only the score skipped by Inspect's API error.
                row['scores']['appworld_success']={'value':'C' if state['success'] else 'I'}
                row['recovered_terminal']=dict(reason=reason,appworld_state=state,source_log=row['source_log'],
                    note='Terminal episode salvaged from original evaluated world state; no tool re-execution or resampling.')
                if reason=='serving_context_limit':row['model_context_stops']+=1
            row['recovery_source']=str(source/'trajectories.jsonl')
        ids={r['trajectory_id'] for r in rows}
        assert {r['episode_id'] for r in requests}==ids
        (target/'plan.json').write_text((source/'plan.json').read_text())
        (target/'requests.jsonl').write_text((source/'requests.jsonl').read_text())
        (target/'trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        spec['ranks'][str(rank)]=dict(episodes=len(rows),requests=len(requests),sources_sha256=source_hashes,
            sha256={f:digest(target/f) for f in ('plan.json','requests.jsonl','trajectories.jsonl')})
    (output/'recovery.json').write_text(json.dumps(spec,indent=2)+'\n')
    old=json.loads((Path('src/experiments/training_against_role_probes/on_policy_harmony')/f'{name}_migration.json').read_text())
    config=dict(old['new_config']);config.update(output=str(output),wandb_id=config['wandb_id']+'-recovery')
    # Semantic training settings remain identical; only output/run identity changes.
    old['new_config']=config
    assert digest(Path(old['checkpoint'])/'training.pt')==old['checkpoint_sha256']
    for file,sha in old['previous']['source_sha256'].items():assert digest(Path(file))==sha,file
    (CODE/f'{name}.json').write_text(json.dumps(config,indent=2)+'\n')
    (CODE/f'{name}_migration.json').write_text(json.dumps(old,indent=2)+'\n')
    print(name,[(r,v['episodes'],v['requests']) for r,v in spec['ranks'].items()],flush=True)
script=Path('src/experiments/training_against_role_probes/on_policy_harmony/train.sbatch').read_text().replace('on_policy_harmony.train','on_policy_recovery.train')
(CODE/'train.sbatch').write_text(script)
