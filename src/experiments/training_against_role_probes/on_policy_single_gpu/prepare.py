"""Merge audited two-rank recovery caches for one-rank continuation."""
import json
from pathlib import Path
from role_probe_training.on_policy.data import sha256, validate_requests
from role_probe_training.on_policy_recovery.data import schedule
from role_probe_training.on_policy_recovery.recovery import stage_cache

BASE = Path('experiment_reports/training_against_role_probes/on_policy')
CODE = Path('src/experiments/training_against_role_probes')


def main():
    for name in ('appworld270_batch64', 'appworld270_batch16'):
        original = CODE/'on_policy_recovery'/f'{name}_migration.json'
        migration = json.loads(original.read_text())
        config = dict(migration['new_config'])
        oldroot = Path(config['output'])
        root = BASE/name/'runs/epoch1_single_gpu'
        if root.exists():
            raise FileExistsError(root)
        config.update(output=str(root), wandb_id=config['wandb_id']+'-single-gpu')
        batches = schedule(config['trajectories'], config['validation'], config['seed'],
                           episodes_per_task=config['episodes_per_task'], batch_size=config['batch_size'])
        spec = json.loads((oldroot/'recovery.json').read_text())
        assert spec['step'] == migration['step']+1
        episodes = batches[migration['step']]
        plan = dict(episodes=episodes, policy_step=migration['step'],
                    generation=config['generation'], image=config['image'], data=config['appworld_data'])
        rows, requests = [], []
        root.mkdir(parents=True)
        for rank in range(2):
            folder = root/'audit'/f'rank{rank}'
            folder.mkdir(parents=True)
            stage_cache(spec, rank, dict(plan, episodes=episodes[rank::2]), folder)
            rows += [json.loads(x) for x in (folder/'cached_trajectories.jsonl').read_text().splitlines()]
            requests += [json.loads(x) for x in (folder/'requests.jsonl').read_text().splitlines()]
        rowmap = {r['trajectory_id']: r for r in rows}
        assert len(rowmap) == len(rows)
        rows = [rowmap[e['id']] for e in episodes if e['id'] in rowmap]
        validate_requests(requests, [e for e in episodes if e['id'] in rowmap], plan['policy_step'])
        cache = root/'episode_cache/rank0'
        cache.mkdir(parents=True)
        (cache/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        for filename, records in [('trajectories.jsonl', rows), ('requests.jsonl', requests)]:
            (cache/filename).write_text(''.join(json.dumps(r)+'\n' for r in records))
        new_spec = dict(step=spec['step'], cache=str(cache.parent), ranks={'0':dict(
            episodes=len(rows), requests=len(requests), source_recovery=str(oldroot/'recovery.json'),
            source_recovery_sha256=sha256(oldroot/'recovery.json'),
            sha256={p.name:sha256(p) for p in cache.iterdir()})})
        (root/'recovery.json').write_text(json.dumps(new_spec, indent=2)+'\n')
        migration.update(new_config=config, world_size_change=[2,1])
        for path,digest in migration['previous']['source_sha256'].items():
            assert sha256(path)==digest, path
        assert sha256(Path(migration['checkpoint'])/'training.pt')==migration['checkpoint_sha256']
        for suffix,content in [('.json',config),('_migration.json',migration)]:
            (CODE/'on_policy_single_gpu'/f'{name}{suffix}').write_text(json.dumps(content,indent=2)+'\n')
        check=root/'audit/merged';check.mkdir()
        stage_cache(new_spec,0,plan,check)
        print(name, 'cached',len(rows),'fresh',len(episodes)-len(rows),'updates',len(batches),'resume',migration['step'])


if __name__=='__main__':
    main()
