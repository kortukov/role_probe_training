"""Independently validate score aggregates, coverage and saved token artifacts."""
import json
from pathlib import Path
import numpy as np

root=Path('experiment_reports/training_against_role_probes')
checks=[]
for sub in ['', 'astra_sft_1k','depthbench','base_model']:
    for benchmark,expected in [('appworld_test',168),('tau2_retail',114),('swe_bench',500)]:
        directory=root/sub/f'predictions/{"post_training_whole" if benchmark=="tau2_retail" else "post_training"}/{benchmark}'
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['trajectories']==expected
        assert not summary.get('segmented_trajectories')
        rows=[json.loads(line) for line in (directory/'index.jsonl').read_text().splitlines()]
        assert len(rows)==summary.get('scored_trajectories',expected)
        assert len({row['trajectory_id'] for row in rows})==len(rows)
        excluded=summary.get('excluded_trajectories',[])
        assert len(rows)+len(excluded)==expected
        assert not {row['trajectory_id'] for row in rows} & {x['trajectory_id'] for x in excluded}
        layers=summary.get('layers',{'12':summary})
        for layer,values in layers.items():
            counts=np.zeros((5,5),dtype=np.int64)
            for row in rows:
                matrix=row['confusion_counts']
                counts+=np.asarray(matrix[layer] if isinstance(matrix,dict) else matrix)
            np.testing.assert_array_equal(counts,values['confusion_counts'])
            assert abs(counts.trace()/counts.sum()-values['accuracy'])<1e-12
        for row in [rows[0],rows[-1]]:
            with np.load(directory/row['file'],allow_pickle=False) as data:
                labels=data['labels'];assert len(labels)==len(data['input_ids'])
                assert len(labels)<=131072
                for layer in layers:
                    key=f'probabilities_{layer}' if f'probabilities_{layer}' in data else 'probabilities'
                    probabilities=data[key].astype(np.float32)
                    assert probabilities.shape==(len(labels),5)
                    assert np.isfinite(probabilities).all()
                    np.testing.assert_allclose(probabilities.sum(1),1,atol=0.002,rtol=0)
        checks.append({'model':sub or 'appworld_trained','benchmark':benchmark,
                       'scored_histories':len(rows),'unscorable_histories':len(excluded),
                       'index_confusions_match':True,'sampled_token_artifacts_valid':True})
(root/'post_training/validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(f'PASS: coverage, confusion aggregates, and sampled token artifacts for {len(checks)} model/benchmark cohorts')
