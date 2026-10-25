"""Validate every saved prediction against its confusion counts and cohort coverage."""
import argparse
import json
from pathlib import Path
import numpy as np


def validate(root):
    checks=[]
    for benchmark,expected in [('appworld_test',168),('tau2_retail',114),('swe_bench',500)]:
        folder=root/'predictions/post_training'/benchmark
        summary=json.loads((folder/'summary.json').read_text())
        rows=[json.loads(line) for line in (folder/'index.jsonl').open()]
        sources=[json.loads(line) for line in (root/'data/post_training'/benchmark/'trajectories.jsonl').open()]
        expected_ids={str(row['trajectory_id']) for row in sources}
        scored_ids={str(row['trajectory_id']) for row in rows}
        excluded_ids={str(row['trajectory_id']) for row in summary['excluded_trajectories']}
        assert len(expected_ids)==len(sources)==expected==summary['trajectories']
        assert len(scored_ids)==len(rows)==summary['scored_trajectories']
        assert not scored_ids & excluded_ids
        assert scored_ids | excluded_ids == expected_ids
        counts=np.zeros((5,5),dtype=np.int64)
        for i,row in enumerate(rows):
            with np.load(folder/row['file'],allow_pickle=False) as data:
                labels=data['labels']; predicted=data['predictions_12']
                assert labels.shape==predicted.shape==data['input_ids'].shape
                assert len(labels)<=summary['context_limit']==131072
                assert np.isin(labels,[-1,0,1,2,3,4]).all()
                assert np.isin(predicted,[0,1,2,3,4]).all()
                valid=labels>=0
                actual=np.bincount(labels[valid].astype(np.int64)*5+predicted[valid],minlength=25).reshape(5,5)
                np.testing.assert_array_equal(actual,row['confusion_counts']['12'])
                counts+=actual
                if i in (0,len(rows)-1):
                    probabilities=data['probabilities_12'].astype(np.float32)
                    assert probabilities.shape==(len(labels),5) and np.isfinite(probabilities).all()
                    np.testing.assert_allclose(probabilities.sum(1),1,atol=.002,rtol=0)
        np.testing.assert_array_equal(counts,summary['confusion_counts'])
        assert abs(counts.trace()/counts.sum()-summary['accuracy'])<1e-12
        checks.append(dict(benchmark=benchmark,histories=len(rows),unscorable=len(excluded_ids),
                           all_saved_predictions_match=True,exact_cohort_coverage=True))
    (root/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(f'PASS: all saved predictions and complete cohort coverage: {root}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    validate(parser.parse_args().root)
