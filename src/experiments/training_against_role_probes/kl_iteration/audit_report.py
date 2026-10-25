"""Check the required four-model, two-benchmark table against validated results."""
import argparse
import json
from pathlib import Path

ROOT=Path('experiment_reports/training_against_role_probes')
BENCHMARKS={'AppWorld test_normal':'appworld_test','τ² retail':'tau2_retail'}
VARIANTS={'AppWorld + agentic KL (800)':'agentic800','AppWorld + same-batch KL':'same_batch'}


def table_rows(text,marker):
    block=text.split(f'<!-- {marker} -->',1)[1].split(f'<!-- /{marker} -->',1)[0]
    rows={}
    for line in block.splitlines():
        if not line.startswith('| '):continue
        cells=[cell.strip() for cell in line.strip('|').split('|')]
        if cells[0] not in BENCHMARKS:continue
        key=(cells[0],cells[1])
        assert key not in rows, f'Duplicate table row: {key}'
        rows[key]=cells[2:]
    return rows


def audit(report):
    text=report.read_text()
    rows=table_rows(text,'kl-iteration-results')
    original=table_rows(text,'fresh-rollout-results')
    labels=['Base','AppWorld-trained',*VARIANTS]
    assert set(rows)=={(benchmark,label) for benchmark in BENCHMARKS for label in labels}
    comparison=json.loads((ROOT/'post_training/comparison.json').read_text())
    results={label:json.loads((ROOT/'kl_iteration'/variant/'post_training/results.json').read_text()) for label,variant in VARIANTS.items()}
    for (benchmark,label),cells in rows.items():
        key=BENCHMARKS[benchmark]
        row=(comparison[label] if label in comparison else results[label])[key]
        values=[row['task']['metrics'][row['task_metric']]['success_rate'],row['probe']['accuracy'],*row['probe']['per_role_recall']]
        assert len(cells)==7
        assert cells==[f'{100*x:.1f}' for x in values], (benchmark,label,cells,values)
        if label in ('Base','AppWorld-trained'):
            assert cells==original[(benchmark,label)]
    assert '<!-- kl-iteration-message -->' in text
    assert text.rstrip().endswith('[How to reproduce the experiment](RUNNING.md)')
    print(f'PASS: all 8 required model/benchmark rows and 56 metric cells: {report}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'README.md')
    audit(parser.parse_args().report)
