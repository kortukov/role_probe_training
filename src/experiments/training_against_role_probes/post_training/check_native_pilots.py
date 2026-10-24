"""Gate a full SWE rerun on successful infrastructure checks and pilot exports."""
import json
from pathlib import Path
root=Path('experiment_reports/training_against_role_probes')
controls=json.loads((root/'runs/swe_native/scorer-1504794/results.json').read_text())
assert len(controls)==4 and all(not x['error'] for x in controls)
for sub,job in [('base_model','1504795'),('astra_sft_1k','1504796')]:
    folder=root/sub/f'runs/swe_native/{job}/shard0'
    context=json.loads((folder/'context_check.json').read_text())
    assert context['server_context_limit']==131072 and context['request_above_32k_passed']
    summary=json.loads((folder/'swe_bench/summary.json').read_text())
    assert summary['samples']==6 and not summary['missing_sample_ids']
    assert summary['sample_errors']==summary['export_errors']==summary['agent_model_errors']==0,summary
    # Inspect converts context errors to normal model stops; check these explicitly.
    assert summary['samples_with_model_context_stops']==0,summary
    print(sub,summary['metrics'])
print('PASS: full-rerun infrastructure gate')
