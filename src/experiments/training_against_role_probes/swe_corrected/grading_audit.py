"""Inspect saved grading failures without rerunning or replacing model attempts."""
import collections
import json
from pathlib import Path
import zipfile
import inspect_ai._util.zipfile

ROOT = Path('experiment_reports/training_against_role_probes')
rows = []
for path in sorted((ROOT/'base_model/runs/swe_corrected_v2/1516499').glob('shard*/swe_bench/logs/*.eval')):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith('samples/') or not name.endswith('.json'):
                continue
            sample = json.loads(archive.read(name))
            attachments = sample.get('attachments', {})
            def resolve(value):
                if isinstance(value, str) and value.startswith('attachment://'):
                    return attachments[value.removeprefix('attachment://')]
                if isinstance(value, dict):
                    return {k: resolve(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [resolve(v) for v in value]
                return value
            score = resolve(sample['scores']['swe_bench_scorer'])
            meta = score.get('metadata', {})
            tail = meta.get('test_output_tail', '')
            report = meta.get('official_report', {})
            statuses = report.get('tests_status', {})
            row = {'id': sample['id'], 'repo': sample['metadata']['repo'],
                   'score': score['value'], 'pytest_internal_error': 'INTERNALERROR>' in tail,
                   'localhost_resolution_error': 'socket.gaierror' in tail and 'self.sock.bind' in tail,
                   'test_counts': {kind: {k: len(v) for k, v in counts.items()} for kind, counts in statuses.items()},
                   'test_output_tail': tail, 'log': str(path)}
            # Agent-side outputs are a separate diagnostic from grading.
            outputs = [str(resolve(event.get('result', ''))) for event in sample['events'] if event['event'] == 'tool']
            row['agent_localhost_resolution_error'] = any('socket.gaierror' in text and 'self.sock.bind' in text for text in outputs)
            rows.append(row)
assert len(rows) == len({r['id'] for r in rows}) == 500
summary = {'samples': len(rows), 'successes': sum(r['score'] == 1 for r in rows),
           'pytest_internal_error_ids': [r['id'] for r in rows if r['pytest_internal_error']],
           'localhost_resolution_error_ids': [r['id'] for r in rows if r['localhost_resolution_error']],
           'agent_localhost_resolution_error_ids': [r['id'] for r in rows if r['agent_localhost_resolution_error']],
           'by_repo': {repo: {'samples': sum(r['repo'] == repo for r in rows), 'successes': sum(r['repo'] == repo and r['score'] == 1 for r in rows)} for repo in sorted({r['repo'] for r in rows})}}
output = ROOT/'swe_corrected/base_audit_v2'
(output/'data/grading_audit.json').write_text(json.dumps(rows, indent=2)+'\n')
(output/'grading_audit.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
