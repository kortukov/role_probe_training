"""Replay the timeout attempt's exact editor calls; compare official tests with gold.

Diagnostic only: preserve the original cohort and make no model calls.
"""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import inspect_ai._util.zipfile
from inspect_ai import eval
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import solver
from inspect_ai.tool import text_editor
from inspect_ai.util import sandbox
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench.scorers import make_swebench_test_spec
from inspect_evals.swe_bench.solvers import swe_bench_oracle_solver
from swebench.harness.grading import get_eval_report
from role_probe_training.trajectories.swe_testbed import sandbox_config

ID = 'sphinx-doc__sphinx-7590'
ROOT = Path('experiment_reports/training_against_role_probes')


def original_sample():
    for path in (ROOT/'base_model/runs/swe_corrected/1505929/shard1/swe_bench/logs').glob('*.eval'):
        with zipfile.ZipFile(path) as archive:
            for name in set(archive.namelist()):
                if name.startswith('samples/') and ID in name and name.endswith('.json'):
                    raw = json.loads(archive.read(name))
                    def resolve(value):
                        if isinstance(value, str) and value.startswith('attachment://'):
                            return raw['attachments'][value.removeprefix('attachment://')]
                        if isinstance(value, dict):
                            return {k: resolve(v) for k, v in value.items()}
                        if isinstance(value, list):
                            return [resolve(v) for v in value]
                        return value
                    return resolve(raw)
    raise RuntimeError('Original attempt missing')


@solver
def replay_edits():
    async def solve(state, generate):
        sample = original_sample()
        edits = [e for e in sample['events'] if e['event'] == 'tool'
                 and e.get('function') == 'text_editor'
                 and e['arguments']['command'] in ('str_replace', 'insert')]
        assert len(edits) == 2
        for event in edits:
            await text_editor()(**event['arguments'])
        result = await sandbox().exec(['git', 'diff'], cwd='/testbed')
        saved = [e for e in sample['events'] if e['event'] == 'sandbox' and 'git diff --cached' in e.get('cmd', '')][-1]['output']
        # Transcript truncates only the final context line. Verify all retained bytes.
        prefix = saved.split('\nOutput truncated (')[0]
        assert result.stdout.startswith(prefix), 'Reconstructed edits differ from saved patch'
        (args.output/'reconstructed.patch').write_text(result.stdout)
        state.store.set('patch_verification', 'Exact original editor calls; diff matches retained transcript prefix')
        return state
    return solve


@scorer(metrics=[mean()])
def diagnostic_score():
    async def score(state, target):
        box = sandbox()
        spec = make_swebench_test_spec(state)
        await box.write_file('/tmp/official-eval.sh', spec.eval_script)
        timed_out = False
        try:
            await box.exec(['bash', '-c', 'bash /tmp/official-eval.sh > /tmp/test-output.txt 2>&1'], timeout=1800)
        except TimeoutError:
            timed_out = True
        output = await box.read_file('/tmp/test-output.txt')
        (args.output/f'{variant}-test-output.txt').write_text(output)
        report = None
        if not timed_out:
            with tempfile.NamedTemporaryFile('w') as f:
                f.write(output); f.flush()
                report = get_eval_report(spec, {'instance_id': ID, 'model_name_or_path': 'diagnostic', 'model_patch': 'diagnostic'}, f.name, include_tests_status=True)[ID]
        return Score(value=float(bool(report and report['resolved'])), metadata={'timeout': timed_out, 'report': report, 'output_tail': output[-6000:]})
    return score


parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
rows = []
for variant, agent in [('gold', swe_bench_oracle_solver()), ('saved_model_edits', replay_edits())]:
    logs = eval(swe_bench(sandbox_type='apptainer_testbed', sandbox_config=sandbox_config, scorer=diagnostic_score()),
                model='mockllm/model', solver=agent, sample_id=[ID], max_samples=1,
                log_dir=str(args.output/variant), display='plain', fail_on_error=False)
    for log in logs:
        for sample in log.samples or []:
            rows.append({'variant': variant, 'id': sample.id, 'error': sample.error.message if sample.error else None,
                         'scores': {k: v.model_dump() for k, v in (sample.scores or {}).items()}})
    (args.output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
