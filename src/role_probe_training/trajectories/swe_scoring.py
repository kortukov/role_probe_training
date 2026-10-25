"""Grade exported SWE patches in a fresh pinned image with upstream scripts."""
import json
import tempfile
from types import SimpleNamespace

from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.util import sandbox
from inspect_evals.swe_bench.scorers import make_swebench_test_spec
from inspect_evals.swe_bench.solvers import START_COMMIT_STORE_KEY
from swebench.harness.grading import get_eval_report
from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
from swebench.harness.run_evaluation import GIT_APPLY_CMDS

from .apptainer import ApptainerSandbox
from .apptainer_mirror import sandbox_config


@scorer(name='swe_bench_scorer', metrics=[mean()])
def fresh_swe_scorer():
    async def score(state, target):
        original = sandbox()
        start = state.store.get(START_COMMIT_STORE_KEY, state.metadata['base_commit'])
        # Capture the patch with Git, without requiring a Python subprocess.
        result = await original.exec( ['bash', '-c', 'git add -A && git diff --cached "$1"', 'bash', start],
            cwd='/testbed')
        if not result.success:
            raise RuntimeError('Cannot capture model patch: '+result.stderr)
        patch = result.stdout
        if not patch.strip():
            return Score(value=0.0, explanation='Empty model patch', metadata={'model_patch':patch})
        spec = make_swebench_test_spec(state)
        config = sandbox_config('apptainer', SimpleNamespace(id=state.sample_id, metadata=dict(state.metadata)))
        environments = await ApptainerSandbox.sample_init('fresh-swe-score', config.config, state.metadata)
        try:
            box = environments['default']
            await box.write_file('/tmp/model.patch', patch)
            for command in GIT_APPLY_CMDS:
                applied = await box.exec(['bash','-c',command+' /tmp/model.patch'],cwd='/testbed')
                if applied.success:
                    break
            if not applied.success:
                return Score(value=0.0, explanation='Saved patch does not apply to fresh image',
                             metadata={'model_patch':patch,'apply_error':applied.stderr[-3000:]})
            await box.write_file('/tmp/official-eval.sh', spec.eval_script)
            await box.exec(['bash','-c','bash /tmp/official-eval.sh > /tmp/test-output.txt 2>&1'],timeout=1800)
            output = await box.read_file('/tmp/test-output.txt')
            if START_TEST_OUTPUT not in output or END_TEST_OUTPUT not in output:
                raise RuntimeError('Official test script did not complete its test phase: '+output[-3000:])
            prediction = {'instance_id':str(state.sample_id),'model_name_or_path':'role-probe-eval','model_patch':patch}
            with tempfile.NamedTemporaryFile('w',suffix='.txt') as f:
                f.write(output);f.flush()
                report = get_eval_report(spec,prediction,f.name,include_tests_status=True)[str(state.sample_id)]
            return Score(value=float(report['resolved']),explanation=json.dumps(report),
                         metadata={'model_patch':patch,'official_report':report,'test_output_tail':output[-3000:]})
        finally:
            await ApptainerSandbox.sample_cleanup('fresh-swe-score',config.config,environments,False)
    return score
