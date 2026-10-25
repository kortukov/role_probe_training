"""Replay fixed model patches in fresh images using official SWE-bench scripts/grader.

Apptainer substitutes for Docker orchestration; no model generation is performed.
"""
import argparse
import json
from pathlib import Path
import tempfile
from inspect_ai import eval
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import solver
from inspect_ai.util import sandbox
from inspect_evals.swe_bench import swe_bench
from inspect_evals.swe_bench.scorers import make_swebench_test_spec
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from role_probe_training.trajectories.apptainer_mirror import sandbox_config

ROOT=Path('experiment_reports/training_against_role_probes/swe_gap')
PATCHES={r['id']:r for r in json.loads((ROOT/'data/selected_patches.json').read_text())}

@solver
def apply_saved_patch():
    async def solve(state,generate):
        box=sandbox()
        probe="import sys; print(sys.executable); print(sys.path); import importlib.util; print({m:bool(importlib.util.find_spec(m)) for m in ['pytest','mpmath','asgiref','numpy']})"
        env=await box.exec(['python','-c',probe])
        state.store.set('default_python',{'stdout':env.stdout,'stderr':env.stderr,'returncode':env.returncode})
        env=await box.exec(['bash','-c','source /opt/miniconda3/bin/activate testbed; python -c "$1"','bash',probe])
        state.store.set('testbed_python',{'stdout':env.stdout,'stderr':env.stderr,'returncode':env.returncode})
        patch=PATCHES[str(state.sample_id)]['patch']
        await box.write_file('/tmp/saved-model.patch',patch)
        for command in GIT_APPLY_CMDS:
            result=await box.exec(['bash','-c',command+' /tmp/saved-model.patch'],cwd='/testbed')
            if result.success: break
        if not result.success: raise RuntimeError('Saved patch could not apply: '+result.stderr)
        return state
    return solve

@scorer(metrics=[mean()])
def official_scorer():
    async def score(state,target):
        spec=make_swebench_test_spec(state)
        box=sandbox()
        script = spec.eval_script
        if a.inspect_script:
            from inspect_evals.swe_bench.scorers import get_eval_script
            script = get_eval_script(**{k:state.metadata[k] for k in ['test_patch','repo','version','base_commit']})
        await box.write_file('/tmp/official-eval.sh',script)
        await box.exec(['bash','-c','bash /tmp/official-eval.sh > /tmp/official-test-output.txt 2>&1'],timeout=1800)
        output=await box.read_file('/tmp/official-test-output.txt')
        instance_id=str(state.sample_id)
        pred={'instance_id':instance_id,'model_name_or_path':'saved-base-native','model_patch':PATCHES[instance_id]['patch']}
        with tempfile.NamedTemporaryFile('w',suffix='.txt') as f:
            f.write(output);f.flush()
            report=get_eval_report(spec,pred,f.name,include_tests_status=True)[instance_id]
        return Score(value=float(report['resolved']),metadata={'official_report':report,'output_tail':output[-3000:]})
    return score

p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
p.add_argument('--sample-id',action='append');p.add_argument('--inspect-script',action='store_true');a=p.parse_args()
if a.sample_id: PATCHES={k:v for k,v in PATCHES.items() if k in a.sample_id}
a.output.mkdir(parents=True,exist_ok=False)
task=swe_bench(sandbox_type='apptainer',sandbox_config=sandbox_config,scorer=official_scorer())
logs=eval(task,solver=apply_saved_patch(),model='mockllm/model',sample_id=list(PATCHES),
          max_samples=2,log_dir=str(a.output/'logs'),display='plain',fail_on_error=False)
rows=[]
for log in logs:
    for sample in log.samples or []:
        rows.append({'id':sample.id,'original_score':PATCHES[str(sample.id)]['original_score'],
                     'official_score':sample.scores['official_scorer'].value if sample.scores else None,
                     'error':sample.error.message if sample.error else None,
                     'environment':sample.store,
                     'official_report':sample.scores['official_scorer'].metadata if sample.scores else None})
result={'inspect_script':a.inspect_script,'protocol':'SWE-bench 4.1.0 official TestSpec.eval_script and get_eval_report; fresh pinned images via Apptainer, not Docker CLI',
        'samples':len(rows),'errors':sum(r['error'] is not None for r in rows),
        'disagreements':[r['id'] for r in rows if r['official_score'] is not None and r['official_score']!=r['original_score']],
        'results':rows}
(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='results'}))
assert len(rows)==len(PATCHES)
