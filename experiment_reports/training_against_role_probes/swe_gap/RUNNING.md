# Reproduction

Code: [`src/experiments/training_against_role_probes/swe_gap/`](../../../src/experiments/training_against_role_probes/swe_gap/).
Run from `/data/cluster/users/kortukov/role_probe_training` with the existing locked
`src/experiments/agentic_eval/swe` uv environment. All diagnostics use CPU Slurm
allocations; they do not call a model or modify benchmark results.

```bash
sbatch src/experiments/training_against_role_probes/swe_gap/audit.sbatch
# After the audit has written data/selected_patches.json:
sbatch src/experiments/training_against_role_probes/swe_gap/regrade.sbatch
sbatch src/experiments/training_against_role_probes/swe_gap/tool_environment.sbatch
sbatch src/experiments/training_against_role_probes/swe_gap/tool_environment_fixed.sbatch
```

The audit reads the four original base shards under
`../base_model/runs/swe_native/1504808/`, resolves Inspect attachments, and writes
`audit.json`, `data/attempt_audit.json`, and `data/selected_patches.json`. Large raw
logs, saved patches, and diagnostic logs stay under this report's ignored `data/`
and `logs/` folders on cluster storage. `provenance.json` records jobs, package
versions, input hashes and public source revisions. Small reviewed summaries are
tracked here; they do not replace the current four-model comparison.

`regrade.py` applies the saved patch in a fresh pinned image using upstream's
patch-application commands, executes upstream's generated evaluation script,
and calls upstream's grader. `--inspect-script` substitutes Inspect's evaluation
script to investigate disagreements; `--sample-id` restricts this diagnostic.
The script uses Apptainer instead of the official Docker orchestration because
Docker is unavailable. SWE-bench's current repository is newer than the pinned
4.1.0 used here; we do not mutate the environment serving concurrent experiments.

The opt-in provider
[`swe_testbed.py`](../../../src/role_probe_training/trajectories/swe_testbed.py)
registers `apptainer_testbed` and activates Conda `testbed` for every tool command,
including child processes. To use it in a **new** SWE runner, import its
`sandbox_config` instead of `apptainer_mirror.sandbox_config`. Existing runners
continue using the original provider. The fixed environment test covers both
Inspect tools and direct execution on two repositories; it does not measure
model task success.

Public references:
- [Official SWE-bench](https://github.com/SWE-bench/SWE-bench).
- [OpenAI evaluation description](https://deploymentsafety.openai.com/gpt-oss/evaluation).
- [HarmonyAgent paper](https://arxiv.org/html/2604.00362v1) and
  [pinned high-reasoning configuration](https://github.com/borislavmavrin/harmonyagent/blob/175dda4a212193b79e5d860186c5043fd9cdc8e1/src/harmonyagent/config/swebench_harmony_high.yaml).

Summarize the completed diagnostic runs (substitute new job-specific directories
when reproducing):

```bash
.tools/bin/uv run --project src/experiments/agentic_eval/swe --frozen --no-sync python src/experiments/training_against_role_probes/swe_gap/summarize.py --original-env environment-1505908 --fixed-env environment-fixed-1505909 --regrade regrade-1505905 --inspect-regrade inspect-regrade-1505912
```

The last input is produced by `inspect_regrade.sbatch`, a one-patch fresh-image
control for the observed disagreement.
