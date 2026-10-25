# Reproduction and continuation

Use the locked `src/experiments/agentic_eval/swe` client and `server_current` serving
environments, plus `training_against_role_probes/runtime` for probes. All execution
uses Slurm from `/data/cluster/users/kortukov/role_probe_training`. Do not change
source or environments used by active jobs. This runner is isolated from concurrent
KL collection code; consolidate shared logic after the protocol is validated.

Code: [`swe_corrected/`](../../../src/experiments/training_against_role_probes/swe_corrected/).
[Pipeline manifest](pipeline.json) records job IDs, settings, source hashes, controls,
and the six model roots. Never submit a duplicate active cohort.

1. `controls.sbatch` checks the activated tool environment and fresh-image scorer
   with gold/no-op patches. A proxy-API issue in the initial control was fixed
   before model generation; all four controls passed on the rerun.
2. `collect.sbatch ROOT full`, submitted with `--array=0-3`, generates disjoint
   125-task shards. `ROOT/runs/swe_corrected/JOB/shardN/swe_bench/` holds the logs,
   trajectories and manifests. Base generation reuses the unchanged BF16 export
   in `base_model/data/swe_native/model`; trained models use their existing
   `data/post_training/model` exports. No new model or probe training is needed.
3. `finish_base.sbatch ROOT JOB` assembles all 500 attempts and audits the base logs.
   Read `base_model/post_training/swe_corrected_task_results.json` and
   `swe_corrected/base_audit/audit.json`. Inspect representative failure traces and
   budget/context stops. If performance remains far below 60.7%, investigate
   additional environment/tool failures, fresh-patch scoring discrepancies, and
   agent/Harmony differences before declaring the baseline validated. A higher
   score alone is not a correctness check. Keep protocol changes in new cohorts.
4. Once the baseline is reviewed, submit the five remaining model roots from the
   manifest under exactly the same validated settings. Assemble each with
   `assemble.py --root ROOT --collection-id JOB`. Only pre-inference infrastructure
   failures with zero benchmark model calls can be replaced; preserve provenance.
5. `score.sbatch ROOT` reads `ROOT/data/swe_corrected/evaluation/trajectories.jsonl`
   and writes `ROOT/predictions/swe_corrected/`. It scores whole histories, explicitly
   excluding over-native histories only from role metrics. Base reads all five
   existing layers; the report compares layer 12 consistently across models.
6. Validate complete 500-task coverage, identical selections/settings, model/probe
   revisions, input hashes, confusion aggregates and sampled token artifacts for
   all six models. Update the parent README's original and KL SWE rows together,
   with task success, overall/per-role accuracy and coverage. Put percentage-only
   matrices in the appendix. Preserve concurrent AppWorld/retail results and KL
   changes. Review, commit and push the complete report; partial cohorts are not
   final results.

Current settings: 131,072 context tokens; high reasoning; temperature 1; seed 123;
16,384 output tokens/call; 2,000 messages; 10 million cumulative tokens; 7,200
seconds/task; four concurrent samples/shard. Fresh scoring has a 1,800-second test
command timeout. Infrastructure and task-level errors remain visible.

The official scorer implementation is
[`swe_scoring.py`](../../../src/role_probe_training/trajectories/swe_scoring.py);
the activated tool provider is
[`swe_testbed.py`](../../../src/role_probe_training/trajectories/swe_testbed.py).
Missing test-phase markers raise an evaluation error instead of silently scoring
an incomplete test run as a model failure. Saved patch metadata remains in Inspect
logs. Large artifacts remain on cluster storage at the paths above.

`watch.sh THREAD_UUID` is a one-shot watcher of `pipeline.json`. It queues this
existing conversation on completion, failure or a 30-minute progress check. Rearm
it only after the previous watcher has exited; maintain `watch_jobs` and `ready_job`
as the pipeline advances. Thread UUID and local watcher PID are recorded in the
manifest and ignored `logs/watcher.json`. After the final reviewed report is pushed,
do not rearm. The watcher depends on the running Codex daemon.
