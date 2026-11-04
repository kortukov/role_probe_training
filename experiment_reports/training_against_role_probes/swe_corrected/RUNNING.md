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

Startup-only replacements are recorded under `infrastructure_repairs` in the
manifest. Canonical shard paths under the original collection ID are symlinks
to replacement shards; failed startup logs remain in suffixed sibling folders.
The assembly and audit read those canonical paths. The collection script retries
server startup at most three times before any benchmark call; it never retries
a completed or failed model task.

## Residual baseline diagnosis

The first corrected cohort completed with 226/500 successes. It is retained as a
separate cohort and is not yet the validated six-model comparison. Environment
controls job `1516255` used `residual_environment.sbatch`; raw output is in
`data/residual-env-1516255/results.json`. Job `1516309`, submitted with
`timeout_diagnostic.sbatch`, compares the original Sphinx timeout's saved editor
calls with the gold patch. Its artifacts are in
`data/timeout-diagnostic-1516309/`; no model generation is performed. The original
patch transcript loses its final context line, so diagnostic replay uses the two
exact editor calls and verifies the retained diff prefix before scoring.

Next: inspect both controls, fix the Python interpreter selection in an isolated
new protocol, and assess the binding cumulative token budget. Keep the original
500 attempts intact. Do not launch the five trained cohorts until the resulting
protocol has passed validation; do not select retries by their outcome.

### Second corrected cohort

Interpreter checks `1516433` passed 20/20 checks on four repositories, including
legacy-Python Requests. `swe_testbed_python.py` changes only Inspect's exact Python
tool command to the activated repository interpreter; the earlier provider remains
unchanged. `collect_v2.sbatch` submits the new cohort to `runs/swe_corrected_v2/`
with `--repository-python --token-limit 0` (zero is converted to no cumulative cap).
Native context, message limit, per-task time limit and all other settings are
unchanged. Base array `1516499` has four disjoint shards. Review job `1516502`
assembles all 500 into `base_model/data/swe_corrected_v2/evaluation/`, writes
`base_model/post_training/swe_corrected_v2_task_results.json`, and audits into
`base_audit_v2/`. The assembly/audit entry points accept `--cohort swe_corrected_v2`.
Do not combine either corrected cohort with the other, and do not reuse the old
`score.sbatch` paths for this cohort without explicitly selecting its input/output.
All six reported models still require task and whole-history role results from
the eventually validated protocol. The first baseline's result is a diagnostic,
not a final six-model report.

Timeout diagnostic `1516309` completed: the gold patch passed 25 tests in 4.02s;
the saved model edits reproduced the 1,800s test-phase timeout. Full test outputs
and the reconstructed patch are retained under its diagnostic artifact directory.


### Resumed pipeline — 2026-10-27

The original v2 base shard `1516499_1` reported host OOM after all 125 results
were saved. Released the blocked dependency of review `1516502`; assembly and
raw-log audit completed successfully. All 500 unique attempts match the selection,
and all raw verdicts match exports. Original attempts and source hashes are retained.
No baseline generation was repeated.

The v2 protocol is accepted for matched Inspect-harness comparison, with the
model-card gap explicitly unresolved. The current `pipeline.json` is authoritative
for job IDs: five trained arrays, dependent assembly and whole-history scoring,
plus base whole-history scoring. Collection requests 256 GiB host memory with
`--array=0-3%2`; generation settings remain identical to the base run. New scripts
`assemble_v2.sbatch` and `score_v2.sbatch` explicitly select v2 inputs/outputs.
The older commands above document the earlier cohort only.

Continue by checking every collection, assembly and scoring job in the manifest.
Validate all six task/probe summaries, input hashes, model/probe revisions, source
settings, coverage and per-token artifacts before updating all parent and KL SWE
rows together. Do not run the old four-model `compare.py --native-swe` as the v2
publisher: it reads older cohorts. Reports and GitHub publication remain pending.
The watcher now targets the resumed conversation and requires all six score jobs
for readiness; a failure or 30-minute check also wakes it.


### Grading gate revoked — 2026-10-27

`grading_audit.py` scans saved outputs and found 33 Matplotlib pytest startup
crashes from localhost resolution; the same error occurs in 13 agent trajectories.
Gold control 1610621 reproduces it. Its Requests case could not apply the gold
patch because the oracle file helper invokes Python 2 without pathlib; separate
Requests control 1610623 uses the base sandbox file helper and the identical fresh
scorer. This diagnostic error does not describe the original model attempts.

All five trained collection arrays are held with `scontrol hold`; their dependent
assembly/scoring jobs remain queued. Base replay need not regenerate trajectories.
Do not release collection until a versioned environment/scorer correction passes
broader gold controls. Regrade saved patches to isolate grading effects; retain
original attempts, and assess agent-side damage separately. The original no-network
Apptainer namespace and hostname setup need inspection. Do not enable unrestricted
network as an unexamined workaround. Watcher now monitors the diagnostic gate.

Requests control 1610623 completed: both gold patches scored zero, with no sample
errors. Requests2931 passes its issue-specific test but fails two required
regressions; Requests1724 fails 35 required regressions. Small evidence is included
in base_audit_v2/grading_audit.json; raw controls retain complete logs.


### Opt-in localhost repair and saved-patch regrading

Network diagnostic 1610624 confirms empty image `/etc/hosts`, failed localhost
resolution, working numeric loopback. `swe_localhost.py` modifies only the unpacked
image-owned hosts file and verifies a localhost socket connection. It retains
`--net --network none`. Sandbox file transfer uses base Python independently of
the repository interpreter. `swe_scoring_localhost.py` uses this provider for fresh
grading, retains full test outputs under SWE_GRADING_ARTIFACTS, and raises on
pytest INTERNALERROR. Existing providers/scorers remain unchanged.

Gold controls 1610625: Django14155, Matplotlib20859/24570, SymPy13878 pass; Requests
1724/2931 fail due external HTTP fixtures/missing pytest-httpbin/network timeout
semantics. `regrade_saved.sbatch` job 1610633 extracts and hashes all 42 original
Matplotlib/Requests patches and regrades without model calls. Output: `data/saved-localhost-regrade-1610633/`. The manifest records code hashes and jobs.

All-eight Requests localhost gold survey: job1610636, `requests_localhost_controls.sbatch`,
output `data/requests-localhost-controls-1610636/`. Namespace capability control
1610635 failed before any route setup: writing /proc/self/uid_map is prohibited.
The probe attempted only a new user/network namespace, never host route changes.

Held v2 collection scripts still reference the old environment. Do not simply
release them after controls pass: a corrected protocol must explicitly select
the new provider/scorer and retain distinct cohort provenance. Agent-side damage
in v2 also prevents treating regraded saved patches as fresh corrected trajectories.


### Completed regrade and broader gold validation

Regrade1610633:42 attempts,15 successes (all Matplotlib),0 evaluation errors. Every
new success has the exact original patch hash. Six unsuccessful re-exports differ
in bytes and remain diagnostic; original patches are retained. Provisional total
243/500 retains the other458 outcomes and is not a fresh corrected-agent result.
Requests gold1610636 passes only6028. Small summaries live in base_audit_v2/.

HTTP fixture1610652 uses locked dependencies in the separate http_fixture project,
installed via uv into data/http_fixture/py311 and py39. Existing environments are
unchanged. The earlier diagnostic1610648 was stopped after a missing pytest import
in its service helper; the rerun uses wsgiref. Fixtures improve individual tests but
no additional gold verdicts pass. These providers are diagnostic, not production.

Gold survey1610674:four CPU shards cover all492 non-Requests tasks, original gold
patches, localhost provider/scorer, no model calls. Output data/gold-survey-1610674/.
Snapshot control1610678 examines pre-existing image build files in exported gold
patches; output data/snapshot-controls-1610678/. Keep source frozen while jobs run.
A user question is pending about treatment of gold-invalid tasks; do not silently
exclude them or release held runs. Corrected agent collection also requires a
separate protocol: regraded baseline traces still contain agent-side failures.

Snapshot control1610678 completed: Requests1142 and6028 both pass (exported gold
patches800/486 bytes). `swe_snapshot.py` commits initial image files before the
solver so initial untracked build output is not exported as a model edit. Original
artifacts remain unchanged. The492-task gold survey still uses localhost-only,
so its failures may include this additional snapshot defect. Do not modify it
mid-run; diagnose/recheck failures separately with explicit provenance.


### Full gold survey and residual controls
Survey1610674 completed:483/492 gold controls pass,9 fail,0 sample errors.
Five failing controls require external HTTP (Sphinx8475/7985/8269/10435,
Matplotlib20488). Pylint4661 lacks appdirs. Django10097 imports installed egg
code instead of checkout. Sphinx9711 actually reports one passing test but its
nonverbose output lacks a parsed test verdict; Astropy7606 reports242 passes
but names the dimensionless parameter unit0 rather than expected empty brackets.
These latter two are diagnostic findings, not accepted score corrections.

Isolated control1615409 tests snapshot + PYTHONPATH=/testbed + verbose pytest
arguments passed through tox on Sphinx9711/Astropy7606/Django10097, with gold
and unchanged patches. Empty unchanged patches return zero before test execution;
these are capture controls, not proof that a broken solution fails the tests.
Image inspection1615410 reads dependency/test metadata for Astropy7606,
Pylint4661 and Requests1766/5414. No model calls or original attempts changed.
New scorer swe_scoring_diagnostic.py is separate; do not modify while active.
Five trained arrays remain HELD on old scripts; publication remains pending.

Pylint missing dependency control1615411 uses appdirs==1.4.4 copied only into fresh diagnostic images; prepared via uv pip --target data/appdirs-fixture. Shared environments unchanged. Image inspection1615410 completed: Astropy Python3.6/pytest3.3.1; Requests5414 setup.py pins pytest-httpbin==0.0.7 (earlier local fixture used2.1.0). This mismatch remains to investigate.

Pylint4661 control1615411 completed: gold passes after supplying appdirs1.4.4; empty unchanged patch scores0; no sample errors. Small evidence base_audit_v2/appdirs_controls.json. Parser/source-path1615409 still running; watcher rearmed for this thread.


### Residual control follow-up
Parser/source-path1615409 completed: all three gold verdicts still0. Sphinx9711
now prints the explicit PASSED node, but upstream parser only accepts status at
line start/end and misses trailing [100%]. New isolated summary control1615414
adds pytest -rA, gold plus a harmless marker-file patch as negative control so
negative grading actually executes tests. Django10097 imports checkout after
PYTHONPATH repair but still lacks templates; source-path alone is insufficient.
Astropy remains a parameter-ID mismatch. Read-only identity inspection1615416
checks original test/pytest source and Django template provenance. No model calls.


### Verified Sphinx summary repair and Astropy identity control
Summary control1615414 completed: Sphinx9711 gold1, nonfunctional marker patch0;
negative control executes tests, unlike previous empty-patch controls. Add -rA
to pytest arguments passed through tox; upstream parser misses trailing [100%].
Identity diagnostic1615416 confirms Astropy ids=str(unit), and pytest3.3.1 _idval
uses `if s` so empty dimensionless name falls back to unit0. Narrow normalization
swe_test_identity.py renames this exact node for this task only, preserves verdicts
and avoids collisions. Paired gold/marker control1615419 running. Original output
is retained; only parser input normalized. Django still fails with checkout imports;
full-suite trace includes SQLite missing django_site__old and missing templates.
Do not claim its source-path repair validated. Requests and external HTTP unresolved.


### Astropy validated and saved-patch effect measured
Astropy identity1615419: gold1 and marker-negative0, full grading executed.
Saved-patch regrade1615421 covers all four tasks with newly validated repairs:
Astropy7606/Sphinx9711 now pass with EXACT original patch hashes; Pylint4661
remains0, Requests1142 remains0 (its re-export differs; retain as diagnostic).
Provisional total245/500 (49.0%) =228 original+15 localhost+2 new repairs,
retaining remaining outcomes; this is not a matched corrected-agent cohort.

Requests historical fixture survey1615422 runs all8 gold patches with snapshot
and separate py39 pytest-httpbin0.0.7/httpbin0.7.0/Flask1.1.4/Werkzeug1.0.1/
MarkupSafe2.0.1 target, matching the test's declared pytest-httpbin version.
Exact resolved versions: base_audit_v2/legacy_fixture_dependencies.json.
Base py311 HTTP service dependencies unchanged; original environments untouched.
No host networking changed. Protocol is diagnostic; source frozen while active.
Django and external HTTP still unresolved; trained arrays remain HELD.


### Historical Requests fixture outcome and dual-protocol control
Legacy fixture1615422 completed: gold1142/6028 pass; other6 fail. Required
failures for2931/5414 now only connect-timeout tests (network-none returns
ENETUNREACH). Required failures1724/1766 now only mixed-case scheme test,
which sends HTTP and HTTPS to the same explicit local port.1921/2317 also
retain external-domain/timeout expectations. Exact failures in
base_audit_v2/legacy_fixture_controls.json.

New dual HTTP/TLS fixture control1615424 covers1724/1766 gold and marker-negative
patches. Local127.0.0.1:8765 peeks TLS handshake and serves either protocol.
Creates/trusts certificate only within each fresh image and snapshots trust
before solver so it is not exported as a model edit. TLS verification remains
on; no host trust/network changes or external access. Earlier providers unchanged.
No model calls. Source frozen while active. Provisional model total remains245/500.

Dual HTTP/TLS1615424 DONE: Requests1766 gold1 marker0;1724 both controls stopped before tests because Python2 returns relative requests/cacert.pem. Separate provider/scorer v2 resolves via os.path.abspath; control1615425 reruns only1724. No model calls or old-source changes.

Requests1724 v2 control1615425 gold0: legacy Python2 matcher ignores certificate IP SAN and reports hostname127.0.0.1 versus DNS localhost. New isolated v3 control1615428 adds DNS-form127.0.0.1 alongside IP SAN. Verification stays on; fresh-image trust only, no test/expectation changes. Old alarm1615424 is stale; follow pipeline current job.

TLS v3 control1615428 DONE: Requests1724 gold1 marker0. Both1724/1766 now have successful paired gold/negative controls. Saved-patch regrade1615429 tests only their original v2 model patches under v3 fixture, no new model calls. Check exact patch hashes before interpreting changes; provisional total remains245/500 until results.


### TLS saved-patch results and Django test isolation
Regrade1615429 completed: Requests1724/1766 BOTH pass, both exact original patch
hashes, zero errors. Provisional247/500 (49.4%) =228+15+2+2, remaining outcomes
retained. Still not a corrected-agent comparison; original agent damage persists.
Evidence base_audit_v2/dual_http_regrade.json.
Django image inspection1615431 completed: expected admin templates exist and admin
is in ALWAYS_INSTALLED_APPS. Separate diagnostic1615432 runs auth_tests,
generic_inline_admin and validators modules with gold to detect full-suite
interference. THIS IS A DIAGNOSTIC SUBSET, NEVER an accepted capability score,
gold-valid task reclassification, or denominator exclusion. Original full-suite
failure remains unresolved. All original attempts and test expectations retained.


### Combined repair regression survey and Django finding
Django isolation1615432 completed:867 tests pass, including all seven auth template
tests that fail in full suite. Full-suite interference remains unresolved; subset
result is NOT an accepted gold score or exclusion. Full-suite module order is not
alphabetical; do not infer missing template files from earlier failures.

Combined repair gold survey1615435 (array0-3) runs all492 non-Requests tasks under
snapshot/appdirs/source-path/verbose-summary/identity repairs. Tests combined
changes against previously passing controls; no model calls. Output:
data/gold-repairs-survey-1615435/shard{0..3}. Source frozen while running.
First wrapper1615433 lacked --array and exited before Python with unbound
SLURM_ARRAY_JOB_ID; zero evaluation samples, replaced by explicit --array=0-3.
Trained arrays remain HELD; provisional247/500 unchanged; no publication yet.
