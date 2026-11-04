# Baseline validation

## Newly discovered grading failure — 2026-10-27

All 34 Matplotlib tasks failed. In 33, the fresh-image pytest runner crashed in
`pytest_rerunfailures.ServerStatusDB` at `self.sock.bind(("localhost", 0))`, with
`socket.gaierror: [Errno -3] Temporary failure in name resolution`. The other attempt
had an empty patch. Thirteen agent trajectories also contain this failure during
their own tool use. A gold-patch control on `matplotlib__matplotlib-20859` reproduces
the same failure (job 1610621), independently of any model-generated patch.

The scorer only checked for start/end markers, which the shell script still prints
after pytest initialization crashes. Therefore zero reported evaluation errors
did not establish that tests ran. Baseline acceptance is revoked and the five
trained collection arrays are held. Requests also has connection-related regressions. Gold controls (1610623) on
Requests1724 and Requests2931 both score zero with no sample errors. In Requests2931
the gold patch passes the issue-specific test but fails two required regression
tests; Requests1724 has 35 regression failures even with the gold patch. This
confirms additional grading-environment problems; the full eight-task impact
requires corrected regrading.

The 33 known grading failures could contribute at most 6.6 percentage points to
task success if every affected patch were correct; this is an upper bound, not a
corrected score. Agent-side effects need separate assessment.

[Per-task grading audit](base_audit_v2/grading_audit.json). Raw test tails and
controls are retained under `base_audit_v2/data/` and `data/grading-controls-1610621/`.

## Localhost correction controls

The isolated image has an empty `/etc/hosts`. A numeric-loopback connection works,
while resolving `localhost` fails. An opt-in provider supplies localhost entries
without enabling external networking. With this change, gold patches for both
Matplotlib20859 and Matplotlib24570 pass (89 and 277 tests respectively); Django14155
and SymPy13878 also pass their gold verdicts. Controls job: 1610625.

Requests1724 and Requests2931 still fail. Full output identifies external httpbin
calls in the older suite, a missing `pytest-httpbin` fixture in the newer suite,
and timeout tests receiving `Network is unreachable` for 10.255.255.1 rather than
a timeout. These are separate from localhost resolution. Upstream reports also
identify [Requests external-service problems](https://github.com/SWE-bench/SWE-bench/issues/484)
and [timeout-test networking sensitivity](https://github.com/SWE-bench/SWE-bench/issues/488).

The completed localhost regrade recovered **15/34 Matplotlib successes**; all
eight Requests patches remain unsuccessful. All 15 passing patches have identical
original/re-exported SHA256 hashes. No new model calls were made. Adding these 15
to the original 228 gives **243/500 (48.6%)**, provisionally keeping all other scores
unchanged. This is a scoring correction, not a fresh corrected-agent evaluation.
[Per-attempt regrade](base_audit_v2/localhost_regrade.json).

The all-eight Requests gold survey passes only Requests6028. A local HTTP service
and pytest-httpbin fixture restore many individual tests but no additional full
gold verdicts; TLS expectations, timeout semantics and compatibility issues remain.
[Gold controls](base_audit_v2/requests_gold.json). A 492-task gold survey checks the
remaining repositories before any new baseline acceptance. Collection remains held.

## Initial-image patch contamination

A separate control records the initial repository tree, including pre-existing
untracked build files, before applying the gold patch. Requests1142 then passes
its gold verdict with an 800-byte exported patch; Requests6028 remains passing.
The earlier gold export had included pre-existing build output. This confirms
another grading-path defect, separate from networking. Original model patches
and trajectories are unchanged. [Snapshot controls](base_audit_v2/snapshot_controls.json).

## Earlier validation (superseded by the grading audit)

The second baseline has 228/500 successes, zero evaluation/export errors, eight
empty patches and 20 native-context stops. No sample-level time, message or
cumulative-token limit events occurred. Raw-log scores and exported trajectory
scores agree on every task. All 500 requested high reasoning.

There were 113 attempts with traceback-style import errors and 297 with tool
errors. These counts do not establish harness failures. Inspected examples include
a model removing the repository from its Python path, importing optional NumPy
in pinned SymPy/pytest environments, and importing Requests for a network lookup
in a scikit-learn task. Interpreter controls passed 20/20 checks across four
repositories before generation. Six attempts tried an unavailable `apply_patch`
command; the provided editing tool is `text_editor`.

The residual model-card gap remains unresolved. The 500-task dataset, Inspect
Python/shell/editor interface and serving implementation do not reproduce the
reference setup. The initial acceptance of this protocol was revoked by the subsequent grading
audit; it was never evidence of model-card parity. No failed attempt was replaced.

Sources: [current task results](../base_model/post_training/swe_corrected_v2_task_results.json),
[current audit](base_audit_v2/audit.json), [pipeline](pipeline.json).

## Earlier protocol diagnosis

The completed 500-task cohort has 226 successes, one scoring timeout, 13 empty
patches, no message-limit stops and three native-context stops. The cumulative
10-million-token budget stopped 67 attempts; eight already had passing patches.
This budget sums input and output tokens across calls, including repeated history;
it is distinct from the model's 131,072-token context window. Even converting all
59 failed budget-limited attempts and the scoring error into successes would give
57.2%, so these cases alone cannot reach the 60.7% reference. This is an upper
bound, not an estimate of the benefit of increasing the budget.

Fresh-image controls confirmed that Inspect's Python tool calls `python3`. On
`psf__requests-1724`, the activated `testbed` environment has a legacy Python;
`python3` instead resolves to base Conda and fails to import the older Requests
code. In the checked SymPy and pytest images, tools correctly select `testbed`,
but that environment does not contain NumPy. Import-error counts therefore mix
model mistakes, optional dependencies and environment problems; they must not be
interpreted as counts of harness-caused failures.

The single scoring timeout is `sphinx-doc__sphinx-7590`. Replaying its saved editor actions
in a fresh image reproduced the 1,800-second timeout during the test phase, after
one test passed. The gold patch passed all 25 tests in 4.02 seconds under the same
setup. This points to the model edits causing the hang. The diagnostic made no
new model calls and does not replace the original unsuccessful attempt.
The trained-model comparison and percentage role-confusion matrices are pending.

Sources: [task results](../base_model/post_training/swe_corrected_task_results.json),
[full audit](base_audit/audit.json), and [pipeline state](pipeline.json).


## Full gold survey

The non-Requests survey passes483/492 gold controls. Five failures involve
external HTTP; one lacks appdirs; one imports an installed Django package;
two show passing test output but unmatched test identifiers/verdicts. These
remain evaluation defects under investigation, not model failures.
[Control results](base_audit_v2/gold_survey.json). All500 model attempts remain
in the task denominator. No corrected six-model result is accepted yet.

Pylint4661 control1615411 completed: gold passes after supplying appdirs1.4.4; empty unchanged patch scores0; no sample errors. Small evidence base_audit_v2/appdirs_controls.json. Parser/source-path1615409 still running; watcher rearmed for this thread.


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
