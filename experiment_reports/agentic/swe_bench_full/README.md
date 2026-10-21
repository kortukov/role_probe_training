# Full SWE-bench Verified

## Motivation & RQ

Measure role-probe transfer to realistic agentic data and compare on-policy benchmark traces with off-policy Dolci trajectories.

## Evaluation setup

All 500 verified tasks, four disjoint seeded shards; eight concurrent samples per GPU; stock Inspect solver and scorer; Apptainer with writable job-local root filesystems, no host project/home binds and no network. Compatible swebench 4.1.0 in a separate locked client environment. Gold-patch oracle and Python/bash_session/text_editor smoke passed on django__django-14155 (score 1.0). Production images use upstream latest tags; results must retain runtime provenance and this mutable-image limitation.

## Factual observations

Run in progress. No final performance claim yet. Data and predictions are excluded from Git.

## Message

Interpretation awaits completed runs. Detailed confusion matrices belong in [APPENDIX.md](APPENDIX.md#confusion-matrices).

See [RUNNING.md](RUNNING.md) for reproduction and standalone-viewer generation.
