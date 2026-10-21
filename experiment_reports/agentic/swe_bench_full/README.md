# Full SWE-bench Verified

## Motivation & RQ

Measure role-probe transfer to realistic agentic data and compare on-policy benchmark traces with off-policy Dolci trajectories.

## Evaluation setup

All 500 verified tasks, four disjoint seeded shards; eight concurrent samples per GPU; stock Inspect solver and scorer; Apptainer with writable job-local root filesystems, no host project/home binds and no network. Compatible swebench 4.1.0 in a separate locked client environment. Gold-patch oracle and Python/bash_session/text_editor smoke passed on django__django-14155 (score 1.0). Production images use upstream latest tags; results must retain runtime provenance and this mutable-image limitation.

## Factual observations

The first full attempt exhausted Docker Hub’s unauthenticated pull quota; many tasks failed before agent execution. This is an infrastructure diagnostic, not a completed model-performance evaluation. A clean full rerun uses all 500 official images through Google’s public cache, pinned by digest in [image_manifest.json](image_manifest.json). Its runs are under `../swe_bench_full_mirror/runs/`. No final performance claim yet.

## Message

Interpretation awaits completed runs. Detailed confusion matrices belong in [APPENDIX.md](APPENDIX.md#confusion-matrices).

See [RUNNING.md](RUNNING.md) for reproduction and standalone-viewer generation.
