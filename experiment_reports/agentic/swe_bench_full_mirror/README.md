# Full SWE-bench Verified: pinned public-cache images

## Motivation & RQ

Complete the full 500-task benchmark and produce on-policy role-probe trajectories after the first attempt hit Docker Hub pull limits.

## Evaluation setup

Same stock Inspect solver/scorer and Apptainer sandbox. Images are the official `swebench` images cached at `mirror.gcr.io`, with all 500 digests recorded in [image_manifest.json](../swe_bench_full/image_manifest.json). This is not a change to an independently rebuilt image collection. Four shards, eight concurrent samples per shard. Generation run IDs: 1461197 (shard 0), 1461195 (1), 1461196 (2), 1461193 (3).

## Factual observations

All 500 image manifests resolved successfully. Full evaluation is running.

## Message

Model performance and probe transfer interpretation await completed runs.

See [RUNNING.md](RUNNING.md).
