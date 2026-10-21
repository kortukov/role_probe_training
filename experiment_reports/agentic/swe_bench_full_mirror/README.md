# Full SWE-bench Verified: pinned public-cache images

## Motivation & RQ

Complete the full 500-task benchmark and produce on-policy role-probe trajectories after the first attempt hit Docker Hub pull limits.

## Evaluation setup

Same stock Inspect solver/scorer and Apptainer sandbox. Images are the official `swebench` images cached at `mirror.gcr.io`, with all 500 digests recorded in [image_manifest.json](../swe_bench_full/image_manifest.json). This is not a change to an independently rebuilt image collection. Four shards, eight concurrent samples per shard. Generation run IDs: 1461197 (shard 0), 1461195 (1), 1461196 (2), 1461193 (3).

## Factual observations

All 500 image manifests resolved successfully. All four shards completed. Two missing image blobs required infrastructure-only retries; two malformed tool names required normalization repairs.

## Message

The selected, repaired 500-task result and interpretation are in the [final SWE report](../swe_bench_full/README.md). This directory preserves the original shard outputs before those repairs.

See [RUNNING.md](RUNNING.md).
