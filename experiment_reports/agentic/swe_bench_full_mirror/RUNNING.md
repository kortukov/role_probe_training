# Reproduction

The pinned image map is `../swe_bench_full/image_manifest.json`; `src/experiments/agentic_eval/check_images.py` documents resolution but rerunning it may change latest-tag digests. Reuse the saved map for reproduction.

```bash
sbatch --array=0-3 src/experiments/agentic_eval/swe_mirror_job.sbatch
```

The SWE client/server environments and upstream dataset revision are frozen in source locks and preparation/run manifests. Each run retains its full Inspect log, normalized trajectories, per-call records, task scores and error/limit counters.

## Recorded execution settings

Same stock Inspect solver/scorer and Apptainer sandbox. Images are the official `swebench` images cached at `mirror.gcr.io`, with all 500 digests recorded in [image_manifest.json](../swe_bench_full/image_manifest.json). This is not a change to an independently rebuilt image collection. Four shards, eight concurrent samples per shard. Generation run IDs: 1461197 (shard 0), 1461195 (1), 1461196 (2), 1461193 (3).

## Final selection

The four original shard outputs are intermediate artifacts. Finalization and the two infrastructure-only retries are documented in [full SWE reproduction](../swe_bench_full/RUNNING.md); use its consolidated data for analysis.
