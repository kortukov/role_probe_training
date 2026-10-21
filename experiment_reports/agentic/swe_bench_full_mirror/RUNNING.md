# Reproduction

The pinned image map is `../swe_bench_full/image_manifest.json`; `src/experiments/agentic_eval/check_images.py` documents resolution but rerunning it may change latest-tag digests. Reuse the saved map for reproduction.

```bash
sbatch --array=0-3 src/experiments/agentic_eval/swe_mirror_job.sbatch
```

The SWE client/server environments and upstream dataset revision are frozen in source locks and preparation/run manifests. Each run retains its full Inspect log, normalized trajectories, per-call records, task scores and error/limit counters.
