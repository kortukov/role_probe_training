# SWE-bench full collection: intermediate run

## Motivation & RQ

Can the complete SWE-bench dataset be executed after image-download limits interrupted the initial collection?

## Evaluation setup

We keep the same 500 tasks, agent and test-based scoring while retrieving the official task environments through a public image cache. This is an infrastructure recovery step, not a new scientific comparison. Coverage means the number of tasks for which a selected agent history is available.

## Factual observations

All four evaluation shards completed. Two unavailable image layers required retries before model execution; two malformed tool names required repairs to the saved message representation. The resulting 500-task collection and scores are reported in the [final SWE experiment](../swe_bench_full/README.md).

## Message

The collection is complete after explicit repairs. Use the final report for scientific conclusions; this directory preserves intermediate attempts and their provenance.

See [RUNNING.md](RUNNING.md) for image sources, job records and repair details.
