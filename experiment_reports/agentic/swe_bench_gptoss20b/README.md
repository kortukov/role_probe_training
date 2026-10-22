# SWE-bench Verified pilot preparation

Historical preparation record; subsequent execution is reported in the [full SWE experiment](../swe_bench_full/README.md).

## Motivation & RQ

Can a small software-repair pilot provide task scores and tool-use histories in the same format as the retail evaluation?

## Evaluation setup

We selected five tasks from SWE-bench Verified for GPT-OSS-20B. The planned agent would inspect and modify each repository, with the benchmark test suite determining success. The intended primary metric was the fraction of selected tasks solved. Five tasks would test the collection pipeline, not provide a precise capability estimate.

## Factual observations

The dataset and task selection were prepared. No model inference or task scoring was performed in this pilot: the cluster's available container runtime first needed integration with Inspect.

## Message

This preparation established the inputs but provided no performance evidence. Runtime integration and validation were subsequently completed for the full experiment; do not treat this unexecuted pilot as a benchmark result.

See [RUNNING.md](RUNNING.md) for the historical preparation command and a link to the current execution workflow.
