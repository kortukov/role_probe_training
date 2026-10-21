# GPT-OSS-20B on SWE-bench Verified: pilot preparation

Status: dataset prepared; execution runtime not validated. Date: 2026-10-21.

## 1. Motivation & RQ

Can a small software-engineering pilot provide task scores and tool-bearing
trajectories in the same format as retail for role-probe analysis?

## 2. Evaluation setup

Inspect Evals 0.25.0 `swe_bench`; `princeton-nlp/SWE-bench_Verified`, test split,
revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Five tasks selected with seed
123 from 500 sorted IDs; [manifest](preparation/manifest.json). Planned model:
GPT-OSS-20B, same revision and serving settings as retail. Use the default Inspect
agent/scorer, official x86_64 instance images, and report resolved fraction,
Wilson interval, errors, timeouts and token usage. Five cases give pipeline
signal, not a reliable estimate of full SWE-bench performance. The published
high-reasoning reference is 60.7% in the
[model card](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations),
with a different scaffold/budget.

## 3. Factual observations

The pinned dataset and five task manifests were prepared successfully. No
benchmark inference or scoring has run. The checked cluster node has Apptainer
but no Docker command; stock Inspect's Docker sandbox cannot run there as-is.

## 4. Message — interpretation and conclusion

The same exporter and runner support this benchmark. Execution requires a
validated Docker-capable host/provider or a separately tested Apptainer adapter.
An under-hour cold run cannot yet be promised because image preparation and test
execution may dominate. Start with five tasks, two concurrent sandboxes, after
pre-pulling the selected images. Do not replace the test scorer with an LLM judge.

[How to run this experiment](RUNNING.md)
