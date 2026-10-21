# Full SWE-bench Verified: GPT-OSS-20B

## Motivation & RQ

Measure model task performance and collect realistic on-policy trajectories for role-probe transfer evaluation.

## Evaluation setup

GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`, high reasoning, temperature 0, seed 123, vLLM 0.11.0. One selected model attempt per task; all dataset tasks included. Infrastructure failures before model execution may be retried, with both attempts retained in provenance. Per-call output limit 16,384; per-sample cumulative token limit 500,000 and wall-time limit 1,800 seconds. Source manifests record exact versions, task IDs, code hashes, Slurm jobs and model configuration.

All 500 SWE-bench Verified tasks, pinned dataset revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Stock Inspect solver/scorer, 30-message cap; four shards with eight concurrent samples each. Validated Apptainer isolation and gold-patch scoring. Official images are pinned by digest in [image_manifest.json](image_manifest.json) and retrieved from Google’s public Docker Hub cache. The earlier rate-limited attempt is preserved but excluded from these results.

## Factual observations

| Scorer | Successes / scored | All tasks | Scored rate | Errors as failures | 95% Wilson CI (scored) |
|---|---:|---:|---:|---:|---|
| swe_bench_scorer | 11 / 274 | 500 | 4.01% | 2.20% | 2.26%–7.04% |

Sample errors: 226; limited samples: 271; samples with generation-token caps: 73; failed agent model events: 774. Tool-bearing trajectories: 500/500. See [full metrics](metrics/summary.json), [provenance](metrics/provenance.json), and the [probe report](../probe_transfer/README.md) with its [confusion appendix](../probe_transfer/APPENDIX.md).

Two image failures were retried only after verifying zero prior model calls; both now have agent trajectories. Two malformed tool names were retained as explicit unknown-name metadata without rerunning the model. All 500 canonical records export successfully. All 226 remaining sample errors are model API retry failures. The four successful generation shards took 72–84 minutes concurrently on four B200 GPUs; the image retries took another 10 minutes.

## Message

This Inspect configuration does not reproduce OpenAI’s published 60.7% SWE-bench result. The published setup uses a different terminal harness and evaluation subset; this run uses stock Inspect tools, a 30-message cap, temperature 0 and a 32K serving context. Malformed generated Harmony headers cause additional API failures. The official repository recommends temperature 1/top-p 1; no sampling/harness change was made midway through this run. Do not interpret the observed score as the model’s best achievable SWE-bench performance. [OpenAI model card](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations), [model sampling recommendation](https://github.com/openai/gpt-oss).

The trace set includes both successful and unsuccessful/error-ending attempts. Probe metrics describe the retained conversational tokens, not task success.

See [RUNNING.md](RUNNING.md) for reproduction and standalone viewer generation.
