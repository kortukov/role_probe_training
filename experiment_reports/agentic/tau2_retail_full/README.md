# Full τ²-bench retail: GPT-OSS-20B

## Motivation & RQ

Measure model task performance and collect realistic on-policy trajectories for role-probe transfer evaluation.

## Evaluation setup

GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`, high reasoning, temperature 0, seed 123, vLLM 0.11.0. One attempt per task in this selected run; all dataset tasks included. Per-call output limit 16,384; per-sample cumulative token limit 500,000 and wall-time limit 1,800 seconds. Source manifests record exact versions, task IDs, code hashes, Slurm jobs and model configuration.

114 τ²-bench retail tasks; Azure GPT-4.1 customer simulator (`gpt-4.1-2025-04-14`), 16 concurrent samples. The original upstream 100-turn cap is retained.

## Factual observations

| Scorer | Successes / scored | All tasks | Scored rate | Errors as failures | 95% Wilson CI (scored) |
|---|---:|---:|---:|---:|---|
| retail_scorer | 73 / 114 | 114 | 64.04% | 64.04% | 54.90%–72.25% |

Sample errors: 0; limited samples: 4; samples with generation-token caps: 64; failed agent model events: 32. Tool-bearing trajectories: 112/114. See [full metrics](metrics/summary.json), [provenance](metrics/provenance.json), and the [probe report](../probe_transfer/README.md) with its [confusion appendix](../probe_transfer/APPENDIX.md).

## Message

This measures τ² retail, whereas OpenAI’s published high-reasoning GPT-OSS-20B retail score is 54.8% on the original τ-bench. Treat that as context, not a matched reproduction. Time/token limits and malformed model headers can affect this run. [OpenAI model card](https://deploymentsafety.openai.com/gpt-oss/model-architecture-data-training-and-evaluations).

The trace set includes both successful and unsuccessful/error-ending attempts. Probe metrics describe the retained conversational tokens, not task success.

See [RUNNING.md](RUNNING.md) for reproduction and standalone viewer generation.
