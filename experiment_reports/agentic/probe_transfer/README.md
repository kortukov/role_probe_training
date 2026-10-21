# Layer-12 role-probe transfer to agentic trajectories

## Motivation & RQ

How well do fixed role probes transfer to realistic on-policy benchmark traces and off-policy Dolci tool-use data?

## Evaluation setup

GPT-OSS-20B, revision `6cee5e81ee83917806bbde320786a8fb61efebee`; zero-based post-attention-layernorm / pre-MLP probes. Ground truth comes from structured roles and analysis channels. Accuracy is argmax correctness, not mean true-role probability. Structural tokens and synthetic tool-schema context are excluded from scores. All content tokens are retained; long traces use explicitly recorded 16,384-token context windows with position/context resets. Shorter trajectories preserve causal context throughout. This is standardized retrospective Harmony replay, not a byte-identical reconstruction of serving prompts. Dolci has no analysis-channel labels. Probe fitting warnings in the baseline remain applicable.

## Factual observations

| Dataset | Layer | Trajectories | Tokens | Token accuracy | Mean trajectory accuracy | Windowed traces |
|---|---:|---:|---:|---:|---:|---:|
| dolci_tool_use | 12 | 1000 | 1,516,257 | 29.17% | 43.84% | 11 |

See [confusion matrices](APPENDIX.md#confusion-matrices) and [full metrics](metrics.json) for per-role recall, mean true-role probabilities, individual trajectories, and results restricted to traces fitting one context window.

## Message

On the completed 1,000-trajectory Dolci sample, layer 12 does not transfer reliably across roles: token accuracy is 29.17%, tool recall is 12.24%, and system recall is 2.60%; assistant recall is 85.86%. Tool responses account for 70.20% of scored tokens and strongly influence the token-weighted result. Mean trajectory accuracy is 43.84%. Eleven traces require multiple context windows; accuracy restricted to single-window traces is 38.93%. Dolci has no analysis labels, so it cannot by itself measure or supply supervised training targets for the cot class. Retail and SWE results are still pending and will be added to this report.

These measurements describe transfer under the documented replay and context-window policy. Differences between Dolci and generated benchmarks combine policy, task, role-frequency, formatting and trajectory-length effects; they do not isolate an on-policy causal effect. Dolci is an evaluation sample here, not probe training data. Benchmark task success is distinct from role-probe accuracy.

## Reproduction and standalone viewer

The local `viewer.html` embeds all prediction data and works without network access in a modern browser. It supports dataset, layer and trajectory selection, role-colored tokens, ground-truth/predicted coloring, mismatch filtering and token probability inspection. The file contains data and is deliberately excluded from Git. See [RUNNING.md](RUNNING.md) for commands and data paths.
