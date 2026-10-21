# Role-probe layer comparison on agentic trajectories

## Motivation & RQ

How well do fixed role probes transfer to realistic on-policy benchmark traces and off-policy Dolci tool-use data?

## Evaluation setup

GPT-OSS-20B, revision `6cee5e81ee83917806bbde320786a8fb61efebee`; zero-based post-attention-layernorm / pre-MLP probes. Ground truth comes from structured roles and analysis channels. Accuracy is argmax correctness, not mean true-role probability. Structural tokens and synthetic tool-schema context are excluded from scores. All content tokens are retained; long traces use explicitly recorded 16,384-token context windows with position/context resets. Shorter trajectories preserve causal context throughout. This is standardized retrospective Harmony replay, not a byte-identical reconstruction of serving prompts. Dolci has no analysis-channel labels. Probe fitting warnings in the baseline remain applicable.

## Factual observations

| Dataset | Layer | Trajectories | Tokens | Token accuracy | Mean trajectory accuracy | Windowed traces |
|---|---:|---:|---:|---:|---:|---:|
| dolci_tool_use | 12 | 1000 | 1,516,257 | 29.17% | 43.84% | 11 |
| dolci_tool_use | 16 | 1000 | 1,516,257 | 30.17% | 47.91% | 11 |
| dolci_tool_use | 4 | 1000 | 1,516,257 | 65.90% | 43.55% | 11 |
| dolci_tool_use | 8 | 1000 | 1,516,257 | 29.03% | 48.75% | 11 |
| dolci_tool_use | 20 | 1000 | 1,516,257 | 32.98% | 45.88% | 11 |

dolci_tool_use: always predicting the most frequent ground-truth role yields 70.20% token accuracy; uniform random prediction over the fixed five classes yields 20%. Role-balanced accuracy and mean trajectory accuracy address different aspects of the substantial token imbalance.

See [confusion matrices](APPENDIX.md#confusion-matrices) and [full metrics](metrics.json) for per-role recall, mean true-role probabilities, individual trajectories, and results restricted to traces fitting one context window.

## Message

On the same 1,000 Dolci trajectories, layer 4 has the highest token accuracy (65.90%), but all layers fall below the always-tool majority baseline (70.20%). Its tool recall is 79.77%. Layer 8 leads mean trajectory accuracy (48.75%), followed by layer 16 (47.91%); therefore the token-weighted layer ranking does not carry over to equal trajectory weighting. The current evidence does not identify one uniformly reliable role probe. There are no Dolci analysis labels, and all fitted probes carry optimizer convergence caveats. Retail and SWE comparisons remain pending.

These measurements describe transfer under the documented replay and context-window policy. Differences between Dolci and generated benchmarks combine policy, task, role-frequency, formatting and trajectory-length effects; they do not isolate an on-policy causal effect. Dolci is an evaluation sample here, not probe training data. Benchmark task success is distinct from role-probe accuracy.

## Reproduction and standalone viewer

The local `viewer.html` embeds all prediction data and works without network access in a modern browser. It supports dataset, layer and trajectory selection, role-colored tokens, ground-truth/predicted coloring, mismatch filtering and token probability inspection. The file contains data and is deliberately excluded from Git. See [RUNNING.md](RUNNING.md) for commands and data paths.
