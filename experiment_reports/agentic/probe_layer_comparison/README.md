# Role-probe layer comparison on agentic trajectories

## Motivation & RQ

How well do fixed role probes transfer to realistic on-policy benchmark traces and off-policy Dolci tool-use data?

## Evaluation setup

GPT-OSS-20B, revision `6cee5e81ee83917806bbde320786a8fb61efebee`; zero-based post-attention-layernorm / pre-MLP probes. Replay uses the checkpoint dequantized to BF16 in Transformers, matching probe fitting; benchmark generation uses vLLM MXFP4. Ground truth comes from structured roles and analysis channels. Accuracy is argmax correctness, not mean true-role probability. Structural tokens and synthetic tool-schema context are excluded from scores. All content tokens are retained; long traces use explicitly recorded 16,384-token context windows with position/context resets. Shorter trajectories preserve causal context throughout. This is standardized retrospective Harmony replay, not a byte-identical reconstruction of serving prompts. Dolci has no analysis-channel labels. Probe fitting warnings in the baseline remain applicable.

## Factual observations

| Dataset | Layer | Trajectories | Tokens | Token accuracy | Mean trajectory accuracy | Windowed traces |
|---|---:|---:|---:|---:|---:|---:|
| dolci_tool_use | 4 | 1000 | 1,516,257 | 65.90% | 43.55% | 11 |
| dolci_tool_use | 8 | 1000 | 1,516,257 | 29.03% | 48.75% | 11 |
| dolci_tool_use | 12 | 1000 | 1,516,257 | 29.17% | 43.84% | 11 |
| dolci_tool_use | 16 | 1000 | 1,516,257 | 30.17% | 47.91% | 11 |
| dolci_tool_use | 20 | 1000 | 1,516,257 | 32.98% | 45.88% | 11 |
| tau2_retail | 4 | 114 | 2,876,950 | 17.52% | 21.88% | 79 |
| tau2_retail | 8 | 114 | 2,876,950 | 71.95% | 69.45% | 79 |
| tau2_retail | 12 | 114 | 2,876,950 | 29.91% | 31.60% | 79 |
| tau2_retail | 16 | 114 | 2,876,950 | 35.51% | 39.53% | 79 |
| tau2_retail | 20 | 114 | 2,876,950 | 67.79% | 60.83% | 79 |
| swe_bench | 4 | 500 | 7,265,153 | 46.71% | 55.97% | 144 |
| swe_bench | 8 | 500 | 7,265,153 | 45.02% | 36.76% | 144 |
| swe_bench | 12 | 500 | 7,265,153 | 26.98% | 28.43% | 144 |
| swe_bench | 16 | 500 | 7,265,153 | 21.71% | 23.00% | 144 |
| swe_bench | 20 | 500 | 7,265,153 | 41.38% | 34.95% | 144 |

dolci_tool_use: always predicting the most frequent ground-truth role yields 70.20% token accuracy; uniform random prediction over the fixed five classes yields 20%. Role-balanced accuracy and mean trajectory accuracy address different aspects of the substantial token imbalance.

tau2_retail: always predicting the most frequent ground-truth role yields 83.91% token accuracy; uniform random prediction over the fixed five classes yields 20%. Role-balanced accuracy and mean trajectory accuracy address different aspects of the substantial token imbalance.

swe_bench: always predicting the most frequent ground-truth role yields 46.37% token accuracy; uniform random prediction over the fixed five classes yields 20%. Role-balanced accuracy and mean trajectory accuracy address different aspects of the substantial token imbalance.

Benchmark task performance (distinct from probe accuracy):

| Dataset | Scorer | Successes / scored | Observed | Scored success rate | Errors |
|---|---|---:|---:|---:|---:|
| tau2_retail | retail_scorer | 73 / 114 | 114 | 64.04% | 0 |
| swe_bench | swe_bench_scorer | 11 / 274 | 500 | 4.01% | 226 |

Scored-only success rates exclude errored samples; full summaries also give errors-as-failures rates and confidence intervals. Interpret serving failures, context/time/token limits, and harness differences before comparing with published model scores. Dolci has no task-success metric because GPT-OSS did not generate these static trajectories.

See [confusion matrices](APPENDIX.md#confusion-matrices) and [full metrics](metrics.json) for per-role recall, mean true-role probabilities, individual trajectories, and results restricted to traces fitting one context window.

## Message

There is no uniformly reliable layer. Layer 4 has the highest token accuracy on Dolci (65.90%) and SWE (46.71%), while layer 8 leads retail (71.95%). The corresponding majority-role baselines are 70.20%, 46.37% and 83.91%: only SWE layer 4 marginally exceeds that token-frequency baseline, by 0.34 percentage points.

The layers capture different roles. Layer 4 tool recall is 79.77% on Dolci, 93.84% on retail and 83.38% on SWE, but reasoning recall is only 12.94% on retail and 12.23% on SWE. Layer 8 reverses that pattern: reasoning recall is 80.70% / 87.97%, while tool recall is 3.98% / 1.76%, respectively. Layer 8 also has the highest role-balanced accuracy on every dataset (48.93%, 44.93%, 40.37%), despite not always leading token accuracy. See [Dolci](APPENDIX.md#dolci_tool_use-layer-4), [retail](APPENDIX.md#tau2_retail-layer-4) and [SWE](APPENDIX.md#swe_bench-layer-4) matrices for the full role-by-layer comparisons.

Equal trajectory weighting changes the Dolci ranking: layer 8 leads at 48.75%, whereas layer 4 is 43.55%. Long traces and role imbalance matter substantially. Results restricted to single-window traces are available in metrics and the viewer; those are a different subset, not a controlled test of windowing. In particular, 79/114 retail and 144/500 SWE histories exceed one replay window.

All five probes use the exact same tokenized histories and score masks. Layers 4/8/20 were fitted with the original baseline training scale and split in the [additional-layer experiment](../../role_probes/gptoss20b_layers/README.md); their optimizer warnings are recorded there. These exploratory weights should not be treated as a validated general-purpose role classifier.


These measurements describe transfer under the documented replay and context-window policy. Differences between Dolci and generated benchmarks combine policy, task, role-frequency, formatting and trajectory-length effects; they do not isolate an on-policy causal effect. Dolci is an evaluation sample here, not probe training data. Benchmark task success is distinct from role-probe accuracy.

## Reproduction and standalone viewer

The [standalone five-layer viewer](viewer.html) is 1.73 GB (decimal). It passed an offline Chromium check across every dataset and layer, including each longest trajectory. It embeds all prediction data and works without network access in a modern browser. It supports dataset, layer and trajectory selection, role-colored tokens, ground-truth/predicted coloring, mismatch filtering and token probability inspection. The file contains data and is deliberately excluded from Git. See [RUNNING.md](RUNNING.md) for commands and data paths.
