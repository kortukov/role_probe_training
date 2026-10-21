# Layer-12 role-probe transfer to agentic trajectories

## Motivation & RQ

How well do fixed role probes transfer to realistic on-policy benchmark traces and off-policy Dolci tool-use data?

## Evaluation setup

GPT-OSS-20B, revision `6cee5e81ee83917806bbde320786a8fb61efebee`; zero-based post-attention-layernorm / pre-MLP probes. Replay uses the checkpoint dequantized to BF16 in Transformers, matching probe fitting; benchmark generation uses vLLM MXFP4. Ground truth comes from structured roles and analysis channels. Accuracy is argmax correctness, not mean true-role probability. Structural tokens and synthetic tool-schema context are excluded from scores. All content tokens are retained; long traces use explicitly recorded 16,384-token context windows with position/context resets. Shorter trajectories preserve causal context throughout. This is standardized retrospective Harmony replay, not a byte-identical reconstruction of serving prompts. Dolci has no analysis-channel labels. Probe fitting warnings in the baseline remain applicable.

## Factual observations

| Dataset | Layer | Trajectories | Tokens | Token accuracy | Mean trajectory accuracy | Windowed traces |
|---|---:|---:|---:|---:|---:|---:|
| dolci_tool_use | 12 | 1000 | 1,516,257 | 29.17% | 43.84% | 11 |
| tau2_retail | 12 | 114 | 2,876,950 | 29.91% | 31.60% | 79 |
| swe_bench | 12 | 500 | 7,265,153 | 26.98% | 28.43% | 144 |

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

Layer 12 does not transfer reliably under this replay protocol: token accuracy is 29.17% on Dolci, 29.91% on retail and 26.98% on SWE, below each dataset's majority-role baseline. Role-balanced accuracy is 40.37%, 33.08% and 34.79%, respectively (Dolci averages four observed roles; the others five).

Tool recognition is especially weak: recall is 12.24% on Dolci, 1.14% on retail and 3.73% on SWE. Retail reasoning is classified as ordinary assistant output 66.8% of the time; SWE tool tokens are classified as user tokens 57.1% of the time. These are structured-label comparisons, not labels inferred from text. See the separate matrices for [Dolci](APPENDIX.md#dolci_tool_use-layer-12), [retail](APPENDIX.md#tau2_retail-layer-12), and [SWE](APPENDIX.md#swe_bench-layer-12).

The [five-layer experiment](../probe_layer_comparison/README.md) evaluates layers 4, 8, 12, 16 and 20 on exactly the same trajectories. The 1,000-row [Dolci subset](../dolci_1000/README.md) can support a future tool-role training experiment, but contains no analysis-channel ground truth and cannot alone cover the reasoning class. Reserve a disjoint evaluation split if using it for training.


These measurements describe transfer under the documented replay and context-window policy. Differences between Dolci and generated benchmarks combine policy, task, role-frequency, formatting and trajectory-length effects; they do not isolate an on-policy causal effect. Dolci is an evaluation sample here, not probe training data. Benchmark task success is distinct from role-probe accuracy.

## Reproduction and standalone viewer

The [standalone layer-12 viewer](viewer.html) is 470 MB (decimal) and passed an offline Chromium check across all datasets, including each longest trajectory. It embeds all prediction data and works without network access in a modern browser. It supports dataset, layer and trajectory selection, role-colored tokens, ground-truth/predicted coloring, mismatch filtering and token probability inspection. The file contains data and is deliberately excluded from Git. See [RUNNING.md](RUNNING.md) for commands and data paths.
