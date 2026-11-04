# Serving choice and small-scale benchmark

## Proposed integration

Use a persistent vLLM inference process while retaining the custom HF/PEFT
trainer and its differentiable probe loss. Keep the training loop synchronous:
finish every episode in the current batch, train once, refit the probe, transfer
the new weights, invalidate old caches, verify the new version, then collect the
next batch. No episode may span an optimizer update. The numerical agreement of
training and serving distributions must be checked separately from weight identity.

The user selected a two-GPU colocated prototype on 2026-10-29. Each GPU hosts
one training rank and a separate persistent vLLM process. Generation and training
alternate at complete-episode barriers. vLLM sleeps at level 1 during updates and
probe refits. The first implementation stages merged BF16 expert weights and
FP32 attention adapters through node-local safetensors; this is deliberately not
an in-memory IPC implementation. Its transfer cost must be measured.

The installed server is vLLM 0.32.0/PyTorch 2.13.0; the trainer uses PyTorch
2.9.1+cu128. These versions were read locally on 2026-10-28. The installed server
contains pause/resume and weight-update APIs, including version labels.

The [vLLM async guide](https://docs.vllm.ai/en/latest/training/async_rl/)
describes pausing with `keep`, which can mix old and new weights within a
completion. We should use an episode-level barrier instead: finish all tool
interactions before pausing the engine, rather than merely draining individual
generation requests. Version labels are audit metadata, not proof of correct
weights. Cache invalidation is required at a policy change.

The [RLHF guide](https://docs.vllm.ai/en/latest/training/rlhf/) links to the
[weight-transfer API](https://docs.vllm.ai/en/latest/training/weight_transfer/):
NCCL for dedicated inference GPUs and IPC for colocated processes. These can
avoid repeated model exports and server restarts. We still need an adapter-aware
weight source and receiver. Our current export merges expert deltas in BF16 and
keeps attention adapters in FP32; blindly merging all attention deltas or sending
ordinary PEFT parameter names would not preserve the established representation.

The [TRL guide](https://huggingface.co/docs/trl/en/vllm_integration) illustrates
server and colocated operation. Its documented compatibility range currently ends
at vLLM 0.31.0. Its provided trainer integration is not a drop-in replacement for
our custom loss, standard-data probe refits, or AppWorld tool loop. Borrow the
serving/weight-sync pattern, rather than switching this experiment to GRPO/RLOO.
[Sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/) is an option
for the colocated design; its memory and wake-up costs remain unmeasured here.

## Benchmark design

Array **1691825** compares HF/PEFT and vLLM on one B200 at a time in
`ber-testing`, with 55 minutes per backend and serial array concurrency.
Summary job **1691826** follows both attempts, including failures.

Both backends use the existing 2,700-trajectory same-batch model at update 169,
adapter SHA-256 `bf5def61b31ae13c7dcee646b39c38f3b47bfa3c00036e8742d0dfe4301cfcea`.
HF loads the original adapter; vLLM loads its existing checked BF16-expert export
with separate FP32 attention adapters. This tests a nonzero adapter snapshot,
not a new training run. The prototype does not change production settings.

- Cold model/server startup is recorded separately from a warm-up request.
- Short generation uses four requests with identical task prompts, a 128-token
  output cap, temperature 1 and top-p 1, at concurrency one and four. Early stops
  are allowed; measured output-token counts and per-request timings are retained.
- Episode tests use the same two training tasks (`b7a9ee9_1`, `82e2fac_1`) at
  concurrency one and two. Each backend gets 50 tool steps, 8,192 tokens per
  generation, 32,768 context and a **diagnostic five-minute task budget**.
  All attempts, failures and limits are retained. These are serving diagnostics,
  not task-utility estimates or members of the training corpus.
- vLLM prefix caching is disabled so repeated benchmark prompts do not receive
  an advantage unavailable to the current HF path.

The benchmark measures serving throughput and small-batch episode wall time.
It does **not** measure live weight synchronization, prove backend numerical
equivalence, or measure co-resident training memory. No speedup is reported if
the HF CUDA path fails. Different episode lengths, tool actions and time-limit
hits must be considered when interpreting episode wall time.

For a subsequent integration test, verify two successive weight versions using
fixed token-prefix probability comparisons, complete updated attention/expert
weights, cache invalidation and policy-version checks. Measure synchronization
and training/refit time before estimating total update speedup:

`(HF rollout time + update/refit time) / (vLLM rollout time + sync time + update/refit time)`.

## Execution and artifacts

Code: [benchmark job](../../../../src/experiments/training_against_role_probes/on_policy/benchmark.sbatch),
[shared client](../../../../src/experiments/training_against_role_probes/on_policy/benchmark_client.py),
[HF driver](../../../../src/experiments/training_against_role_probes/on_policy/benchmark_hf.py)
and [summary](../../../../src/experiments/training_against_role_probes/on_policy/benchmark_summary.py).
Syntax and Python compilation checks passed. Instantiating the full vLLM CLI
parser on the CPU workspace could not infer a GPU device; vLLM GPU startup subsequently passed in job 1691825_1.

Saved outputs are under the cluster experiment directory at
`runs/serving_benchmark/1691825/{0,1}/`: startup/warm-up/micro timings, episode
timings, Inspect logs, original trajectories and server logs. `comparison.json`
is generated after both array elements finish. Job IDs are in [jobs.json](jobs.json).

[Experiment reproduction](RUNNING.md)

## Completed benchmark (2026-10-30)

HF job 1726415 and vLLM job 1691825_1 used the same trained checkpoint and
fixed refitted probe. Output throughput counts generated tokens divided by
wall time in four short requests (512 output tokens total per condition).

| Concurrent requests | HF tokens/s | vLLM tokens/s | vLLM/HF |
| --- | ---: | ---: | ---: |
| 1 | 4.66 | 275.91 | 59.2 |
| 4 | 5.36 | 378.15 | 70.5 |

HF serializes requests; vLLM batches them. These numbers compare the implemented
servers, not all possible HF implementations or complete training pipelines.

Each episode condition contains two assigned AppWorld tasks with a five-minute
per-task cap. Task success counts successes over all assigned attempts. Probe
cross-entropy (nats) and argmax role accuracy are content-token-weighted over
whole histories scored with the same HF model and fixed probe. All attempts
were role-scorable; no sample errors occurred.

| Backend | Concurrent episodes | Task successes | Probe cross-entropy | Role accuracy |
| --- | ---: | ---: | ---: | ---: |
| HF | 1 | 0/2 | 0.01187 | 99.59% |
| HF | 2 | 0/2 | 0.00302 | 99.96% |
| vLLM | 1 | 2/2 | 0.00364 | 99.91% |
| vLLM | 2 | 1/2 | 0.00175 | 99.97% |

All HF episodes reached the time cap; no vLLM episode did. Different sampled
histories, the small task count and unequal work completed before the time cap
prevent a utility-equivalence conclusion. HF's decoding-mask repair avoided the
observed CUDA failure, and disconnect handling allowed collection/scoring to finish.

The evidence supports vLLM for this generation setup. It does not establish that
training outcomes are equivalent. [Machine-readable results](serving_comparison.json)
include checkpoint hashes, full metric definitions and confusion counts.
Detailed matrices are in [the benchmark appendix](SERVING_APPENDIX.md).

[Experiment reproduction](RUNNING.md)
