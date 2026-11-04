# Reproduction

From `/data/cluster/users/kortukov/role_probe_training`, use the existing locked
training and Inspect environments. Do not synchronize or alter shared environments
while other experiments run. No dependencies were added for this iteration.

```bash
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python src/experiments/training_against_role_probes/on_policy/checks.py
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python src/experiments/training_against_role_probes/on_policy/serving_check.py
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/on_policy/submit.py
```

[Configuration](../../../../src/experiments/training_against_role_probes/on_policy/config.json)
and [job script](../../../../src/experiments/training_against_role_probes/on_policy/train.sbatch).
Reusable implementation: [on_policy](../../../../src/role_probe_training/on_policy/).
Job IDs and exact submission commands are saved in [jobs.json](jobs.json).

The original pilot failed before update 1 and its dependent jobs were cancelled.
The intended pipeline remains: the first job performs update 1; a separate job restores its adapter, optimizer
and probe and performs update 2. A CPU gate checks both ranks, fresh episode
coverage, token records, initial zero KL, finite gradients, complete probe refits,
changed adapters and checkpoint provenance. Production can start only after that
gate succeeds. The gate also rejects throughput projected to exceed its 72-hour
allocation. Evaluation and publication are not yet submitted by this script.

One epoch contains 2,700 episodes, global batch 16, 169 updates (last batch 12).
Two B200 GPUs each host a full BF16/dequantized base with rank-64, alpha-128
attention/expert LoRA; generation and optimization use the same loaded model.
Generation uses temperature 1, top-p 1, top-k disabled, high reasoning, at most
8,192 new tokens per call, 32,768 serving context and 50 tool steps. AppWorld
runs in the existing process-isolated container, one world per rank at a time.
Tool execution and task scoring reuse the existing Inspect/AppWorld integration.

AdamW learning rate is 5e-5, weight decay 1e-4, gradient clip 1 and KL weight 16.
Model role loss is the existing full-history, content-token mean over all roles.
KL is normalized by all sampled assistant tokens, including framing/termination,
and uses hidden state t-1 to predict sampled token t. Every request preserves its
exact prompt and completion IDs; KL replays that complete request, never a
reconstructed or truncated prompt. Vocabulary projections are checkpointed in
chunks for memory, but attention histories are not split. Role training retains
the existing metadata-labelled whole-conversation replay convention.

Probe refits retain the original train/heldout passage split, all content tokens,
2,000 warm-start SGD updates, learning rate 1e-3, weight decay 1e-4, batch 32
passage-role examples and the existing warmup/cosine schedule. Validation on the
saved disjoint AppWorld development histories is diagnostic, not an on-policy
task-utility estimate. Final benchmark probes must be fitted afresh.

The base revision is `6cee5e81ee83917806bbde320786a8fb61efebee`.
Relevant upstream pins: prompt-injection-as-role-confusion
`ec333c40fd43fe991e1ebf66765051b6d7e35784`; training_against_probes
`9c3f80cc6d6d6fa2a434ca49ffdd150c4223c427`.
The original source corpus is `experiment_reports/training_against_role_probes/data/train2700/trajectories.jsonl`.
At startup, `runs/epoch1/manifest.json` records configuration, source/input/image
hashes, environment-lock hashes, Git/submodule revisions and hardware.
`schedule.json` fixes task/episode/seed order. Resume rejects changed provenance.

Large artifacts remain on cluster storage under this experiment's `runs/epoch1/`:
`rollouts/stepNNNNN/rankR/` contains exact request tokens, original Inspect logs,
AppWorld worlds, normalized trajectories and collection manifests. Each request
records the policy step that generated it. Every update saves adapter, optimizer
and refitted probe; only the two latest complete checkpoints are retained.
Model-generated failures are preserved. Collection/export errors stop the run for
audit rather than silently dropping or rerolling attempts. An interrupted batch
directory likewise blocks blind regeneration. This is scratch storage; archive
these artifacts before cleanup if needed for reproduction.

## Current repair status (2026-10-28)

Pilot 1610626 failed before any optimizer update. Jobs 1610627–1610629 were
cancelled before editing the on-policy source. All original attempts and errors
remain in `runs/epoch1/rollouts/step00001/`; none were silently discarded or
replaced. The original checkpoint's source hashes no longer match the repaired
source: do not blindly resume it or regenerate into the same rollout directory.

The template fix was tested against the exact saved rank-1 request and with a
real Inspect client performing a tool round trip. It removes only whitespace
from an empty text field when thinking is present, retaining meaningful text and
reasoning. The serving thread now explicitly selects its rank's CUDA device.
Exact input IDs are saved before generation, and requests after a server failure
are rejected without rerunning generation on a potentially poisoned CUDA context.
The CUDA root cause is not yet established; explicit device selection is not a
claim that the illegal access is resolved.

Diagnostic array 1691770 runs
[`generation_diagnostic.sbatch`](../../../../src/experiments/training_against_role_probes/on_policy/generation_diagnostic.sbatch)
on `ber-testing`: one GPU, four CPUs, 96 GiB host memory, 20 minutes per control,
serial concurrency. It compares flex attention in the serving thread, flex in the
main thread and eager attention in the serving thread. All replay the failed
rank-0 input with synchronous CUDA errors and at most 768 generated tokens, then
check reverse-KL backward after generation if generation succeeds. These short
diagnostic limits do not change production training budgets. Eager attention is
a diagnostic control only; it is not enabled for full-history production.
Results are retained in `runs/diagnostics/1691770/`. The colocated replacement pipeline is described below.

## vLLM serving study

See [serving design and benchmark](SERVING.md) for the comparison of vLLM/TRL
integration approaches and the matched small-scale serving benchmark.
Array 1691825 and summary 1691826 are separate from CUDA diagnostic 1691770.
The opt-in colocated trainer is now implemented and queued behind the diagnostic gates; no GPU colocation result is available yet.

## Two-GPU colocated pipeline (2026-10-29)

Submission: `uv run --frozen --no-sync python src/experiments/training_against_role_probes/on_policy/submit_colocated.py`.
The script refuses to overwrite an existing pipeline manifest. The current
[manifest](colocated_jobs.json) contains exact commands, IDs and source hashes.
Keep these sources and environments unchanged while the jobs are queued/running.

- **1691925**: two-GPU testing-partition diagnostic, 55 minutes, 16 CPUs, 256 GiB
  host memory. One complete episode per rank before and after one update; all
  standard probe passages and 2,000 refit steps. These diagnostic episodes are
  excluded from the 2,700-episode training epoch.
- **1691926 / 1691927**: batch-16 production pilots, six hours each, two GPUs.
  First update then checkpoint resume for the second update.
- **1691928**: CPU validation checks episode/token coverage, policy versions,
  finite gradients/losses, probe refits, probability comparisons, changed adapter,
  unchanged source/data provenance, and projected remaining runtime below 68 h.
- **1691929**: production continuation through update 169, at most 72 hours.
  It cannot start unless all preceding stages succeed.

The configuration is `src/experiments/training_against_role_probes/on_policy/colocated_config.json`.
Training outputs use `runs/colocated_epoch1/`; diagnostic outputs use
`runs/colocation/1691925/`. All attempts remain separate from failed original
`runs/epoch1/` artifacts. Each serving rank uses its own persistent vLLM process,
8 GiB KV cache budget and level-1 sleep during training/refits. Trainer weights
remain on the GPU. Transfers stage exact merged expert weights plus separate
FP32 attention adapters on node-local storage; no shared environment downgrade
or TRL trainer is involved. Full-vocabulary next-token comparisons require
matching argmax and KL below 0.1 nat, an engineering gate rather than proof of
identical distributions. Original batch size, loss, refit and episode limits
remain unchanged for production.

The submitted pipeline ends with training. Final model export and held-out task
evaluation are not wired into this new colocation pipeline yet.

## Isolated HF generation repair

The training/colocation source and environments are frozen while queued jobs use
them. The HF repair lives separately in `src/role_probe_training/generation/`
and `src/experiments/training_against_role_probes/hf_generation/`.
[HF job records](hf_generation_jobs.json) track the controls and benchmark.

Array 1691943 reproduces the trained-adapter crash with normal and synchronous
CUDA execution and tests the regular FlexAttention kernel. The synchronous trace
localizes the failure to compiled sliding-window block-mask construction, not
the MoE operation where the asynchronous trace first reported it. Forcing the
regular attention kernel did not fix the trained-adapter failure.

Array 1691975 tests ordinary PyTorch construction of single-query decode masks;
compiled prefill/training masks and FlexAttention itself remain unchanged. The
CPU mask checks cover 32 causal/window/padding/cache-offset combinations.
Benchmark 1691976 depends on the trained-adapter repair test passing. It uses the
same snapshot, task IDs, sampling parameters and budgets as vLLM benchmark
1691825_1, then scores both saved trajectory sets using the same trained HF model
and its fixed step-169 standard-data probe. Report role cross-entropy in nats,
content-token-weighted accuracy, whole-history scoring coverage, task success,
errors/limits and timing. The two-task, five-minute-per-task diagnostic is too
small to establish equivalent utility and may time-limit the slower backend.
HF serializes concurrent requests; vLLM batches them. Per-call sampling seeds
are derived differently, so exact sampled completions are not expected to match.

### HF disconnect repair rerun

Benchmark 1726415 uses the isolated
`src/role_probe_training/generation/serving.py` handler, retaining the original
sampling settings and task budgets. Submission:

```bash
.tools/bin/uv run --project src/experiments/training_against_role_probes/runtime --frozen --no-sync python src/experiments/training_against_role_probes/hf_generation/check_disconnects.py
env HF_DIAGNOSTIC_JOB=1691975 sbatch --export=ALL src/experiments/training_against_role_probes/hf_generation/disconnect_benchmark.sbatch
```

For the CPU regression command, use `uv run --project
src/experiments/training_against_role_probes/runtime --frozen --no-sync` to match
the training environment used for verification. Completed GPU outputs are in
`runs/hf_benchmark/1726415/`, including `comparison.json` from successful episode
collection/scoring. Disconnects are recorded in `tokens/disconnects.jsonl` and
cancelled generations in `tokens/requests.jsonl`; neither becomes a fatal model
error. Time-limited tasks remain in the denominators. No on-policy source or
shared environment was modified for this repair.

## Current segmented continuation

[segmented_jobs.json](segmented_jobs.json) is the authoritative replacement for
blocked job 1758052. Jobs 1798633–1798639 resume from step 2 and stop at steps
26, 50, 74, 98, 122, 146 and 169 respectively. Each requests two GPUs and 24 hours;
later jobs use `afterok` dependencies on the preceding segment. A failed segment
blocks its dependents and requires investigation; this is not automatic error recovery.
The smaller 270-episode comparisons retain higher scheduling priority.

The entry point is `role_probe_training.on_policy_segmented`; its job script and
checkpoint-specific `migration.json` live under
`src/experiments/training_against_role_probes/on_policy_segmented/`.
The first resume permits exactly the recorded step-2 checkpoint/source transition
and checks all other training provenance unchanged. Later resumes use strict
source equality. It imports the tested mixed-channel serving repair without
editing sources used by other queued jobs. The runtime writes
`runs/colocated_epoch1/serving_repair_transition.json` with before/after manifests.
Model, optimizer and probe states, original 2,700-episode schedule, and run output
location are retained. CPU checks verified step-2 optimizer/probe state, the exact
saved schedule, segment coverage and original source hashes; syntax checks passed. This code was removed from the tree; it is archived at commit `2e475d0`.

Training-time validation runs every five updates and at each segment boundary.
Final AppWorld/retail/SWE evaluation and probe scoring are now attached; see EVALUATION.md. Automatic GitHub publication remains separate. Completed benchmark results are copied
to the small tracked-ready `serving_comparison.json`; large raw logs and trajectories
remain under the paths above on cluster storage.

## Automatic final evaluation

See [EVALUATION.md](EVALUATION.md) and [evaluation_jobs.json](evaluation_jobs.json) for the submitted dependencies, exact commands and script hashes. The final checkpoint is exported once, then fresh probe fitting and benchmark generation proceed in parallel. Existing prepared uv runtimes are reused. Retail inherits Azure credentials via plain `--export=ALL`; no secrets are stored in the manifest.

Large merged models, probe features and assembled trajectories live under this experiment's `data/`; original logs live under `runs/`, per-token scores under `predictions/`. Fresh probe weights and small result JSON/Markdown files live under `post_training/`. These paths are relative to this experiment on cluster storage. SWE uses the existing native-context Inspect generous snapshot protocol, four shards at concurrency four, with 72 hours per shard; AppWorld and retail retain the existing comparison settings. GPU array concurrency is capped at two per collection family.

Each downstream job requires predecessor success. Infrastructure failures block affected results rather than silently dropping tasks. Result writers verify task counts, trajectory hashes and checkpoint identity. No additional training or GPU pilot was launched for this wiring.

## Current Harmony-parser restart (2026-10-31)

[harmony_jobs.json](harmony_jobs.json) supersedes the earlier training submission; [evaluation_jobs.json](evaluation_jobs.json) records the reconnected evaluations. The run resumes from update 2 into `runs/epoch1_harmony/`, preserving model, optimizer and probe state. A checkpoint-hashed migration verifies original sources and the unchanged task schedule. The two 270-episode runs precede the 2,700-episode continuation in user job priority.

The vLLM worker now invokes the installed vLLM 0.32 HarmonyParser with oss-harmony 0.0.11; no heuristic moves tool names from channels into recipients. Invalid/unsupported model output produces no executable tool call, ends the Inspect tool loop, and is scored on the actual AppWorld state. `parser_events.jsonl` and trajectory `invalid_model_outputs` record these events. Exact sampled tokens remain in `requests.jsonl` for reverse KL; retrospective role scoring covers the returned conversation, not discarded malformed segments. Such segments remain auditable in raw-token records. CUDA/transport/provenance failures remain fatal.

For the interrupted batch, `recovery.json` hashes saved completions. They are replayed only at identical prompt IDs, seed and policy step; mismatch aborts instead of drawing replacement samples. All attempts and failed-run artifacts remain retained. New generations begin after the saved prefix.

CPU regression checks covered both malformed headers, ordinary controls, the full-run failing history, real Inspect episode termination, HTTP isolation of failures, raw-token retention, checkpoint state and schedule identity. No additional GPU pilot was submitted. Test entry points are in `src/experiments/training_against_role_probes/on_policy_harmony/`. This code was removed from the tree; it is archived at commit `2e475d0`.

## Current whole-episode recovery (2026-11-01)

[recovery_jobs.json](recovery_jobs.json) supersedes the failed Harmony restarts. The active output is `runs/epoch1_recovery/`; model/optimizer/probe checkpoints still resume at steps 0, 3 and 2 for batch-64, batch-16 and full training respectively. [evaluation_jobs.json](evaluation_jobs.json) points exports and the existing AppWorld/retail/SWE/probe-scoring DAG to the replacement final checkpoints.

Re-executing saved tool calls changed unordered set output and Python object addresses, so exact prompt replay could not reproduce prior contexts. Recovery now stages audited same-policy trajectories and their exact generation tokens, never re-executing their tools. The short batch-64 run recovers 41 episodes and generates the remaining 23 for its first update; the interrupted batch-16 and full-run updates each recover all 16 episodes. `runs/epoch1_recovery/recovery.json` records cache hashes and source hashes.

Three terminal attempts retain original error records and are labelled `recovered_terminal`: two malformed-model-output attempts and one context-limited attempt. Their task scores come from the original Inspect log's AppWorld final-world metadata, using the same success criterion as the scorer. They are not rerolled or removed. Raw completion tokens remain in reverse KL; retrospective role supervision covers the saved returned conversation, with any unreturned malformed segment retained only in raw-token diagnostics.

Future context exhaustion records `episode_stops.jsonl`, ends that episode with zero generated tokens, and lets AppWorld score the current world state. Nothing is truncated and the server remains usable. Infrastructure/model-compute failures remain fatal. Validation evidence is in [recovery_validation.json](recovery_validation.json); executable checks and submission code are under `src/experiments/training_against_role_probes/on_policy_recovery/`.
