# Reproduction

Run from the repository on cluster storage. The isolated driver environment uses
Python 3.12 and AgentDojo 0.1.35; its exact dependencies are in
[uv.lock](../../../src/experiments/agentdojo_robustness/driver/uv.lock).
The existing server environment pins vLLM 0.32.0. GPU jobs use one B200,
four CPUs and 256 GiB RAM. The smoke test uses ber-testing; production uses
ber-gpu1 with a 24-hour budget. No existing user jobs are changed.

```bash
.tools/bin/uv sync --locked --project src/experiments/agentdojo_robustness/driver
.tools/bin/uv run --project src/experiments/agentdojo_robustness/driver --frozen --no-sync python src/experiments/agentdojo_robustness/check_driver.py
sbatch --export=ALL --partition=ber-testing --time=01:00:00 src/experiments/agentdojo_robustness/evaluate.sbatch trained smoke
# For a still-live smoke job, production can use afterok:
sbatch --export=ALL --dependency=afterok:SMOKE_JOB src/experiments/agentdojo_robustness/evaluate.sbatch base full
sbatch --export=ALL --dependency=afterok:SMOKE_JOB src/experiments/agentdojo_robustness/evaluate.sbatch trained full
.tools/bin/uv run --project src/experiments/agentdojo_robustness/driver --frozen --no-sync python src/experiments/agentdojo_robustness/summarize.py
```

The driver explicitly runs clean tasks before attacks in every suite. Four suites
run concurrently against one server. Generation uses temperature 1, medium
reasoning, seed 123, max_tokens 8192, max_model_len 12288, native OpenAI tool
parsing and the model's saved chat template. Completed AgentDojo results are
reused on restart. Rerunning a completed job does not intentionally regenerate
existing attempts. Errors and partial coverage remain visible in metrics.

Sources: [driver](../../../src/experiments/agentdojo_robustness/run.py),
[job](../../../src/experiments/agentdojo_robustness/evaluate.sbatch),
[summary](../../../src/experiments/agentdojo_robustness/summarize.py).

Checkpoint: `experiment_reports/training_against_role_probes/data_iteration/same_batch_27000/runs/training/epoch1/checkpoints/step01688`.
The exact export is under that experiment's `data/post_training/model/`.
Base revision `6cee5e81ee83917806bbde320786a8fb61efebee`; original adapter SHA256
`43969311ed93e3b15b1859906a774faa5933303e313976214cc1a9545048fa2f`.
The trained export has expert deltas merged and separate FP32 attention adapters,
loaded through `role_probe_training.post_training.vllm_worker.RoleWorker`.
The fresh base export is `experiment_reports/training_against_role_probes/base_model/data/swe_native/model/`.

Hugging Face sharing destination:
https://huggingface.co/kortukov/gpt-oss-20b-role-perception-appworld-27000
The Hub now contains only the original complete adapter, portable base-model
metadata and loading helpers. Transformers/PEFT resolves the pinned public base.
The supplied vLLM launcher reconstructs the exact evaluation export locally;
603 model and attention tensors match exactly (check job 1964572).
See [adapter validation](adapter_validation.json) and [Hub manifest](huggingface.json).

Historical project: `/data/cluster/users/kortukov/prompt_injection_interp`,
revision `2316d05167498feb7203ad5d1f6ff0016d66140e`.
The versioned `data/historical_base.tar.gz` copies its
`results/agentdojo_trajectories/v2base.tar.gz` archive. Summary generation reads
the archive directly, excluding injection-goal seed runs and workspace tasks
35/38. It reproduces the published baseline without rerunning it.

Outputs: `runs/{base,trained}/vllm_parsed/` contains full trajectory JSON;
`runs/smoke-trained/` contains validation artifacts; `logs/` and each run folder
contain scheduler/server/suite logs. These stay on cluster storage and are
ignored in Git. The small baseline archive, metrics, report and manifests are
versioned. `jobs.json` records submitted IDs; `provenance.json` records hashes.

GitHub publication uses a dedicated branch, `agentdojo-same-batch-27000`.
The [publisher](../../../src/experiments/agentdojo_robustness/publish.py) creates
commits with a temporary Git index containing only this experiment's report and
code. It leaves the shared working branch, staging area and unrelated edits
untouched. A CPU finalizer runs after both production jobs terminate, saves
coverage-aware tables even on failure, and pushes the branch. GitHub push access
was verified with the initial report publication. No pull request is merged.

Live submission: smoke 1964347 completed successfully. Slurm had already purged
its live job record, so production was submitted after checking its successful
accounting record rather than referencing it as a dependency. Base: 1964385;
trained: 1964386; final report: 1964387 (afterany on both production jobs).
Validation evidence is retained in [validation.json](validation.json).

Recovery: original full suites stopped on malformed model tool-call JSON.
The driver now retains invalid calls and returns the stock invalid-tool error
feedback, allowing model correction. Any other per-attempt exception is logged
in `runs/<model>/<suite>.failures.jsonl`; the remaining attempts continue.
Such errors receive no utility/attack credit and remain outside scored coverage.
Completed trajectories are reused. CPU integration exercises malformed tool
arguments and recovery; GPU smoke 1964575 passed. Current IDs are in jobs.json.

## Fresh AgentDojo role probes

The implementation is [agentdojo.py](../../../src/role_probe_training/post_training/agentdojo.py)
with [normalization](../../../src/experiments/agentdojo_roles/assemble.py),
[GPU job](../../../src/experiments/agentdojo_roles/evaluate.sbatch),
[CPU report job](../../../src/experiments/agentdojo_roles/finalize.sbatch) and
[automatic main-branch publisher](../../../src/experiments/agentdojo_roles/publish.py).
No model generations or task outcomes are changed. Both models receive new
independent layer-12 probes from the standard 250 C4/Dolma passages and original
train/test split; fitting uses every content token, no scaling, multinomial L2
logistic regression, C=0.005 and cuML max_iter=5000. Feature extraction uses BF16
models and FP32 attention adapters for trained; feature caches match the existing
FP16 standard-data workflow. The probe site is post-attention-layernorm/pre-MLP,
never decoder-block output. Stopping limitations are recorded with the probe.

```bash
.tools/bin/uv run --project src/experiments/agentdojo_robustness/driver --frozen --no-sync python src/experiments/agentdojo_roles/assemble.py
.tools/bin/uv run --project src/experiments/agentdojo_robustness/driver --frozen --no-sync python src/experiments/agentdojo_roles/check.py
sbatch --export=ALL src/experiments/agentdojo_roles/evaluate.sbatch base
sbatch --export=ALL src/experiments/agentdojo_roles/evaluate.sbatch trained
```

Each model job extracts training features, fits in the existing CUDA-13 cuML
interpreter, checks scoring on one whole clean and attacked history, then scores
both complete subsets with the PEFT/CUDA-12 runtime. It uses one GPU, four CPUs,
192 GiB RAM and a 24-hour allocation. Full inference contexts are never chunked
or truncated. The normalization is the same metadata-labelled retrospective Harmony
replay used by the other agentic role reports, not exact serving-prompt recovery.
Tool schemas are recovered from the pinned AgentDojo suites as unscored context.
Tool errors use the driver-visible error text. Ground-truth roles are metadata;
injected tool text is never relabelled from its content.

`role_inputs.json` records all expected attempt IDs, saved/missing histories,
partial task-error histories, source hashes and normalized input hashes. Clean
inputs have 95 histories per model; attacked inputs have 921 base and 918 trained
histories, with three trained histories unavailable. Reasoning text is absent
from all saved AgentDojo messages and is not reconstructed. Zero-support role
recall is null and rendered as a dash. The five-class probe can still predict
reasoning; those predictions appear in confusion-matrix columns.

Normalized histories: `data/role_perception/{base,trained}/{clean,attacked}/`.
Fresh probes and held-out fit metrics: `role_perception/<model>/post_training/`.
Scoring summaries: `role_perception/<model>/{clean,attacked}.json`.
Per-history confusion counts and token predictions:
`predictions/role_perception/<model>/<subset>/`. Only small summary/probe artifacts
are published; histories, features and predictions remain on cluster storage.
The input hash and probe hash guard incremental scoring against stale reuse.

The afterany CPU report job saves whatever completed summaries exist, includes
pending states for failures, and automatically pushes the report, appendix,
probe weights and code to `main`. It retries a racing GitHub update against a fresh
parent without touching the shared working branch or index. See `role_jobs.json`.
