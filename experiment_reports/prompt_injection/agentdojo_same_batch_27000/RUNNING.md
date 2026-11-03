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
# Production submission uses afterok on the successful smoke-test job:
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
The model card there documents the required loader, and the upload includes the
original complete training adapter as well as the exact evaluation export.

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
