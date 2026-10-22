# Reproduction

Use the repository's locked `agentic_eval/swe` Inspect client and `agentic_eval/server_current` vLLM environment. AppWorld runs in its official container, separately from Inspect because their Pydantic requirements conflict. The container currently reports AppWorld 0.1.3; each run records its SHA-256. The host download environment is locked to 0.1.3.post1.

```bash
.tools/bin/uv sync --locked --project src/experiments/agentic_eval/appworld
.tools/bin/uv run --project src/experiments/agentic_eval/appworld --locked appworld install
.tools/bin/uv run --project src/experiments/agentic_eval/appworld --locked appworld download data --root "$PWD/.cache/appworld"
apptainer pull .cache/appworld/appworld.sif docker://ghcr.io/stonybrooknlp/appworld:latest
mkdir -p experiment_reports/agentic/appworld/logs
sbatch --exclude=vca-gpu-0701-03 src/experiments/agentic_eval/appworld_job.sbatch --pilot
# After reviewing the training pilot, compare two replicas and collect all splits:
APPWORLD_PILOT_ROOT="$PWD/experiment_reports/agentic/appworld/runs/<pilot-job>" \
  sbatch --exclude=vca-gpu-0701-03 src/experiments/agentic_eval/appworld_parallel_job.sbatch
```

The pilot uses the same four seeded training tasks with one and eight permitted concurrent worlds (effective concurrency four on this subset). The two-GPU job first repeats the same four tasks across two independent model replicas, then collects train, test_normal and test_challenge sequentially. Each replica handles a disjoint shard with up to eight concurrent worlds. Both replicas use one B200 each on the same node; this is data parallel inference, not tensor parallelism. Each world has its own process and output directory, as required by [AppWorld's parallelization guide](https://github.com/StonyBrookNLP/appworld/blob/main/guides/parallelizing_worlds.md). Reference solutions were checked on training IDs `82e2fac_1` and `82e2fac_2` in parallel, both passing. Reference solutions are never exposed to the evaluated agent.

Each worker runs in Apptainer with no network, no host home/project mounts, read-only benchmark data and its own writable output directory. AppWorld's syntax and execution guards remain enabled. Only the parent Inspect process can access the inference server. The prompt follows the API discovery and completion protocol in the [official minimal agent](https://github.com/StonyBrookNLP/appworld/blob/v0.1.3/notebooks/minimal_agent.ipynb), using a structured Python tool instead of presenting execution output as user messages.

Server: vLLM 0.32.0 with Triton attention and MoE kernels; compiled execution with CUDA graph batch sizes 1, 2, 4, 8 and 16. FlashInfer sampling and unrelated kernel warmup are disabled. The CUDA compiler packages are pinned in its lockfile.

Settings: GPT-OSS-20B, high reasoning, temperature 1, seed 123, at most 50 Python interactions, 8,192 generated tokens per call, 32,768-token context, 2,000,000 cumulative tokens and 1,800 seconds per task. No automatic selection of successful retries. Implementation: [adapter](../../../../src/role_probe_training/trajectories/appworld.py), [runner](../../../../src/experiments/agentic_eval/appworld_run.py), [worker](../../../../src/experiments/agentic_eval/appworld_worker.py). Paths in these links are relative to this report folder.

## Outputs and timings

`runs/<job>/<split>/` contains the selection/configuration manifest, original Inspect logs, `trajectories.jsonl`, `generations.jsonl`, task summary and `timing.json`. The trajectory schema matches the retail and SWE exports: message roles, assistant reasoning, structured tool calls and named tool replies. Timing records separate environment loading, tool execution and total rollout time. Parent-process monotonic clocks avoid AppWorld's simulated time. Data, task descriptions and generated conversations stay on cluster storage and are excluded from Git, following AppWorld's request not to publish decrypted benchmark content.

The official benchmark data source is `https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle`; local data is `.cache/appworld/data`. Preserve manifests and the exact cached container for reproduction; a future `latest` image may differ.

Run `appworld_check.py` in the Inspect client environment to verify two parallel reference solutions and scoring when Inspect interrupts a trajectory at a message limit.

The combined report is generated with `appworld_report.py --dataset NAME=SHARD0_DIRECTORY,SHARD1_DIRECTORY --output experiment_reports/agentic/appworld`. It verifies unique, complete coverage against the shared selection manifest, writes `data/NAME.jsonl`, and updates metrics and the appendix without overwriting the main report. Combined throughput uses the interval from the first shard start to the last shard finish; both clocks are on the same node.

Current collection: optimized one-GPU pilot `runs/1489112`; two-GPU pilot and full splits `runs/1489157`. Starting both inference replicas took 95 seconds with local checkpoint/compilation caches; split collection timings exclude that startup. The native checkpoint uses MXFP4 expert weights and BF16 activation computation. Consolidated training trajectories are `data/train.jsonl`; raw logs and generation contexts remain under the run directories.
