# hx2 cluster workflow

Xiaoxue's cluster. [CLUSTER.md](CLUSTER.md) and the original `*.sbatch` files
describe Evgenii's DE.BER1 cluster and do not apply here. Facts below were
verified live on 2026-10-30 (`sinfo`, `sacctmgr`, a one-GPU probe job) unless marked.

## Access and resources

- Slurm account `hx2-users`, QoS `normal` (per-user cap: 12 GPUs, 3-day wall time).
  The user's budget for this work is up to 8 H200s at once.
- `gpu`: 20 nodes x 4 H200 (143 GB), 64 CPUs, 1.5 TB RAM, 1.5 TB node-local `/scratch`.
  Partition QoS `gpu`: at least one GPU per job, 3-day wall time. Expect queues of
  100+ pending jobs; submit early and chain with `--dependency`.
- `interactive-gpu`: 3 nodes, QoS `interactive`: **one running job per user, at most
  1 GPU, 16 CPUs, 64 GB RAM, 8 hours**. Good for smoke tests; too small for the
  probe fits, which hold activations in host RAM. The partition applies that QoS
  itself: submit with `--partition=interactive-gpu` and no `--qos` (the account only
  allows `normal`; `--qos=interactive` is rejected). Verified 2026-11-09 with the
  probe smoke job, which ran within a minute of submission.
- Driver 580.65.06, CUDA 13.0 (RHEL 9.6): the repo's `cu130` torch and the `cuml-cu13`
  probe-fitting environment run unchanged. `/tmp` on GPU nodes is only 1 GB: put `TMPDIR`, Triton and
  matplotlib caches on `/scratch/$USER/...` (see `hx2/common.sh`). Apptainer is
  installed on login and compute nodes. Compute nodes have outbound internet.
- Storage: weka home `/hx2-weka/home/xy2222` (9 TB filesystem, shared). No per-user
  quota was visible; the user considers ~600 GB of SWE images too much.

## Environment

- `uv` is at `~/.local/bin/uv`; system Python is 3.9, uv installs 3.12 itself.
  Install with `uv sync --locked --extra vllm` in the repo root and
  `uv sync --locked --project src/experiments/gptoss_role_probes/fit_env` for cuML probe
  fitting; jobs use `uv run --frozen --no-sync`. Layout and reasons:
  [ENVIRONMENTS.md](ENVIRONMENTS.md). Both environments are installed on hx2.
- Shared dependency versions come from `/hx2-weka/home/xy2222/prompt_injection_interp/pyproject.toml`
  (rule in [AGENTS.md](../AGENTS.md)). On 2026-11-09 this moved Transformers from 4.57.6
  to 5.9.0, vLLM from 0.32.0 to 0.27.1 and peft from 0.20.0 to 0.18.1. Probe weights and
  scores recorded before that date were computed under Transformers 4.57.6, and the hx2
  AppWorld cohort recorded earlier was served by vLLM 0.32.0.
- The older sub-project folders (`agentic_eval/server_current`, `agentic_eval/swe_bench/env`,
  `training_against_role_probes/runtime`) held only their `.venv` and cache after the merge
  and were deleted on 2026-11-09 (about 19 GB).
- Training code starts its helper processes (Inspect collectors, the colocated vLLM
  server) with the interpreter it is running under, so it needs no `uv` path. Only the
  DE.BER1 job files call `.tools/bin/uv`, which does not exist on hx2; hx2 job files live
  in `hx2/` subfolders and use `uv` from `PATH`.
- Hugging Face cache: `HF_HOME=/hx2-weka/home/xy2222/.cache/huggingface`.
  `openai/gpt-oss-20b` at the pinned revision is already there;
  `.cache/models/gpt-oss-20b` is a symlink to that snapshot.
- Job scripts for hx2 live next to the DE.BER1 ones in `hx2/` subfolders, e.g.
  `src/experiments/gptoss_role_probes/hx2/{common.sh,smoke.sbatch,train.sbatch}`.
  Slurm output goes under the experiment's ignored `logs/` directory, never `/tmp`.

## Current experiments on hx2

- [Probe training-data comparison](../experiment_reports/gptoss_role_probes/training_data/README.md):
  C4/Dolma vs Toucan vs combined probes. Prepared inputs regenerate byte-identically
  (documents hash `5a76ba32...` for the first 250 lines) from the pinned sources.
- Agentic trajectories must be regenerated here; the coauthor's data cannot be copied.
  Priority: AppWorld normal test first, SWE-bench only if tractable, no tau2 retail
  (no Azure simulator credentials).
