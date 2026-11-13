# Python environments

The repository has two uv environments. Before 2026-11-09 it had ten: the root plus nine
sub-projects under `src/experiments/`, each with its own lock. A third environment for cuML
probe fitting existed until 2026-11-13, when every probe fit moved to the project's torch
L-BFGS fitter (`src/role_probe_training/training/probe_lbfgs.py`), which runs in the main
environment.

| environment | defined by | install | used for |
|---|---|---|---|
| main | root `pyproject.toml` and `uv.lock` | `uv sync --locked --extra vllm` | vLLM serving, the Harmony proxy, Inspect collections (AppWorld, retail, SWE-bench), agentdojo, probe fitting and scoring, training |
| AppWorld data | `src/experiments/eval_role_probes/agentic_eval/appworld/env/` | `uv sync --locked --project src/experiments/eval_role_probes/agentic_eval/appworld/env` | the AppWorld package itself (data download, world servers) |

Jobs run `uv run --frozen --no-sync ...` from the repository root for the main environment
and add `--project <folder>` for the AppWorld data environment. A node that never serves the model can
install the main environment without `--extra vllm`; the versions of everything else are
the same either way.

## Why two and not one

- **AppWorld needs pydantic 1**, and everything else needs pydantic 2.
- Everything else resolves together: vLLM, Inspect with inspect-evals and swebench,
  agentdojo, peft, wandb and the probe code.

## Rules built into the root project

- **vLLM is an extra**, as in Xiaoxue's `prompt_injection_interp` project. It pins
  torch 2.13.0 for the whole lock, with or without the extra installed.
- **Only the forked Harmony library is installed.** vLLM requires the upstream
  `openai-harmony`, which installs the same `openai_harmony` module as the pinned fork
  `oss-harmony==0.0.12`. `exclude-dependencies` removes the upstream copy, so vLLM imports
  the fork. Prompts are rendered by this repository's renderer either way.
- **The Inspect harness is held at the recorded versions.** `inspect-ai==0.3.283`,
  `inspect-evals==0.25.0`, `swebench==4.1.0` and `openai==3.36.0` are pinned. Of the 47
  packages that would otherwise have moved from the versions in the old client and
  agentdojo locks, 38 are kept through `uv.lock`. Nine had to move: datasets, fastapi,
  fsspec, s3fs, huggingface-hub, numpy, protobuf, tenacity and tokenizers.
- **`agent-client-protocol<1` is a permanent constraint.** inspect-ai 0.3.283 imports
  `acp.helpers`, which release 1.0 removed, but only requires `>=0.12.0`. Without the
  constraint a fresh resolution installs 1.0 and `inspect_ai.eval` fails at import.
- **Shared packages take the version constraint of `prompt_injection_interp`**
  (rule in [AGENTS.md](../AGENTS.md)): transformers 5.9.0, vLLM 0.27.1, peft 0.18.1,
  agentdojo 0.1.35.

## What changed for existing work

- Removed projects: `agentic_eval/` (retail client), `agentic_eval/swe_bench/env`,
  `agentic_eval/server` (vLLM 0.11.0), `agentic_eval/server_current`,
  `training_against_role_probes/runtime`, `training_against_role_probes/collection_runtime`
  and `agentdojo_robustness/driver`. Their lock files are in Git history at commit
  358845a. Reports written before 2026-11-09 name these environments.
- Every job script and manifest was rewritten to the new layout: no `--project` for merged
  environments, `CUDA_HOME` under the root `.venv`, lock hashes from the root `uv.lock`.
- Version changes for training: torch 2.9.1+cu128 to 2.13.0+cu130 and peft 0.20.0 to
  0.18.1, after transformers 4.57.6 to 5.9.0 earlier the same day. Version change for cuML probe
  fitting: torch 2.14.1 to 2.13.0.
- Manually installing the root package into client environments
  (`uv pip install --no-deps -e .`) is no longer needed.
- On a checkout that still has the old environments, run the two install commands above.
  The `.venv` folders left inside the removed project folders are unused and can be deleted.

## Verification on hx2 (2026-11-09)

- **CPU, main environment:** the Harmony rendering and chat-server unit tests, the export
  checks (which run `inspect_ai.eval`), the probe scoring and probe label checks, the
  training checks (`training_against_role_probes/checks.py`, `kl_iteration/checks.py`,
  `same_batch_epochs/checks.py`, `on_policy/checks.py`), the agentdojo driver check, and
  the peft features the trainer uses (LoRA on modules and on expert parameters). Every
  job script parses, every Python file compiles, and the benchmark runners import.
- **GPU, main environment:** serving smoke job 1648398 (`shared/hx2/serving_smoke.sbatch`):
  vLLM 0.27.1 running on the forked Harmony library, completions by token ID, a tool round
  trip through the Harmony proxy and an Inspect request through the proxy.
- **GPU, former probe-fitting environment:** probe smoke job 1648396
  (`gptoss_role_probes/hx2/smoke.sbatch`): model load, activation site and a cuML fit. The
  smoke test now fits with L-BFGS in the main environment; that version has not been run.
- **Not yet run in the new layout:** a training job, the on-policy colocated serving with
  its custom vLLM worker class, a full benchmark episode (AppWorld pilot, SWE-bench
  sandbox, retail), and anything on DE.BER1.
