# Role probe training

Experiments on training models against role probes.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/guides/projects/):

```bash
# This workspace has a local uv installation; use this if uv is not on PATH.
export PATH="$PWD/.tools/bin:$PATH"
git submodule update --init --recursive
uv sync --locked
uv run python -c "import role_probe_training; print(role_probe_training.__file__)"
```

For a fresh clone, install uv first using its
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
The `.venv` and uv cache are local, ignored directories. Add dependencies with
`uv add`; commit the resulting project metadata and lockfile together.
Training dependencies are locked for the [GPT-OSS role-probe reproduction](experiment_reports/gptoss_role_probes/gptoss20b_baseline/README.md).

## Layout

- `src/role_probe_training/`: reusable Python modules, grouped as needs emerge.
- `src/experiments/`: experiment entry points and Slurm scripts. Probe work is grouped as
  `gptoss_role_probes/` (fitting and training data) and `eval_role_probes/` with
  `agentic_eval/<benchmark>/` and `chat_eval/<dataset>/` subfolders.
- `submodules/`: pinned upstream repositories.
- `experiment_reports/<topic>/<experiment>/`: reports, data and plots, in folders that
  mirror `src/experiments/`; each experiment starts with a README and ends with a link
  to `RUNNING.md`.
- `agent_docs/`: persistent operational documentation.

See [working conventions](AGENTS.md), the [cluster guide](agent_docs/CLUSTER.md)
and the [experiment report template](experiment_reports/_template/README.md).

## Training initial probes
- [C4/Dolma vs Toucan probes](experiment_reports/gptoss_role_probes/training_data/README.md): the same recipe fitted on neutral text, Toucan tool-use text, or both; scored on both held-out sets, clean chat and the [hx2 AppWorld cohort](experiment_reports/eval_role_probes/agentic_eval/appworld/base_trajectories_hx2/README.md).


## Evaluations of trained probes

- [Layer-12 transfer results](experiment_reports/eval_role_probes/agentic_eval/probe_scoring/probe_transfer/README.md): full τ² retail 114 and SWE-bench Verified 500; per-token predictions and an offline viewer.
- [Layers 4, 8, 12, 16 and 20](experiment_reports/eval_role_probes/agentic_eval/probe_scoring/probe_layer_comparison/README.md): the same trajectories, scored together in each forward pass.
- Task performance: [retail](experiment_reports/eval_role_probes/agentic_eval/tau2_retail/base_trajectories/README.md) and [SWE](experiment_reports/eval_role_probes/agentic_eval/swe_bench/base_trajectories/README.md). The SWE run has substantial model API/harness failures and does not reproduce the published score.
- [Operational notes](agent_docs/AGENTIC_EVALS.md): Slurm execution, trace schema, sandboxing, provenance and reproducibility.
- [HX2 cluster notes](agent_docs/HX2.md): information about the hx2 cluster.
- [No Robots chat evaluation](experiment_reports/eval_role_probes/chat_eval/norobots/README.md): the three probe sets on 500 human multi-turn conversations as given, and on first turns answered by GPT-OSS-20B with its own medium reasoning.

## Training and prompt injection robustness

- [Training comparison](experiment_reports/training_against_role_probes/README.md):
  fixed-data and on-policy role training, task utility and whole-history role recovery.
- [Fixed-data epoch continuation](experiment_reports/training_against_role_probes/same_batch_epochs/README.md):
  continuation on the original 2,700 histories, with saved baseline measurements and pending later epochs.
- [AgentDojo robustness](experiment_reports/prompt_injection/agentdojo_same_batch_27000/README.md):
  matched base/trained clean and attacked evaluations, coverage and reproduction.
- [Shared adapter](https://huggingface.co/kortukov/gpt-oss-20b-role-perception-appworld-27000):
  complete PEFT weights and a vLLM launcher preserving evaluation precision.
