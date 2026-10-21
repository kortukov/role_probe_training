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
Training dependencies are locked for the [GPT-OSS role-probe reproduction](experiment_reports/role_probes/gptoss20b_baseline/README.md).

## Layout

- `src/role_probe_training/`: reusable Python modules, grouped as needs emerge.
- `src/experiments/`: experiment entry points and Slurm scripts.
- `submodules/`: pinned upstream repositories.
- `experiment_reports/<topic>/<experiment>/`: reports, data and plots;
  each experiment starts with a README and ends with a link to `RUNNING.md`.
- `agent_docs/`: persistent operational documentation.

See [working conventions](AGENTS.md), the [cluster guide](agent_docs/CLUSTER.md)
and the [experiment report template](experiment_reports/_template/README.md).
