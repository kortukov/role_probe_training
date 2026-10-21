# Running this experiment

Replace placeholders and adjust relative links for the experiment's nesting.
Keep this document brief.

- Code: <link to entry point under src/experiments/ and relevant modules>.
- Data: <links to inputs/results, external paths and provenance if needed>.
- Revisions: <main repo commit, submodule commits, model and dataset revisions>.
- Settings: <config link, seeds, hardware and essential dependencies>.

From the repository root:

```bash
uv sync --locked
uv run python src/experiments/<experiment>/run.py <arguments>
```

For Slurm, link the reviewed job script and give its exact submission command.
Specify the output directory under `experiment_reports/<topic>/<experiment>/`
and the command that regenerates the figures from saved results.
