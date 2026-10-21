# Experiment entry points

Keep experiment scripts and Slurm submission scripts here, grouped by experiment
when useful. Run Python scripts from the repository root with
`uv run python src/experiments/<experiment>/run.py`.

Put reusable logic in `src/role_probe_training/` and import it from the installed
package; avoid modifying `sys.path`. Add related module folders only when needed.
Write outputs to the corresponding `experiment_reports/<topic>/<experiment>/`
folder (or node-local scratch during a cluster job, then copy back).
