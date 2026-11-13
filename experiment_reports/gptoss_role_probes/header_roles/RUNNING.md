# Running this experiment

> **The code of this experiment was removed after commit `3fe11e5`.** The project now fits
> every probe with one L-BFGS fitter (`src/role_probe_training/training/probe_lbfgs.py`), and
> the cuML and AdamW fitters this experiment compares no longer exist in the working tree.
> Check out `3fe11e5` to rerun it; every path below refers to that commit. Header-role probes
> from the current fitter: [L-BFGS fitter](../lbfgs_fitter/README.md).

- Code: `fit.py` (activation
  cache, both fitters, convergence tests, fitter agreement),
  `report.py` (tables,
  appendix, plots), hx2 job scripts
  `header_roles.sbatch` and
  `header_roles_parallel.sbatch`.
  Role definitions and wraps: `src/role_probe_training/harmony/render.py`
  ([HARMONY_RENDERING.md](../../../HARMONY_RENDERING.md) section 6). Fitters:
  `src/role_probe_training/probes/linear.py` and `src/role_probe_training/training/probe_fit.py`.
- Settings: `config.json`
  holds the layers, C, seed, the cuML tolerance and limits and the AdamW schedule lengths.
- Data: the 250 C4/Dolma passages are the first 250 lines of
  `experiment_reports/gptoss_role_probes/training_data/inputs/documents.jsonl` (ignored;
  regenerate with `prepare.py` as in the [training-data experiment](../training_data/RUNNING.md)).
  Their hash `5a76ba32…` equals the baseline's input hash and is checked before extraction.
- Revisions: GPT-OSS-20B `6cee5e81ee83917806bbde320786a8fb61efebee`; oss-harmony 0.0.12;
  cuML 26.10.0, torch 2.13.0, transformers 5.9.0 (probe-fitting environment). The fits ran
  on a working tree with uncommitted changes on top of commit `8fa575f`; every probe's JSON
  and `metrics/run_metadata_fit_*.json` record the SHA-256 of the source files used.
- Hardware: one H200 on the hx2 `interactive-gpu` partition (64 GB host memory is enough
  because activations are memory-mapped and one layer is loaded at a time).

From the repository root on hx2:

```bash
uv sync --locked --project src/experiments/gptoss_role_probes/fit_env
mkdir -p experiment_reports/gptoss_role_probes/header_roles/logs
job=src/experiments/gptoss_role_probes/hx2/header_roles.sbatch
# Extract and cache activations (about 4 minutes, 40 GB), then the cuML fits (about 1 minute per layer).
sbatch $job --backend cuml
# cuML at tighter tolerances; results are kept beside the fits at the configured tolerance of 1e-4.
sbatch $job --backend cuml --tol 1e-5
sbatch $job --backend cuml --tol 1e-6
# AdamW at the upstream learning rate, schedules of 5,000 to 1,280,000 steps (about 28 minutes per layer).
sbatch $job --backend adamw --max-steps 1280000
# AdamW at learning rate 0.1; results are kept beside the upstream-rate fits.
sbatch $job --backend adamw --learning-rate 0.1 --max-steps 1280000
# On a login node, after the jobs: agreement of the AdamW probes with cuML, then tables, appendix and plots.
uv run --frozen --no-sync python src/experiments/gptoss_role_probes/header_roles/fit.py compare
uv run --frozen --no-sync python src/experiments/gptoss_role_probes/header_roles/report.py
```

The partition runs one job per user, so chain submissions with `--dependency=afterany:JOB`.
A rerun skips finished fits and continues an AdamW fit
that has not converged with the longer schedules of `config.json` (`--max-steps` caps them;
`--layers 12 16` selects layers). `header_roles_parallel.sbatch "ARGS" "ARGS" ...` runs several
fit stages at once in one job; four AdamW fits at once took about as long as one after another.

Outputs under `experiment_reports/gptoss_role_probes/header_roles/`:

- `probes_header/layer_L.npz|json` (cuML, tolerance 1e-4) and `probes_header_adamw/` (AdamW, upstream rate):
  the same names `gptoss_role_probes/run.py --roles header` uses, so its clean-chat evaluation
  can read them. `probes_header_adamw_lr0.1/` and `probes_header_adamw_lr0.01/` (layer 12 only)
  hold the other learning rates, `probes_header_cuml_tol1e-05/` and `probes_header_cuml_tol1e-06/`
  the other cuML tolerances. Weights apply to raw activations.
- `metrics/fits/layer_L_FITTER.json`: fit attempts, loss history, train and validation
  metrics (accuracy, per-role accuracy, confusion counts, loss).
- `metrics/fitter_agreement.json`: written by `fit.py compare` from the stored probes; every
  probe of a layer is compared with its cuML probe of the tightest tolerance. A
  stored AdamW probe is the one of the longest schedule run, so rerunning a rate with a longer
  schedule replaces it; the losses of every schedule stay in `metrics/fits/`.
- `metrics/tables.md`, `metrics/summary.json`, `APPENDIX.md`, `plots/`: written by `report.py`.
- `artifacts/activations/` (ignored): float16 activations of all content tokens, about 8 GB
  per layer. `logs/` (ignored): Slurm output and the cuML solver logs, one objective value
  per L-BFGS iteration.

Numerical details. Activations are computed by the MXFP4 checkpoint dequantized to BF16
with eager attention and stored as float16, as in the earlier fits. cuML receives them in
float64. The AdamW fitter runs under bfloat16 autocast with batches of 32 messages, as
upstream; its loss history is evaluated in float32 over all tokens at 200 step counts
spaced evenly on a log scale (the upstream-rate schedules up to 320,000 steps at layer 12
used evenly spaced step counts). The passage split is the one recorded in `documents.jsonl`
(225 fitting and 25 held-out passages); every fit uses seed 123.

Slurm jobs of the reported results: 1649624 (extraction), 1650390 (cuML, tolerance 1e-4),
1650401 (cuML, tolerances 1e-5 and 1e-6, with `header_roles_parallel.sbatch`),
1649669 (AdamW, upstream rate), 1649814 and 1649815
(layer 12 at rates 0.01 and 0.1 up to 320,000 steps), 1649825 (AdamW, rate 0.1, the other
layers and the 1,280,000-step schedule of layer 12; run with `header_roles_parallel.sbatch`).
Each probe's JSON names its job.
