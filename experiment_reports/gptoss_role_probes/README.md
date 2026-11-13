# Probe fitting

How the role probes are fitted on the base model and how accurate they are on held-out
passages. Whether a fitted probe is good enough is decided on real agent runs, in the
[base-model probe evaluation](../eval_role_probes/README.md).

**Current probes:** `lbfgs_fitter/probes/` (five roles) and `lbfgs_fitter/probes_header/`
(ten header roles), layers 12 and 16, fitted to convergence with the project's one fitter
(L-BFGS) on text rendered by the Harmony renderer. Held-out accuracy, five roles: 69.1%
at layer 12 and 85.3% at layer 16.

| Folder | Contents | Status |
|---|---|---|
| `lbfgs_fitter/` | converged L-BFGS probes and their comparison with cuML on the same data (accuracy within 0.1 points); report still to be written | current |
| `gptoss20b_baseline/` | first cuML probes at layers 12 and 16 with clean-chat evaluation; the probes behind every number published before 2026-11-13 | earlier fitter (cuML) |
| `gptoss20b_layers/` | cuML probes at layers 4, 8 and 20 | earlier fitter (cuML) |
| `header_roles/` | ten header roles, cuML against AdamW; the AdamW fits did not converge | earlier fitters (cuML, AdamW) |
| `training_data/` | neutral passages against Toucan tool-use text as fitting data | earlier fitter (cuML) |
