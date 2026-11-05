# Experiment reports

Create `experiment_reports/<topic>/<experiment>/` by copying `_template/`.
Topics mirror the folders of `src/experiments/`:

- `gptoss_role_probes/`: probe fitting on the base model and the training-data comparison.
- `eval_role_probes/agentic_eval/<benchmark>/`: base-model trajectory collections and probe
  scores per benchmark (`appworld`, `tau2_retail`, `swe_bench`), plus cross-benchmark
  scoring under `probe_scoring/`.
- `eval_role_probes/chat_eval/<dataset>/`: probe evaluations on chat data (`norobots`).
- `training_against_role_probes/`, `prompt_injection/`: training the model against the
  probes and its downstream evaluations.

Each experiment's README is the primary record; keep related data and plots in
that folder, with subfolders such as `data/` and `plots/` as needed. Link large
external artifacts with their provenance instead of committing them.

Use the four report sections in the template, keep observations separate from
interpretation, and finish with a link to the short `RUNNING.md` instructions.
