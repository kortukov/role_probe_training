# Experiment reports

The project has two stages. Each has one status page; start there.

1. **Probes.** Find the probe setup that recognises roles best on the *base* model, judged
   on held-out passages and on base-model runs of AppWorld, τ² retail and SWE-bench.
   Status page: [base-model probe evaluation](eval_role_probes/README.md).
   How the probes are fitted: [probe fitting](gptoss_role_probes/README.md).
2. **Training.** Train the model against the chosen probes; check role accuracy and task
   success on the same three benchmarks.
   Status page: [training against role probes](training_against_role_probes/README.md).

**Current setup (since 2026-11-13):** one Harmony renderer for every prompt, probes fitted
to convergence with L-BFGS, sampling temperature 1. Results measured before that used the
old chat templates and cuML probes. They are marked **old template** on every status page
and are not directly comparable with current results.

| Folder | Question it answers |
|---|---|
| [gptoss_role_probes/](gptoss_role_probes/README.md) | How are the probes fitted, and how good are they on held-out text? |
| [eval_role_probes/](eval_role_probes/README.md) | How well do the probes recognise roles in real base-model agent runs? |
| [training_against_role_probes/](training_against_role_probes/README.md) | Does training against the probes work, and what does it cost in task success? |
| [prompt_injection/](prompt_injection/agentdojo_same_batch_27000/README.md) | Is the trained model more robust to prompt injection? (old template, one report) |

## Rules for this folder

- A folder that only contains other folders has a short `README.md`: one sentence on what
  the folder answers, then one line per child with its conclusion and status.
- Status says whether a result is **current**, or from the **old template** or an
  **earlier fitter**; a superseded result names what replaced it.
- A new experiment adds one row to its stage's status table. Copy `_template/` to
  `experiment_reports/<topic>/<experiment>/`; the experiment's README is its main record,
  with the four sections of the template and a link to `RUNNING.md`.
- Topics mirror the folders of `src/experiments/`. Large data stays on cluster storage
  under the experiment folder, with paths and provenance in `RUNNING.md`.
