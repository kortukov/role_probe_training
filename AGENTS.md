# Working in this repository

## Environment and code

- Use uv: `uv sync --locked`, `uv run ...`, and `uv add <dependency>`.
  Keep `pyproject.toml`, `.python-version` and `uv.lock` consistent. Python 3.12
  is the initial default; revisit compatibility when adding training libraries.
- All first-party code lives in `src/`. Reusable code belongs in
  `src/role_probe_training/`; experiment entry points and job scripts belong in
  `src/experiments/`. Upstream Git submodules are the explicit exception.
- Probe experiments are grouped as `src/experiments/gptoss_role_probes/` (fitting and
  training data) and `src/experiments/eval_role_probes/` (`agentic_eval/<benchmark>/`,
  `chat_eval/<dataset>/`); `experiment_reports/` mirrors this hierarchy.
- Move reusable logic into modules and group related modules in their own folders
  as needs emerge. Prefer simple, readable Python and the Zen of Python. Avoid
  speculative abstractions, unnecessary frameworks and premature structure.
- Keep experiment scripts thin. Import the installed package rather than adding
  paths to `sys.path`. Add dependencies only when an actual task needs them.
- Verify changes with checks proportional to their impact. Do not launch training
  merely to validate documentation or scaffolding.

## Upstream code

- `submodules/prompt-injection-as-role-confusion` and
  `submodules/training_against_probes` are pinned Git submodules.
- Initialize with `git submodule update --init --recursive`. Do not automatically
  update upstream revisions or install upstream dependencies into our environment.
- Keep our adaptations in `src/` unless a task requires changes upstream. Record
  relevant submodule revisions in experiment reproduction instructions.

## Experiments and reports

- Store reports, related data and plots under nested
  `experiment_reports/<topic>/<experiment>/` directories. Each experiment's
  `README.md` is the main entry point. Start from `experiment_reports/_template/`.
- Follow [report-writing rules](agent_docs/REPORT_WRITING.md): keep the main report
  brief, conceptual and readable; put supporting analyses in APPENDIX and
  execution details in RUNNING/configuration.
- Completed reports must contain, in order:
  1. Motivation & RQ.
  2. Evaluation setup: conceptual methods, datasets/splits, models, metrics and
     baselines. Define every quantity used in the main results before presenting it.
  3. Factual observations, usually supported by plots/tables. Separate what was
     measured from interpretation; do not invent or overstate findings.
  4. Message: interpretation and conclusion, explicitly answering the RQ.
- End the README with a link to a separate `RUNNING.md` giving brief reproduction
  commands and links to the relevant code and data. Record revisions, settings,
  output paths and essential environment details there.
- Remove superseded experiments, obsolete attempts and unrelated historical material
  from the current repository. Retain only artifacts needed by current conclusions
  and reproduction; consolidate them under the current experiment. Check for stale
  narrative, folders and links before every report push. Git history is the archive.
- Keep small results and plots in Git. For large/private datasets and checkpoints,
  use appropriate cluster storage and put paths, provenance and retrieval details
  in the experiment folder. Add targeted ignores as needed; do not silently lose
  the information needed to reproduce an experiment.

- Put detailed confusion matrices in a separate `APPENDIX.md` and link the relevant
  section from the main report. Keep the main README focused on results and interpretation.

## Persistent working knowledge

- Keep this file concise and store detailed operational knowledge in
  [agent_docs](agent_docs/README.md).
- Read [agent_docs/CLUSTER.md](agent_docs/CLUSTER.md) for Evgenii's jobs/[agent_docs/HX2.md](agent_docs/HX2.md) for Xiaoxue's jobs before preparing cluster jobs.
  Use uv directly in Slurm jobs by default; containers are not part of this
  project's initial setup. Follow cluster storage and resource rules.
- Update documentation when a workflow changes or a useful operational fact is
  verified. Distinguish source documentation, live verification and assumptions.

## Role-probe implementation

- Current reproduction: [GPT-OSS-20B baseline](experiment_reports/gptoss_role_probes/gptoss20b_baseline/README.md).
  Reusable code lives in `src/role_probe_training/probes/`.
- Upstream activations are **post-attention-layernorm outputs / pre-MLP inputs**,
  not decoder-block outputs. Probe metadata must record this site, the zero-based
  layer, model revision and role order. Do not apply weights to a different site.
- Preserve the upstream training scale (configured 250 passages, up to 1,024
  tokens, all content tokens). Do not silently subsample tokens or reduce training
  data to save time. Small conversational evaluation is separate from training.
- Ground-truth roles come from message/channel metadata. Mean role probabilities
  and argmax role accuracy are different metrics. Tool-role transfer requires
  validation on actual tool-bearing trajectories; clean chat alone is insufficient.
- Current probe weights: layers 12/16 in the baseline experiment and layers 4/8/20
  in [additional-layer fitting](experiment_reports/gptoss_role_probes/gptoss20b_layers/README.md).
  Fits have convergence limitations. Use the current [agentic evaluations](experiment_reports/eval_role_probes/agentic_eval/probe_scoring/probe_transfer/README.md)
  to assess transfer to actual reasoning and tool messages. Probes fitted on Toucan tool-use
  text, alone or with the neutral passages, live in the
  [training-data comparison](experiment_reports/gptoss_role_probes/training_data/README.md).

- Prompts are rendered only by `src/role_probe_training/harmony/` with the pinned
  oss-harmony library (see [HARMONY_RENDERING.md](HARMONY_RENDERING.md)). Turns the model
  generated are replayed from their generated token IDs, never re-rendered from text, and
  probes score the token records (`history`) written by the Harmony chat server. Do not
  use the Hugging Face chat template or hand-written Harmony headers anywhere.
- Probe evaluation must replay each scored conversation whole, without chunking
  or truncation. Histories above the model's native context limit are explicitly
  unscorable for role metrics; retain every attempt in task-success denominators
  and report coverage. Reuse saved trajectories when changing probe evaluation.
