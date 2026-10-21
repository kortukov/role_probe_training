# Working in this repository

## Environment and code

- Use uv: `uv sync --locked`, `uv run ...`, and `uv add <dependency>`.
  Keep `pyproject.toml`, `.python-version` and `uv.lock` consistent. Python 3.12
  is the initial default; revisit compatibility when adding training libraries.
- All first-party code lives in `src/`. Reusable code belongs in
  `src/role_probe_training/`; experiment entry points and job scripts belong in
  `src/experiments/`. Upstream Git submodules are the explicit exception.
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
- Completed reports must contain, in order:
  1. Motivation & RQ.
  2. Evaluation setup: metrics, baselines, methods, datasets/splits, models,
     seeds and relevant configuration.
  3. Factual observations, usually supported by plots/tables. Separate what was
     measured from interpretation; do not invent or overstate findings.
  4. Message: interpretation and conclusion, explicitly answering the RQ.
- End the README with a link to a separate `RUNNING.md` giving brief reproduction
  commands and links to the relevant code and data. Record revisions, settings,
  output paths and essential environment details there.
- Keep small results and plots in Git. For large/private datasets and checkpoints,
  use appropriate cluster storage and put paths, provenance and retrieval details
  in the experiment folder. Add targeted ignores as needed; do not silently lose
  the information needed to reproduce an experiment.

## Persistent working knowledge

- Keep this file concise and store detailed operational knowledge in
  [agent_docs](agent_docs/README.md).
- Read [agent_docs/CLUSTER.md](agent_docs/CLUSTER.md) before preparing cluster jobs.
  Use uv directly in Slurm jobs by default; containers are not part of this
  project's initial setup. Follow cluster storage and resource rules.
- Update documentation when a workflow changes or a useful operational fact is
  verified. Distinguish source documentation, live verification and assumptions.

## Role-probe implementation

- Current reproduction: [GPT-OSS-20B baseline](experiment_reports/role_probes/gptoss20b_baseline/README.md).
  Reusable code lives in `src/role_probe_training/probes/`.
- Upstream activations are **post-attention-layernorm outputs / pre-MLP inputs**,
  not decoder-block outputs. Probe metadata must record this site, the zero-based
  layer, model revision and role order. Do not apply weights to a different site.
- Preserve the upstream training scale (configured 250 passages, up to 1,024
  tokens, all content tokens). Do not silently subsample tokens or reduce training
  data to save time. Small conversational evaluation is separate from training.
- [Prior-work audit](agent_docs/PREVIOUS_ROLE_PROBES.md) documents reusable old
  probes and methodological caveats. Import only artifacts needed for the task.
- Ground-truth roles come from message/channel metadata. Mean role probabilities
  and argmax role accuracy are different metrics. Tool-role transfer requires
  validation on actual tool-bearing trajectories; clean chat alone is insufficient.
- Completed baseline: layer 12 gives conversation-balanced true-role probabilities
  of 88.1% user, 62.3% analysis and 99.6% final on 9/12 completed chats. Use it as
  the initial trajectory-evaluation probe; retain layer 16 for comparison. Both
  fits emitted line-search warnings; system-prompt transfer was poor, and actual
  tool-message transfer remains unvalidated. Consult the report before reuse.
