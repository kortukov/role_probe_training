# Running this experiment

- Code: [prepare.py](../../../src/experiments/gptoss_role_probes/prepare.py) (Toucan source,
  server-disjoint sampling, native headers), [run.py](../../../src/experiments/gptoss_role_probes/run.py)
  (`train_sources`, per-collection metrics), hx2 job scripts in
  [hx2/](../../../src/experiments/gptoss_role_probes/hx2/).
- Configs: [training_data/](../../../src/experiments/gptoss_role_probes/training_data/)
  `c4_dolma.json`, `toucan.json`, `combined.json`; identical except `train_sources`.
- Data: prepared inputs in `inputs/` (ignored; `toucan_manifest.json` lists sample UUIDs,
  counts and skip reasons). Each arm's `artifacts/inputs` links to it.
- Revisions: GPT-OSS-20B `6cee5e81…`; C4/Dolma revisions as in the baseline config;
  Toucan-1.5M revision `0df3cf37f2abefb380370cfb02eabea2a35ae782`.
- Hardware: one H200 per job on hx2 (see [HX2.md](../../../agent_docs/HX2.md)).

From the repository root on hx2:

```bash
uv sync --locked --extra vllm
uv sync --locked --project src/experiments/gptoss_role_probes/fit_env
export HF_HOME=/hx2-weka/home/xy2222/.cache/huggingface
uv run --frozen --no-sync python src/experiments/gptoss_role_probes/prepare.py \
  --config src/experiments/gptoss_role_probes/training_data/c4_dolma.json \
  --output experiment_reports/gptoss_role_probes/training_data/inputs
sbatch --partition=interactive-gpu --mem=60G src/experiments/gptoss_role_probes/hx2/smoke.sbatch
for arm in c4_dolma toucan combined; do
  sbatch --job-name=role-probe-train-$arm src/experiments/gptoss_role_probes/hx2/train.sbatch \
    --config src/experiments/gptoss_role_probes/training_data/$arm.json \
    --output experiment_reports/gptoss_role_probes/training_data/$arm --stage train
done
# To compare fitters, add `--backend cuml adamw` to the train and evaluate commands: one activation
# extraction then also fits the training_against_probes step-0 probe (AdamW, 5,000 steps) into
# $arm/probes_adamw/ with metrics/synthetic_adamw.json, next to the cuML results.
# Clean chat: generate the 12 conversations once in c4_dolma, then reuse them (--from) in the other arms.
sbatch src/experiments/gptoss_role_probes/hx2/evaluate.sbatch --config .../c4_dolma.json --output $root/c4_dolma
sbatch src/experiments/gptoss_role_probes/hx2/evaluate.sbatch --config .../toucan.json --output $root/toucan --from $root/c4_dolma
sbatch src/experiments/gptoss_role_probes/hx2/train.sbatch --config .../toucan.json --output $root/toucan --stage audit
# Whole-history scoring of all three probe sets on the hx2 AppWorld cohort (see its RUNNING.md):
sbatch src/experiments/eval_role_probes/agentic_eval/probe_scoring/hx2/whole_score.sbatch \
  experiment_reports/eval_role_probes/agentic_eval/appworld/base_trajectories_hx2/runs/1584591/test_normal/trajectories.jsonl \
  $root/predictions/appworld_test_normal $root/c4_dolma $root/toucan $root/combined
# Tables, appendix and plot from saved metrics:
uv run --frozen --no-sync python src/experiments/gptoss_role_probes/training_data/report.py \
  --appworld $root/predictions/appworld_test_normal
uv run --frozen --no-sync python src/experiments/gptoss_role_probes/training_data/plot.py
uv run --frozen --no-sync python src/experiments/gptoss_role_probes/training_data/cross_evaluation.py
uv run --frozen --no-sync python src/experiments/gptoss_role_probes/training_data/confusion_tables.py
```

Slurm job IDs are in `logs/job_ids.txt`: fits 1584571–1584573, clean chat 1584582–1584584,
audits 1584585–1584587 (rerun as 1601043–1601045 with training accuracy and as
1604677 with training confusion counts; the duplicate 1604673–1604675 were cancelled
unstarted), AppWorld scoring 1584629. Per-token predictions stay in the ignored
`predictions/` folder; `metrics/summary.json` keeps every aggregate used in the report.

Outputs per arm: `probes/layer_*.npz|json`, `metrics/synthetic.json` (overall and
`by_collection`), `metrics/conversations.json`, `metrics/run_metadata_*.json`.
