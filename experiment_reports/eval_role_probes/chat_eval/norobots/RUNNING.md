# Running this experiment

- Code: [norobots/](../../../../src/experiments/eval_role_probes/chat_eval/norobots) `prepare.py`
  (dataset, both message versions), `generate.py` (vLLM client, exact Harmony prefixes),
  `score.py` (whole-conversation scoring of several probe arms), `report.py` (tables),
  and the hx2 jobs in `norobots/hx2/`.
- Probes: the three arms of the [training-data comparison](../../../gptoss_role_probes/training_data/RUNNING.md).
- Data: `inputs/samples.jsonl` and `artifacts/generated.jsonl` are ignored; `inputs/manifest.json`
  and `artifacts/generation_summary.json` record provenance and counts.
- Model: GPT-OSS-20B `6cee5e81…`, served natively by vLLM 0.32.0 for generation and
  dequantised to BF16 with eager attention for scoring.

```bash
export HF_HOME=/hx2-weka/home/xy2222/.cache/huggingface
uv run --frozen --no-sync python src/experiments/eval_role_probes/chat_eval/norobots/prepare.py
g=$(sbatch --parsable src/experiments/eval_role_probes/chat_eval/norobots/hx2/generate.sbatch)
sbatch --dependency=afterok:$g src/experiments/eval_role_probes/chat_eval/norobots/hx2/score.sbatch
uv run --frozen --no-sync python src/experiments/eval_role_probes/chat_eval/norobots/report.py
uv run --frozen --no-sync python src/experiments/eval_role_probes/chat_eval/norobots/analyze.py   # context analyses, CPU only
```

Job IDs are in `logs/job_ids.txt` (generation 1594853, scoring 1594854; two earlier submissions failed before any model call). Outputs: `metrics/version1_replay/metrics.json`,
`metrics/version2_generated/metrics.json` (per-token predictions beside them, ignored)
and `metrics/tables.md`.
