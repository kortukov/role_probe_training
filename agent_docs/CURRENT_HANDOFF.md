# Current handoff — 2026-10-23

## Active SWE rerun

User requested the full repaired SWE-bench Verified rerun, then conversation compaction before discussing AppWorld training.

- CPU image-preparation array: **1492631**, 32 shards with at most 16 concurrent. Uses pinned official images and verified repair blobs; writes `.cache/swe-images/*.sif` and provenance sidecars.
- Dependent GPU evaluation array: **1492632**, four one-GPU shards, all 500 tasks. Starts only after all image preparation succeeds; dependency failure cancels it.
- Inspect `squeue` / `sacct` and `experiment_reports/agentic/swe_bench_full/logs/`. Runs export automatically under `runs/<job-ID>/`.
- Configuration/reproduction: `experiment_reports/agentic/swe_bench_full/RUNNING.md`, submission metadata under `metrics/rerun_submission.json`.
- Modern vLLM 0.32.0 fixed the observed Harmony failures: 8 request replays passed; one complete task reached scoring with 36 model calls and no infrastructure errors (patch unsuccessful).
- New full run: temperature 1, high reasoning, 200-message limit, 8192 tokens/call, 2m cumulative tokens, 1800 seconds/task, concurrency 8 per GPU. Current published 11/500 and probe reports still use the prior cohort; replace only after complete finalization. Do not combine old and new model attempts.
- If image preparation fails, fix/retry infrastructure and resubmit dependent GPU array; never claim evaluation has started while waiting on dependency.

## AppWorld ready for discussion

All 675 tasks collected, no evaluation errors, no model/probe optimization performed:
- Train: 38/90 solved (42.2%), 6.75 minutes on two GPUs.
- Normal test: 45/168 (26.8%), 14.39 minutes.
- Challenge test: 105/417 (25.2%), 38.96 minutes.
- Four-task train timing pilot: one GPU sequential 2.85 min; one GPU concurrent 1.81 min; two GPUs concurrent 1.90 min. Single sampled batches, not a controlled throughput scaling claim.
- Canonical corpus: `experiment_reports/agentic/appworld/data/{train,test_normal,test_challenge}.jsonl`; all preserve reasoning; role/token alignment checked with existing probe tokenizer. No AppWorld probe predictions yet.
- Full run 1489157; one-GPU pilot 1489112. Train/test IDs disjoint. Keep test content held out; use train for development.
- Report and reproduction: `experiment_reports/agentic/appworld/{README,RUNNING,APPENDIX}.md`.

## Standing preferences

Use uv and existing locked environments; no more than four GPUs. No subagents unless requested. Push code/reports, never trajectory data. Paper-style concise reports per `agent_docs/REPORT_WRITING.md`; remove superseded material once replaced. Use timed waits, not busy polling. Do not change code used by active jobs. User wants to discuss AppWorld training next; do not launch optimization without a defined experiment.

## Off-policy role training — active preparation

User confirmed model objective: **improve role recoverability**, not evade probes.
New experiment: `experiment_reports/agentic/training_against_role_probes/`.
Collection array **1492660**: four one-GPU jobs, 90+90+90+30 training attempts,
seeds 7301–7304. Assembly **1492664** depends on successful collection and writes
`data/train.jsonl` plus tracked small `corpus.json`. SWE 1492632 now depends on
`afterok:1492631,afterany:1492660` to avoid exceeding four GPUs.
Read DESIGN.md before implementing optimization. User wants discussion first.
Upstream honesty uses warm-started `retrain` (not from-scratch `reinit`), 30 model
updates, 16 main examples and 16 KL pairs/update, KL coefficient16, LoRA64/128.
Our proposed adaptation is five-way correct-role CE at pre-MLP layer12; layer,
role weighting and sequence unit/context/budget still need discussion. No training
loop launched. KL should use ordinary Dolci-Instruct-SFT prompts with base GPT-OSS
answers (not the removed Tool-Use corpus). W&B key absent from configured ~/.env;
online logging not yet authenticated.
