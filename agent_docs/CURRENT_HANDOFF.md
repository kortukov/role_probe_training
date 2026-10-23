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

User-confirmed objective: improve role recoverability; layer12 and multiclass CE.
**Corrected data target: 300 attempts PER TASK × 90 tasks = 27,000 trajectories.**
Existing collection1492660/assembly1492664 completed; those300 are **debug only**.
Path: `experiment_reports/agentic/training_against_role_probes/data/debug/trajectories.jsonl`.
Small metadata: debug_corpus.json, debug_batch_stats.json, debug_submission.json.
Scripts now collect_debug.sbatch / assemble_debug.py / assemble_debug.sbatch.
No production27k collection or model optimization launched. User prefers full
trajectories; consider microbatch1/gradient accumulation before any truncation.
Debug median12466/max33241 tokens,4.10M total; allreasoning,117/300success,noerrors.
Read DESIGN.md: full training ~369Mtokens extrapolated,1688updates/epoch atbatch16.
Need decide full-run update budget and probe-refit pool (all27k everyupdate costly).
Warm-started upstream probes, KL ordinary Dolci-Instruct-SFT baseGPT-OSS answers.
W&B verified via netrc read-only API as kortukov at https://api.forge.coreweave.com.
Set WANDB_BASE_URL to that host in jobs; no login needed.
Fresh independent probes = new initializations, fit training-side data on frozen
base/final checkpoints, same held-out dev evaluation; no test fitting.

Capacity discussion: batch_stats.py now measures4k/8k/16k prefixes and complete
tool replies. estimate_cost.py/cost_estimate.json contain illustrative UNMEASURED
rates2000train/8000extract tokens/s/GPU; do not cite those as benchmarks. Full
epoch1688updates and369Mtokens. All27k refitted everyupdate implies~623Btoken
positions: not a viable default. Fitting pool/budget remains undecided. Meta
parameter count BF16base38.96GiB; candidate all-expert+attention LoRA64 state
~11.02GiB. Peak training memory/runtime not measured; no training loop exists yet.
