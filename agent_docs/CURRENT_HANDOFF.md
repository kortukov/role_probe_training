# Current handoff — 2026-10-23

## Active SWE rerun

Current jobs: image-preparation retry **1492728_10** (one CPU, single-threaded
compression), dependent four-GPU evaluation array **1492729**, all500tasks.
488/500images cached; remaining12 all belong to preparation shard10.
Original image array1492631 had31successfulshards, one mksquashfs segfault139;
therefore original GPUarray1492632 was cancelled without ever starting inference.
Retry/evaluation submitted under the existing authorization. Do not claim model
inference started until1492729 actually runs. Monitor squeue/sacct and logs.
Source settings unchanged: modernvLLM0.32,temperature1,highreasoning,8192tokens/call,
200messages,2Mcumulative tokens,1800s/task,8tasks/GPU. Prior500task/probe results
still describe oldcohort; finalize only the complete newcohort separately.
Submission metadata: experiment_reports/agentic/swe_bench_full/metrics/rerun_submission.json.

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

## User pause — training work

Training-related work is explicitly paused. No training GPU job was launched.
Interrupted isolated runtime uv sync; untracked runtime files are preparation only.
Before resuming, correct design: probes refit ONLY on standard role-confusion
C4/Dolma data (existing250passages/fiveroles/originalsplit), never AppWorld.
AppWorld is model-update data. Earlier refit-on-AppWorld cost scenarios are invalid
for the intended experiment and must be replaced when training work resumes.
User requests only adding AppWorld normal-test observations/confusion matrices to
agentic/probe_transfer now. Standalone AppWorld results already pushed5091d2a.
Both combined reports now include AppWorld: layer12 transfer and five-layer
comparison, including per-role accuracy tables and confusion matrices. Report
pushed9712608 before final viewer validation. Both viewers passed offline browser
checks; layer-comparison viewer is1.746GB and stays local. Publish reports before
large viewer work in future. Training remains explicitly paused.
