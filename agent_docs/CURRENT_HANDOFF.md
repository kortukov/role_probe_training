# Current handoff — 2026-10-23

## Active throughput calibration

User resumed training work specifically to start throughput measurements.
Job **1493850**, two B200s on vca-gpu-0701-04, passed its two-replica smoke test.
It proceeds through 4096/8192/16384/full debug trajectories, effective batch16,
microbatch1, one warmup and two measured updates, plus longest-example backward.
Entry: `src/experiments/training_against_role_probes/throughput.sbatch`.
Implementation: `src/role_probe_training/training/throughput.py`.
Runtime is isolated under the experiment's `runtime/`; do not change active code.
Outputs: `experiment_reports/agentic/training_against_role_probes/data/throughput/1493850/`.
Logs: that experiment's `logs/role-throughput-1493850.out`.
W&B project: https://forge.coreweave.com/wandb/kortukov/training_against_role_probes

This first phase measures **role-only model updates with a fixed standard-data
probe**. It excludes KL and probe-refit costs; never label it end-to-end throughput.
No production training or 27k trajectory collection is authorized by this benchmark
request. No full training checkpoint is saved/promoted. Measure KL and standard-data
extraction/refitting separately before projecting end-to-end cost. Review measured
memory and timing; GPU smoke success alone does not verify full-context feasibility.

## Confirmed training design

Improve role recoverability, five-class cross-entropy, layer12 post-attention norm
(pre-MLP, zero-based). Probe refitting uses ONLY the standard role-confusion
C4/Dolma250passages × five roles,1024tokens,originalsplit,allcontenttokens.
Never fit probes on AppWorld. Removed obsolete AppWorld-refitting cost scenarios.
AppWorld fixed text is model-update data. Planned corpus:300attempts/task ×90tasks
=27,000trajectories. Existing300 are debug-only, allreasoning,117solved,noerrors,
median12466/max33241tokens. Debug source is `data/debug/trajectories.jsonl`.
Prefer full trajectories; benchmark start-anchored prefixes, no arbitrary windows.
Warm-start refits after each model update. KL ordinary Dolci-Instruct-SFT prompts
plus once-generated baseGPT-OSS answers, KL(base||adapted), coefficient16,16pairs.
KL anchor data not prepared yet. W&B credentials verified, netrc; server
https://api.forge.coreweave.com, entity kortukov. Never print keys.
Working defaults LoRA64/alpha128/dropout0,AdamW5e-5,weightdecay1e-4.
Benchmark targets attention q/k/v/o and fused expert gate/up/down,739,639,296
trainable parameters; token weighting and clipping1 are calibration settings.
Final role weighting, context, effective batch and update budget remain decisions.
Atbatch16 one27kepoch means1688updates. Do not equate upstream30updates tooneepoch.

## Active SWE rerun

Image preparation completed; evaluation array **1492729** started. Last seen shards
0 and1 running. Check all four shards via sacct before declaring complete.
All500tasks, modernvLLM0.32,temperature1,highreasoning,8192tokens/call,
200messages,2Mcumulative tokens,1800s/task,8tasks/GPU. Prior500task/probe results
still describe oldcohort; finalize complete newcohort separately.
Submission metadata: `experiment_reports/agentic/swe_bench_full/metrics/rerun_submission.json`.

## AppWorld and probe reports complete

All675tasks collected without evaluation errors. Train38/90,normaltest45/168,
challengetest105/417. Canonical JSONL under `experiment_reports/agentic/appworld/data/`.
Normaltest168trajectories scored at layers4,8,12,16,20; predictions under
`experiment_reports/agentic/appworld_probe_transfer/predictions/`.
Both combined reports include AppWorld: layer12 `probe_transfer` and five-layer
`probe_layer_comparison`, per-role tables and appendix confusion matrices.
No majority-baseline rows in layer tables. Local standalone viewers updated and
browser-validated; five-layer viewer1.746GB. Reports already pushed before viewers.

## Standing preferences

Use uv, maximum four experiment GPUs, two preferred for training. No subagents
unless requested. Push code/reports, never trajectory data. Brief paper-style
reports per `agent_docs/REPORT_WRITING.md`; remove superseded material. Publish
report tables first, then build/validate large viewers. Timed waits, no busy polling.
