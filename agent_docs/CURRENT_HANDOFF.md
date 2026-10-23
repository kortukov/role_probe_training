# Current handoff — 2026-10-23

## Completed throughput calibration

Original context job1493850 completed all4settings in9m28s. Role-only update means
4K11.8s,8K16.9s,16K20.2s,full21.5s; fullpeak59.9GiB,33kstress63.8GiB.
Small results: `experiment_reports/agentic/training_against_role_probes/context_results.json`.
No KL/refitting overhead measured; these are not end-to-end training estimates.

User requested paired full-trajectory microbatch1/2/4 comparison, effectivebatch16.
Job **1500991** COMPLETED in10m51s on two B200s in ber-testing (vca-gpu-0701-01).
Microbatch1/2/4 means26.60/23.18/24.48s/update; peakupdate63.0/76.5/104.1GiB;
stress63.8/79.7/111.8GiB. Microbatch2 throughput1.148×baseline; padding0/7.9/21.9%.
Allconditions/rankscompleted. Smallsummary `microbatch_results.json`; reportupdated.
Script `src/experiments/training_against_role_probes/microbatch.sbatch`, shared
implementation `src/role_probe_training/training/throughput.py`, isolated runtime.
Five identical seeded batches percondition, sortwithinbatchbylength, one warmup,
four measured updates, plus repeatedlongesttrajectory stressbackward.
Outputs `experiment_reports/agentic/training_against_role_probes/data/microbatch/1500991/`.
Log `logs/role-microbatch-1500991.out` in that experiment.
Aggregator: `summarize_microbatch.py --runs ... --output .../microbatch_results.json`.
No benchmark remains active. No productioncollection/training launched.

User agreed activation identity is not necessary; check loss/gradient agreement.
On4unequalprefixes/5618labeledtokens, batchedvsindividual lossrelativeerror0.0413%,
gradientcosine0.998906,gradientrelativeL2error4.694%. Bothranks passed.
Use original TransformersFlexAttention. Trial FP32sinknormalization did not resolve
activation differences and was removed. No unvalidated kernel adaptation remains.
Gate loss<1%,gradientcosine>.99,gradientrelativeerror<10%; numericalcalibration,
not universaltrainingguarantee. User asked about OOMresilience: production could
retrywholeupdatecoordinatedacrossranks withsmallermicrobatch/sameeffectivebatch,
but this is not implemented; benchmarkkeepssizestoenablefaircomparison.

W&B project: https://forge.coreweave.com/wandb/kortukov/training_against_role_probes
Standing auth to push code/reports, no data. Results summarized in README/APPENDIX/RUNNING; push code and small results only.

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
