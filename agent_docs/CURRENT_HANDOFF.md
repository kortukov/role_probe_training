# Current handoff — 2026-10-23

## Active first-training preparation

User authorized **2,700 AppWorld trajectories**, 30 seeded attempts per each90train
 tasks, baseline probe evaluation, implementation and launch of first training.
Experiment MOVED to `experiment_reports/training_against_role_probes` (data too).
All own source/report paths updated. Do not recreate old agentic location.
`AGENT_COMMUNICATION.md` starts as a placeholder; second agent has appended dataset
and GPU coordination notes. PRESERVE those notes. The shared implementation passed GPU tests and has a scoped READY-for-reuse
signal. Do not claim AppWorld production is running until job1501541 actually starts.
Other agent owns ASTRA/DepthBench data adapters, replay.py ending support and its
subreports. Do not stage/change their files. Coordinate GPU jobs via file; the user explicitly removed the aggregate four-GPU cap.

Current own jobs:
-1501202 COMPLETED all30shards:2700attempts,1032solved(38.2%),allreasoning,noerrors/limits.
-1501282 COMPLETED assembly:data/train2700/trajectories.jsonl,train_corpus.json.
-1501329 array0..1 RUNNING: five-layer full-history baseline scoring.
-1501382 queued afterscoring:baseline report.
-1501541 queued afterbaseline report:productiontwoGPUtraining,169updates,oneepoch.
-1501542 queued afterany1501541:publish finalnumericresults/README/APPENDIXtoGitHub.
-1501243 COMPLETED:fixedKL800ordinaryDolci anchors.
-1501280 COMPLETED:eightdisjointdevtasks.
Pilot1501447+resume1501491passedcompleteupdates/refits/W&B/checkpoints/OOMretry.
SharedimplementationREADYforotheragentreuse at03c1937; AppWorldproductionnotyetstarted.
READY-for-ASTRA release is at topofAGENT_COMMUNICATION; preserveotheragentnotes.

Shared training implemented under `src/role_probe_training/training/`:
 data.py,model.py,objectives.py,refit.py,train.py,score.py.
Train configs/jobs under `src/experiments/training_against_role_probes/`.
CPUchecks.py passed KLdirection/mask/checkpointedgradients andpadding.
**GPU pilot1501447 and resume1501491 COMPLETED successfully** in ber-testing. Attention compile limit raised to64;
fails explicitly at limit instead of quadratic eager fallback. Uses train.sbatch
pilot_config.json on2GPUs (ber-testing,time1h acceptable forvalidation), and
resume checkpoint withmax_updates2 proved restart. Pilot config usesdebug300,
fullprotocoloneupdate,2000refitsteps,16KLpairs,deliberate coordinatedOOMretry.
Production config train_config.json:2700,oneepoch169updates,fulltrajectories,
microbatch2/GPU,effective16,tokenweightedCE,layer12,KLcoefficient16.
Regular validation8devtasks every5updates;checkpoint every10,keeplast2.
Warmstart probeSGD2000steps,lr1e-3,wd1e-4,batch32passage-roleexamples,allcontenttokens,
original250C4/Dolmapassages/originalsplit. NOTAppWorldprobe fitting.
Standardactivationextraction batched16/rank,cachedGPU. Measure real refitcost.
W&B configuredat api.forge.coreweave.com entitykortukov projecttraining_against_role_probes.
Need finish reportupdates, commitownfiles only, evaluate2700 corpus,
launchproductionafterfullbaseline, updatecommunicationREADYthenreportuser.
Productiontrain1501541 is queued; userexplicitlyauthorizedlaunchwhenready.

## Completed throughput calibration

Original context job1493850 completed all4settings in9m28s. Role-only update means
4K11.8s,8K16.9s,16K20.2s,full21.5s; fullpeak59.9GiB,33kstress63.8GiB.
Small results: `experiment_reports/training_against_role_probes/context_results.json`.
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
Outputs `experiment_reports/training_against_role_probes/data/microbatch/1500991/`.
Log `logs/role-microbatch-1500991.out` in that experiment.
Aggregator: `summarize_microbatch.py --runs ... --output .../microbatch_results.json`.
Throughput benchmark complete; current collection/training preparation described above.

Production OOM recovery retries the whole optimizer update on both ranks with
microbatch1 and unchanged effective batch size; pilot tests this with an injection. Keep resolved correctness investigations in code/local
logs, not scientific reports or published result summaries.

W&B project: https://forge.coreweave.com/wandb/kortukov/training_against_role_probes
Standing auth to push code/reports, no data. Results summarized in README/APPENDIX/RUNNING; push code and small results only.

## Confirmed training design

Improve role recoverability, five-class cross-entropy, layer12 post-attention norm
(pre-MLP, zero-based). Probe refitting uses ONLY the standard role-confusion
C4/Dolma250passages × five roles,1024tokens,originalsplit,allcontenttokens.
Never fit probes on AppWorld. Removed obsolete AppWorld-refitting cost scenarios.
AppWorld fixed text is model-update data. First corpus:30attempts/task ×90tasks =2,700trajectories. Existing300 are debug-only, allreasoning,117solved,noerrors,
median12466/max33241tokens. Debug source is `data/debug/trajectories.jsonl`.
Prefer full trajectories; benchmark start-anchored prefixes, no arbitrary windows.
Warm-start refits after each model update. KL ordinary Dolci-Instruct-SFT prompts
plus once-generated baseGPT-OSS answers, KL(base||adapted), coefficient16,16pairs.
KL anchor generation is complete. W&B credentials verified, netrc; server
https://api.forge.coreweave.com, entity kortukov. Never print keys.
Working defaults LoRA64/alpha128/dropout0,AdamW5e-5,weightdecay1e-4.
Benchmark targets attention q/k/v/o and fused expert gate/up/down,739,639,296
trainable parameters; token weighting and clipping1 are calibration settings.
First run uses token weighting, full context, effective batch16 and one epoch169updates.

## SWE rerun incomplete

Array **1492729** finished: shards0/1 completed; shards2/3 failed during startup
(exit1 after15seconds). No SWE job is running. Investigate failed shards and
finish the cohort before reporting a new full-evaluation result.
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

Use uv. User explicitly allows more than four experiment GPUs; request reasonable resources per job. Two GPUs remain preferred for training. No subagents
unless requested. Push code/reports, never trajectory data. Brief paper-style
reports per `agent_docs/REPORT_WRITING.md`; remove superseded material. Publish
report tables first, then build/validate large viewers. Timed waits, no busy polling.

Pilot completed two updates with a checkpoint restart. Dev accuracy29.2%→39.2%→54.7%,
KL secondstep0.00150, refit47–51s. Keep pilot measurements out of main report.
Production waits for corpus assembly1501282, score1501329 and report1501382.
Automatic final-report publisher: report_training.sbatch RUN_DIRECTORY (afterany training).
