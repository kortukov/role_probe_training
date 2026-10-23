# Current handoff — 2026-10-24

## Active AppWorld training

User requested2700AppWorld training trajectories, baseline probes, and a complete
training implementation/start. All prerequisites completed and production is RUNNING.
Report root: `experiment_reports/training_against_role_probes/` (moved outofagentic).

- Collection1501202 and assembly1501282 COMPLETED:2700attempts=30×90tasks,
  1032solved(38.2%), allreasoning, zeroerrors/limits. Canonicaldata/train2700/trajectories.jsonl.
- Baseline1501329 COMPLETED:all2700,fivelayers4/8/12/16/20,fullhistory,
  pertokenpredictionsunderpredictions/train2700. Layer12accuracy26.12%,macro33.73%.
  Report/appendix+baseline_probe_results.json pusheda61a58e.
- Production **1501541 RUNNING**,twoB200s,vca-gpu-0702-03,livewalltime12h.
  Oneepoch169updates,effective16,microbatch2/GPU,fulltrajectories.
  Outputs:runs/training/appworld2700; config:src/experiments/training_against_role_probes/train_config.json.
- Finalreportpublisher **1503672** afterany1501541 updates/pushes numericresults,
  README andAPPENDIX. Reproduce via report_training.py --run RUN_DIRECTORY.
- Initial5updates:devaccuracy29.2%→80.6%,macro35.4%→42.4%; reasonrecall77.4%→13.8%,
  user41.1%→19.6%; assistant/toolsimproved. Do not present overallgain asuniformrecovery.
  Reporterautomaticallyflagsperroledeclines. Keepagreedtokenweightedobjectiveunchanged.
- Warmupdates73–83s includingrefit32s; initial122s. Memorypeak<79GiB sofar, noOOMretry.
- W&B:https://forge.coreweave.com/wandb/kortukov/training_against_role_probes/runs/appworld2700-l12-epoch1

Training code stable at03c1937; do notedit sharedtraining/*.py whilejobsrun/queued.
Pilot1501447+resume1501491 passed fullupdates,KL,all2000refitsteps,dev,W&B,
checkpointsave/restore(optimizer+probe+adapter) andcoordinatedinjectedOOMretry.
Production retries wholeupdateatmicrobatch1, samebatch/objective; failurethere stops.
Attentioncompilelimit64/fail-explicit preventsquadratic eagerfallback.
Checkpointinitial,every10updates,final;keeplast2. ResumeCLI inRUNNING.md.

Modelgoal: IMPROVE role recoverability, layer12 postattentionnorm/preMLP, fiveclassCE.
Probe refit ONLY standard250C4/Dolma passages/originalsplit/allcontenttokens; neverAppWorld.
WarmSGD2000steps,lr1e-3,wd1e-4,32passageroleexamples/minibatch,50warmup+cosine.
ModelLoRA64alpha128dropout0,AdamW5e-5wd1e-4,clip1,globaltokenmean.
KL800fixedordinaryDolci+GPTOSSbaseanswers,data/kl/anchors.jsonl (job1501243complete),
16sampled/update,coefficient16,unshifted2048completionmask upstreamprotocol.
SamefrozenmodelwithLoRAdisabled actsasteacher; nosecondfullmodel.
Dev8disjointAppWorldtasks(job1501280) runs/dev/1501280/dev/trajectories.jsonl.
Fulltrainingcorpus fixed/offpolicy. Validationfixedbase-modeltrajectories; taskbehaviornotyettested.

## Concurrent data/training agent

EXPLICIT READY atAGENT_COMMUNICATION.md; AppWorldproductionstarted, allrequirementsmet.
Secondagent owns ASTRA/DepthBench adapters/configs/subreports and replay.py optionalending.
Preserve theirfiles/notes; stageonlyyourownfiles. ASTRAtraining **1503671** submittedtwoGPUs12h,
1000histories63updates afterpublishedbaseline. DepthBench1501281 collecting,baseline1501448
queued; owntrainingafterbaseline,12980histories(groups1/2/3),812updates,24h planned.
The user removedaggregatefourGPUcap; use reasonable resources,trainingtwoGPUs/run.
Allruns distinctoutput/W&B. Shareddev/KL allowed; explicitlycross-domainforotherdatasets.

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
All data/scoring/report prerequisites completed. Production running; wait fordevstep5,
run report_training.py andpush initialresults. Finalpublisher1501542 handlescompletion.
Automatic final-report publisher: report_training.sbatch RUN_DIRECTORY (afterany training).
