# Base-model probe evaluation: status

Stage 1 of the project: how well does a probe setup recognise roles in real runs of the
*unchanged* GPT-OSS-20B? A setup is a way of fitting the probes (data, layer, role set,
fitter), produced in [probe fitting](../gptoss_role_probes/README.md). Every candidate is
scored here before any [training](../training_against_role_probes/README.md) uses it.

## Where we are

All numbers in %. Held-out is accuracy on passages not used for fitting. For the three
benchmarks the first number is balanced accuracy (the mean of the five per-role
accuracies) and the second is overall token accuracy, on whole conversations of the base
model. Current rows use the Harmony renderer, converged L-BFGS probes and temperature 1.

| Probe setup | Held-out | AppWorld test | τ² retail | SWE-bench | Status |
|---|---|---|---|---|---|
| Neutral passages, five roles, layer 12 | 69.1 | 41.8 / 44.9 | pending | pending | current candidate |
| Neutral passages, five roles, layer 16 | 85.3 | 50.6 / 47.0 | pending | pending | current candidate, planned choice |
| Neutral passages, five roles, layer 12, cuML | 68.9 | 34.6 / 26.5 | 30.4 / 19.9 | 43.6 / 36.2 | old template |
| Neutral passages, five roles, layer 16, cuML | 84.2 | 29.7 / 31.7 | 30.6 / 33.1 | 38.3 / 29.2 | old template |

Base-model task success under the current setup: AppWorld 21.4% (old template 26.8%),
SWE-bench 52.0% (50.8%), retail pending (64.0%).

Known weak spots of the current candidates: system tokens are never recognised in agent
runs (0%), and tool outputs are recognised 25–41% of the time on AppWorld.

**Pending (2026-11-13):** retail at temperature 1 and probe scoring of the temperature-1
SWE-bench run are queued. The full report of the current rows, with per-role tables and
the temperature-0 comparison, will be written when they finish.

## Folders

| Folder | Contents | Status |
|---|---|---|
| `agentic_eval/appworld/harmony_base_t1/` | AppWorld test run of the base model | current |
| `agentic_eval/swe_bench/harmony_base_t1/` | SWE-bench Verified run of the base model | current |
| `agentic_eval/tau2_retail/harmony_base_t1/` | τ² retail run of the base model | current, queued |
| `agentic_eval/probe_scoring/harmony_lbfgs/` | probe scores of the runs above, layers 12 and 16 | current |
| `agentic_eval/*/harmony_base/`, `tau2_retail/harmony_base_ctx131k/` | the same runs at temperature 0, which makes the model repeat itself in its reasoning | superseded by temperature 1 |
| `agentic_eval/*/base_trajectories*/` | base-model runs behind every number published before 2026-11-13 | old template |
| `agentic_eval/probe_scoring/probe_transfer/`, `probe_layer_comparison/`, `appworld/probe_transfer/` | probe scores of those runs, layers 4–20 | old template |
| `chat_eval/norobots/` | probes on human chat conversations | earlier probes (cuML) |
