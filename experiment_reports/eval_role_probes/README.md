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
| Five roles, layer 12 | 69.1 | 41.8 / 44.9 | 39.9 / 40.1 | 30.8 / 18.1 | current candidate |
| Five roles, layer 16 | 85.3 | 50.6 / 47.0 | 38.7 / 51.0 | 36.7 / 30.8 | current candidate |
| Ten header roles, layer 12 | 49.0 | 22.7 / 28.4 | 31.1 / 50.4 | 23.0 / 8.4 | current candidate |
| Ten header roles, layer 16 | 68.4 | 23.8 / 7.2 | 30.5 / 51.4 | 25.3 / 5.3 | current candidate; training uses it since 2026-11-13 |
| Five roles, layer 12, cuML | 68.9 | 34.6 / 26.5 | 30.4 / 19.9 | 43.6 / 36.2 | old template |
| Five roles, layer 16, cuML | 84.2 | 29.7 / 31.7 | 30.6 / 33.1 | 38.3 / 29.2 | old template |

All probes are fitted on the neutral passages. Balanced accuracy averages over the roles
that occur in the benchmark (five, or seven to nine of the ten). Every conversation was
scored (168, 114 and 500). The ten-role numbers are not directly comparable with the
five-role ones: ten classes are a harder task. Merging the ten-role predictions back into
the five roles gives, at layer 16, 21.0% overall on AppWorld, 54.6% on retail and 19.1% on
SWE-bench, against 47.0%, 51.0% and 30.8% for the five-role probe.

Base-model task success under the current setup: AppWorld 21.4% (old template 26.8%),
retail 57.0% (64.0%), SWE-bench 52.0% (50.8%).

Known weak spots: the five-role probes never recognise system tokens in agent runs (0%);
the ten-role probes recognise the developer message (79–93% at layer 16) but call most
reasoning and tool output something else (analysis 2–42%, tool output under 5% at layer
16). Per-role numbers for both role sets are in the
[ladder progress log](../training_against_role_probes/on_policy_progress/README.md) on the
`on-policy-ladder` branch until the full report is written.

## Folders

| Folder | Contents | Status |
|---|---|---|
| `agentic_eval/appworld/harmony_base_t1/` | AppWorld test run of the base model | current |
| `agentic_eval/swe_bench/harmony_base_t1/` | SWE-bench Verified run of the base model | current |
| `agentic_eval/tau2_retail/harmony_base_t1/` | τ² retail run of the base model | current |
| `agentic_eval/probe_scoring/harmony_lbfgs/`, `harmony_lbfgs_header/` | probe scores of the runs above with the five-role and the ten-role probes, layers 12 and 16 | current |
| `agentic_eval/appworld/harmony_base/`, `tau2_retail/harmony_base/`, `tau2_retail/harmony_base_ctx131k/` | the same runs at temperature 0, which makes the model repeat itself in its reasoning (AppWorld 12.5%, retail 45.6–50.0% task success; one SWE-bench shard solved 45 of 125 tasks before the run was stopped and its data deleted) | superseded by temperature 1 |
| `agentic_eval/*/base_trajectories*/` | base-model runs behind every number published before 2026-11-13 | old template |
| `agentic_eval/probe_scoring/probe_transfer/`, `probe_layer_comparison/`, `appworld/probe_transfer/` | probe scores of those runs, layers 4–20 | old template |
| `chat_eval/norobots/` | probes on human chat conversations | earlier probes (cuML) |
