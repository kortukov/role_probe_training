# Reproduction

Source: [TIER / DepthBench](https://github.com/anaykulkarni/TIER), commit
`7e309ca7b651a02b5ee23d2b208ff360b1ff0d9a`. Dataset/split hashes and exact training
IDs are in [data_manifest.json](data_manifest.json). Upstream code/data and
rollouts are retained under ignored `data/` and `runs/` on cluster storage.

```bash
git clone https://github.com/anaykulkarni/TIER.git experiment_reports/training_against_role_probes/depthbench/data/upstream
git -C experiment_reports/training_against_role_probes/depthbench/data/upstream checkout 7e309ca7b651a02b5ee23d2b208ff360b1ff0d9a
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/prepare_corpora.py
.tools/bin/uv sync --locked --project src/experiments/training_against_role_probes/collection_runtime
.tools/bin/uv pip install --python src/experiments/training_against_role_probes/collection_runtime/.venv/bin/python \
  --no-deps -e . -e experiment_reports/training_against_role_probes/depthbench/data/upstream
sbatch --chdir=/data/cluster/users/kortukov/role_probe_training \
  src/experiments/training_against_role_probes/collect_depthbench.sbatch
```

Collection job **1501281**, one B200, sixteen CPUs, 128 GB host RAM.
Logs: `../logs/depthbench-data-1501281.out`; source outputs:
`runs/1501281/{smoke,collection}/calls_0/` through `calls_6/`.
The seven-task smoke requires explicit reasoning before production collection.

The vLLM server uses the existing isolated `server_current` lock (vLLM0.32),
original MXFP4 GPT-OSS-20B revision `6cee5e81ee83917806bbde320786a8fb61efebee`,
32768 maximum context, Triton attention/MoE, 16 concurrent sequences.
Sampling: temperature1, medium reasoning, 8192 new tokens per assistant call,
24 assistant-turn limit. Seed = 9100000 + task_id*10000 + attempt*100 + turn.
20 attempts/task. All raw generation responses, errors and limits are retained.

The adapter follows the upstream chain-enabled training partition: the first50
zero-call tasks plus all call-bearing tasks whose tools are in the upstream
training function set. It yields 944 tasks. The upstream test loader subsequently
renames GetLocation to GetCurrentLocation; we use no test tasks. Tools are the
relevant set plus3 seeded distractors from the training tool pool; this adapts the
upstream default distractor count to interactive native function calling. A fresh
copy of local tool state is used per rollout. Reference calls/answers are not
sent to the model. We do not claim to reproduce TIER's original reward protocol.

After collection, explicitly verify all task/attempt identities and combine:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/assemble_depthbench.py \
 --collection experiment_reports/training_against_role_probes/depthbench/runs/1501281/collection \
 --error-audit experiment_reports/training_against_role_probes/depthbench/collection_error_audit.json
sbatch --chdir=/data/cluster/users/kortukov/role_probe_training \
 --job-name=depthbench-probes --time=02:00:00 \
 src/experiments/training_against_role_probes/evaluate_corpus.sbatch \
 --input experiment_reports/training_against_role_probes/depthbench/data/train_1_2_3/trajectories.jsonl \
 --output experiment_reports/training_against_role_probes/depthbench/data/baseline
```

The expected combined size is 12,980. Groups0/4/5/6 stay separate and do not enter
training. Failed task-solving attempts remain included; model API failures must
be audited before assembly. No task-success metric is inferred from probe scores.
Baseline evaluation uses the same full-context BF16 model and five probes as
[ASTRA](../astra_sft_1k/RUNNING.md). No model training is started before reporting
the baseline and receiving [implementation readiness](../AGENT_COMMUNICATION.md).

Completed baseline job **1503678** scored all12,980 verified training histories using
`evaluate_corpus.sbatch`; output: `data/baseline/`. The original collection
remains intact, including its error field. The explicitly audited failure is
`depthbench/364/attempt17`; [collection_error_audit.json](collection_error_audit.json)
records exact record/error hashes. Assembly permits only these reviewed failures,
requires exact coverage, and rejects new or modified failures. See
[collection_manifest.json](collection_manifest.json) for all group/output hashes.

The malformed generated tool name contains literal Harmony control tokens.
Retrospective replay preserves it; labels still come exclusively from structured
message/channel metadata, and header/control tokens are excluded from scoring.
No final response is fabricated for the aborted history.

Generate the compact metrics and appendix:

```bash
.tools/bin/uv run --frozen --no-sync python src/experiments/training_against_role_probes/summarize_corpus.py \
 --baseline experiment_reports/training_against_role_probes/depthbench/data/baseline \
 --report experiment_reports/training_against_role_probes/depthbench \
 --expected-trajectories 12980
```

[baseline_submission.json](baseline_submission.json) records source/probe hashes;
[baseline_manifest.json](baseline_manifest.json) records model/replay settings.
All histories use full context; no truncation, segmentation or downsampling.
