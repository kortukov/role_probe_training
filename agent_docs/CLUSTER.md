# DE.BER1 cluster workflow

## Source and scope

Summarized on 2026-10-21 from the user-provided local copy:
`/home/fe/kortukov/workdir/prompt_injection_interp/cluster_doc.html`.
It identifies [wiki revision 28254](https://viwiki.fe.hhi.de/w/index.php?title=DE.BER1_HPC_Cluster&oldid=28254),
last edited 2026-08-31. Partition limits below come from that snapshot, not a live
scheduler query. Use `sinfo` / `scontrol show partition` to check current limits.

The guide says AI assistants must not run on cluster login nodes or in jobs and
that generated code and submission scripts must be reviewed before cluster
execution. Develop with assistants locally and transfer reviewed code to the
cluster. This setup does not submit a job or validate GPU execution.

## uv without a container

This project's default is a uv environment and direct `uv run` inside Slurm.
Apptainer/Singularity is not technically required to launch Python this way.
However, uv manages Python packages, not the host OS, NVIDIA driver or all native
libraries. Select compatible training wheels and validate them on an allocated
GPU when training dependencies are added. A container may become useful if
native dependencies cannot be supplied by the host or wheels.

The cluster guide recommends containers to reduce filesystem traffic. uv does
not provide that benefit by itself: keep environments, code and caches on
`/data/cluster` or node-local storage, never under `/home/fe` for running jobs.
See also the [uv project guide](https://docs.astral.sh/uv/guides/projects/) and
[environment configuration](https://docs.astral.sh/uv/concepts/projects/config/).

This workspace uses Python 3.12, `.venv/`, a repo-local `.uv-cache/` configured in
`pyproject.toml`, and a local uv binary at `.tools/bin/uv`. On a new machine install
uv separately; `.tools/` is not versioned. If uv downloads a Python interpreter,
set `UV_PYTHON_INSTALL_DIR` to storage under `/data/cluster` as well. This initial
environment uses the system interpreter `/usr/bin/python3.12`.

Before submitting, run `uv sync --locked --extra vllm` from the repo on appropriate storage
(see [ENVIRONMENTS.md](ENVIRONMENTS.md)).
Jobs can use `uv run --frozen --no-sync ...` against that prepared environment to
avoid downloads and concurrent environment changes. Do not change an environment
or code used by queued/running jobs; Slurm copies the submission script, not the
Python source it invokes. Use a separate checkout/environment for concurrent
experiments that need different revisions.

## Passing environment variables to Slurm (verified 2026-10-30)

Use inherited environment variables with plain `--export=ALL` when submitting
jobs on this cluster:

```bash
env RUN_CONFIG=/data/cluster/users/kortukov/config.json \
    sbatch --export=ALL src/experiments/<experiment>/job.sbatch
```

Avoid `sbatch --export=ALL,RUN_CONFIG=...` and explicit variable lists in
`--export`. The [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html#OPT_get-user-env)
states that `--export=NONE` and `[ALL,]<environment_variables>` implicitly enable
`--get-user-env`. This retrieves the user's login environment on the execution
node; failure or timeout aborts, requeues and holds the job before its script runs.
Plain `--export=ALL` passes the submitting process's environment without requesting
that additional retrieval step. In Python submission code, put extra variables in
`subprocess`'s `env` argument and pass `--export=ALL` separately.

**Live evidence:** HF benchmark 1691976 repeatedly entered
`user_env_retrieval_failed_requeued_held`; training pilot 1691926 encountered the
same failure before executing its script. Simply releasing the benchmark's hold
did not resolve it. Resubmission using `env VARIABLE=value sbatch --export=ALL`
allowed benchmark jobs 1726276 and 1726415 to execute; 1726415 completed
successfully. Training was resubmitted as 1758049–1758052 with the same pattern.
See the [HF job records](../experiment_reports/training_against_role_probes/on_policy/appworld2700/hf_generation_jobs.json)
and [training resubmission](../experiment_reports/training_against_role_probes/on_policy/appworld2700/colocated_resubmission.json).

**Diagnosis limits:** the submission syntax and documented retrieval behavior
are established; the cluster-side reason retrieval fails is not. This failure
occurs before Python/CUDA startup and is distinct from application errors.
Inspect `scontrol show job <id>` for the full reason. If replacing a held job,
first check for existing run artifacts, preserve them, and update/rebuild its
downstream dependencies; they otherwise continue pointing to the old job ID.

## Access and resources

Primary login/submit host: `hpc-ber1-headnode-01.fe.hhi.de`.
`hpc-ber1-headnode-02` has a different, local home directory; the usual CEPH home
is mounted at `/home/fe_ceph` there when available.

| Work | Account | Partition | Snapshot maximum runtime |
| --- | --- | --- | --- |
| CPU | `cpu` | `ber-amd1` | 42 days |
| GPU training/evaluation | `gpu` | `ber-gpu1` | 3 days |
| Short GPU job validation | `gpu` | `ber-testing` | 1 hour |

GPU nodes have 8 B200 GPUs with 192 GB each, 96 CPU cores without hyperthreading,
and 1.5 TB RAM according to the source. Request only the resources needed; start
with 4 CPU cores for one GPU and respect the guide's conservative cap of 16 CPUs
per GPU job. The CPU account/partition are defaults, but explicit flags are clearer.
Use the DE.BER1 names above rather than the generic `gpu`/`testing` partition
names that appear in older examples in the source.

Do not run training, evaluation or other compute workloads on login nodes. Use
Slurm allocations for interactive debugging too. `ber-testing` is for validating
jobs, not production workloads. Jobs longer than the partition limit need
checkpoint/resume support.

## Storage

- `$DATA_CLUSTER/users/$USER` (`/data/cluster/users/<user>`): active code,
  environments, submission directories, logs and saved experiment results.
  This is scratch storage without backup. Archive important completed results.
- `$LOCAL_JOB_DIR`: fast node-local scratch, initialized with
  `source /etc/slurm/local_job_dir.sh`. Use for frequent writes, temporary files
  and checkpoints. It is deleted after the job; explicitly copy required outputs
  back before exit, with time budgeted for the transfer.
- `$DATAPOOL1` / `$DATAPOOL3`: shared ML datasets with caching on GPU nodes.
  Prefer the user-managed `$DATAPOOL3/datasets/` for new datasets. Check available
  space and existing datasets first. The source differs on DATAPOOL1 write
  access; verify before writing there. Sites' DATAPOOLs are not kept in sync.
- `/data/cluster/ai/LLM/`: shared pretrained weights. Check here before downloading
  another large model. Contact the cluster maintainers to add shared weights.
- `$DATA_ARCHIVE`: long-term storage after a run, not live job I/O.
- `$HOME`: configuration and light interactive files, not job workdirs,
  datasets, environments, checkpoints or frequent logs.

Reports and small derived data/plots belong in the corresponding
`experiment_reports/<topic>/<experiment>/` directory. For large training datasets
or weights, keep a manifest/path and provenance there linking the appropriate
shared storage. Reduce progress-bar frequency. Prefer explicit result transfer
(often a tar archive for many files); `$LOCAL_JOB_DIR/results` autocopy is only a
fallback and can hold a node in COMPLETING while copying.

## Minimal direct-uv submission example

When an experiment exists, save its actual job script under
`src/experiments/<experiment>/`. Adapt the resource requests and replace the
placeholder entry point below. Review the resulting script before submission.
This single-node example assumes inputs and the uv environment are already ready.
The experiment must implement the illustrated `--output-dir` option.

```bash
#!/bin/bash
#SBATCH --job-name=role-probe-test
#SBATCH --account=gpu
#SBATCH --partition=ber-testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=experiment_reports/<topic>/<experiment>/logs/%x-%j.out

set -euo pipefail
source /etc/slurm/local_job_dir.sh
cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.tools/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TMPDIR="$LOCAL_JOB_DIR/tmp"
mkdir -p "$TMPDIR" "$LOCAL_JOB_DIR/output"

# Separate per-job artifacts; create the parent logs directory before sbatch.
run_dir="$PWD/experiment_reports/<topic>/<experiment>/data/$SLURM_JOB_ID"
mkdir -p "$run_dir"
# Preserve partial outputs on ordinary failure as well as success.
# SIGKILL/node failure cannot run this trap: plan periodic saves for long runs.
trap 'tar -C "$LOCAL_JOB_DIR/output" -cf "$run_dir/artifacts.tar" .' EXIT

srun uv run --frozen --no-sync python src/experiments/<experiment>/run.py \
    --output-dir "$LOCAL_JOB_DIR/output"
```

Submit from the `/data/cluster/...` checkout, after replacing placeholders:

```bash
export PATH="$PWD/.tools/bin:$PATH"
uv sync --locked --extra vllm
mkdir -p experiment_reports/<topic>/<experiment>/logs
sbatch src/experiments/<experiment>/job.sh
# After successful short validation, use the production partition and runtime:
sbatch --partition=ber-gpu1 --time=1-00:00:00 src/experiments/<experiment>/job.sh
```

For an interactive allocation:

```bash
srun --account=gpu --partition=ber-testing --nodes=1 --ntasks=1 \
    --gpus=1 --cpus-per-task=4 --mem=32G --time=00:15:00 --pty bash
```

Inside it, source `/etc/slurm/local_job_dir.sh`, enter the repo on `/data/cluster`,
and run the prepared experiment with uv. Adapt allocations for CPU-only jobs
using `--account=cpu --partition=ber-amd1` and omitting GPU requests.

## Monitoring and reproducibility

```bash
squeue -u "$USER"
scontrol show job <job_id>
sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode,MaxRSS
scancel <job_id>  # Only when intentionally stopping your job.
```

Record the job ID, repository/submodule commits, model/data revisions, command,
seeds, configuration and hardware in the experiment's `RUNNING.md`. Link logs,
metrics and plots from its README. Investigate failures before scaling up.
Cluster support: `hpc-cluster@hhi.fraunhofer.de`;
[current status](https://viwiki.fe.hhi.de/wiki/HPC_Cluster_Status).

### Verified node failure (2026-10-24)

Exclude `vca-gpu-0701-05` from these evaluations as well as the previously excluded
`vca-gpu-0701-03`. On `0701-05`, SWE job1504057_3 produced repeated Apptainer
`lookup userid 11850603: connection refused` failures; job1504057_6 failed to create
its Slurm local directory with permission denied. The same pipeline ran on other
nodes. This is a live observation, not a diagnosis of the node's account service.

GPU-only whole-history probe scoring was verified on `0701-05` on 2026-10-24 using a unique `/tmp` scratch directory. This does not resolve its Apptainer account lookup failure: exclude it from benchmark generation, but it can run container-free scoring.

On 2026-10-25, `vca-gpu-0702-02` also failed Apptainer account lookup
(`lookup userid 11850603: connection refused`) for every sample in SWE shards
`1504811_2` and `1504811_3`. Exclude this node from container-based evaluations.
Both jobs exited zero despite 250 image-preparation errors and no model calls;
check sample-level results, not only Slurm completion. Current SWE assembly rejects
errors occurring before any model attempt.

### SWE agent environment (verified 2026-10-25)

The original `apptainer_mirror` provider launches tools in base Conda, while SWE
dependencies live in `/opt/miniconda3/envs/testbed`. Both Inspect `python` and
`bash_session` reproduced missing dependencies on fresh Django/SymPy images;
the scorer activates `testbed` separately, masking this distinction in gold-patch
controls. New evaluations can opt into
`role_probe_training.trajectories.swe_testbed.sandbox_config`, which activates
`testbed` for each command. This provider passed direct/Python/shell controls on
both images; its task-success impact is not yet measured. Do not change the
original provider while concurrent jobs use it, or merge corrected generations
into old cohorts. [Evidence](../experiment_reports/training_against_role_probes/swe_gap/README.md).


### SWE localhost and Requests controls (verified 2026-10-27)

Pinned Apptainer images launched with the original isolated provider have empty
`/etc/hosts`. Numeric 127.0.0.1 socket connections work, but localhost resolution
fails (diagnostic1610624). The opt-in `swe_localhost` provider supplies image-local
hosts entries while retaining network isolation; Matplotlib gold controls now pass
(1610625). This is not an external-network enablement.

Requests additionally lacks HTTP fixtures in some images and expects timeout
behavior for 10.255.255.1; network-none instead returns ENETUNREACH. A CPU job
testing a separate user+network namespace failed at `/proc/self/uid_map` with
Operation not permitted (1610635). No host networking was modified. Broader
Requests gold validation1610636 and saved-patch regrade1610633 are separate from
model generation; consult the corrected-SWE pipeline before submitting runs.

### Future training/SWE CPU requests (2026-11-01)

User requested smaller requests for future submissions only, without updating or
restarting queued jobs. `on_policy_recovery/train.sbatch` now requests 8 CPU cores
(previously 16); standard `swe_generous/collect.sbatch` requests 4 (previously 16).
Accelerated SWE concurrency 8/16 retains 16 cores. GPU counts, RAM (256 GiB) and
time limits are unchanged. Slurm retains the already-submitted scripts and
requests; historical submission hashes describe those original scripts. These
smaller CPU allocations have not yet been benchmarked.

Live inspection found unallocated GPUs stranded on nodes with no available CPU
cores, plus one eligible node with one GPU and four CPU cores. Training job
1787804 averaged about 1.6 CPU cores and reported 117.85 GiB peak RSS, but averages
do not establish peak requirements. Do not reduce SWE RAM from ordinary low RSS
readings: shard 1691740_1 hit OUT_OF_MEMORY at reported 254.58 GiB.

Live priority configuration: GPU and CPU jobs use separate account associations;
usage half-life is four days, age weight 8000 (saturates at seven days), fair-share
weight 100000. GPU billing weights are CPU=1, RAM=0.0625/GiB, GPU=48.
User-specific priority is dynamic: inspect `sprio -l` and `sshare -l` rather than
inferring fair share from a cumulative GPU-days email. Nice penalties intentionally
favor training but also lower evaluations relative to other users.

### Pipeline allocation policy (user preference, 2026-11-01)

Minimize scheduler handoffs: normally one allocation for a short training run,
and only two or three long allocations for work exceeding the 72-hour GPU limit.
Each afterok boundary may wait for resources again; it does not reserve a slot.
Do not use Nice penalties to order the user's jobs: they penalize those jobs
against other users too. Do not change or restart existing queued jobs merely
to adopt these defaults.

Future on-policy recovery defaults now use endpoints 58, 114, 169 (three
48-hour segments from the original step-2 checkpoint), instead of seven
24-hour segments. Check actual progress and timing before a later recovery;
do not blindly rerun the historical restart script or reuse its old job IDs.
Per-update checkpoints remain enabled. Short runs remain one 12-hour job.

For new evaluations use `on_policy_compact/submit.py` under
`src/experiments/training_against_role_probes/`. It defaults to a dry run and
requires an explicit root, checkpoint and final training job ID. It refuses
submission if either evaluation manifest already exists. The four job groups
replace the previous twelve: export+probe fit (8h); AppWorld/retail collection
+assembly+scoring+results (60h); four parallelizable SWE collection shards
(72h each, max two concurrent); SWE assembly+scoring+results (30h). These are
one-GPU allocations; their dependency depth is at most three. SWE collection
remains separate because it can already occupy the partition's full limit.
Subordinate sbatch scripts execute with bash inside the allocation, not sbatch;
their resource directives do not create new jobs. Each child's cleanup runs
before the next stage. A failed group stops; completed artifacts remain, but
manual recovery must select unfinished work rather than blindly regenerating
completed episodes. Compact orchestration is CPU-mock checked, not GPU-run.

### Configurable on-policy GPU count (2026-11-01)

Use the common `role_probe_training.on_policy_training.train` module and
`src/experiments/training_against_role_probes/on_policy_execution/train.sbatch`
for new training. Set `sbatch --gpus=1` or `--gpus=2`; the launcher derives
torchrun ranks from `SLURM_GPUS_ON_NODE`. Effective episode batch size is
independent of GPU count. Do not create per-GPU-count copies of the trainer.
Historical `on_policy_recovery/train.py` remains frozen for the existing 2700
jobs and checkpoint source verification. Changing checkpoint world size requires
an explicit migration plus redistribution of audited recovery data, not simply
weakening provenance guards.

User authorized replacement of only the two 270 runs: 1816749 (batch64) and
1816751 (batch16), each one GPU/eight cores/256 GiB/24h. Evaluations reconnected.
User clarified that the other pending jobs should be raised slightly: lowered
existing positive Nice values by 100 for 50 pending groups, never below zero.
This is a partial removal of old penalties, not a new penalty. New jobs use
normal Nice=0. Recorded in on_policy/priority_adjustment_20261101.json.

Priority clarification (2026-11-01): user wants the two 270 runs ahead, without
penalizing other jobs against other users. Negative Nice is denied by Slurm for
this account (attempt Nice=-1 was rejected). Both 270 jobs 1816749/1816751 started
before the correction, so their allocations already satisfy the immediate ordering.
Reset all 50 remaining positively-niced pending groups to Nice=0; verified all
user jobs now have Nice=0. The earlier 100-point adjustment is superseded by
on_policy/priority_reset_20261101.json. Do not claim user-only queue ordering is
available through Nice; it changes priority against everyone.


### One-rollout epoch comparisons (2026-11-04)

The paired [on-policy](../experiment_reports/training_against_role_probes/on_policy_epochs/README.md)
and [off-policy](../experiment_reports/training_against_role_probes/off_policy_epochs/README.md)
experiments use ten epochs, 90 AppWorld tasks once per epoch and batch size 16.
Batch-refresh generation uses the current policy before every update;
epoch-refresh generation collects all 90 episodes before any epoch update and
consumes that saved pool once. Retail evaluation includes matched BF16 epoch 0
and every epoch end, with fresh standard-data probes and whole-history replay.

Submission uses `epoch_refresh/submit.py` from the frozen Git worktree
`.cache/epoch_refresh_source_20261104` (source commit `591c718`). `PYTHONPATH`
points to its frozen `src`; prepared environments, model/data cache and report
storage are shared links. Do not edit the snapshot while jobs are queued or active.
Epoch checkpoints are hard-linked before rolling recovery checkpoints are pruned.
Each successful retail evaluation schedules scoped GitHub publication of small
results, probe weights, tables and appendix matrices. Explicit `!checkpoints/`
rules in each report override the parent ignore so these small results stay in Git.

Initial training jobs: on-policy `2020749`, off-policy `2020781`; exact dependency
DAGs and commands are in each report's `jobs.json`. At submission both were pending
scheduler Priority, with no start estimate. CPU schedule/report checks and training
imports passed; production GPU execution and the measurements are still pending.
