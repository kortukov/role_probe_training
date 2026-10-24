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

Before submitting, run `uv sync --locked` from the repo on appropriate storage.
Jobs can use `uv run --frozen --no-sync ...` against that prepared environment to
avoid downloads and concurrent environment changes. Do not change an environment
or code used by queued/running jobs; Slurm copies the submission script, not the
Python source it invokes. Use a separate checkout/environment for concurrent
experiments that need different revisions.

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
uv sync --locked
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
