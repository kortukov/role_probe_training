#!/bin/bash
#SBATCH --job-name=role-swe-netns-control
#SBATCH --account=cpu
#SBATCH --partition=ber-amd1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=experiment_reports/training_against_role_probes/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# The first operation creates a new user AND network namespace. All following
# interface/route operations occur only inside that namespace, never on the host.
unshare --user --map-root-user --net bash -eu -c '
 /usr/sbin/ip link set lo up
 /usr/sbin/ip link add swe-dummy type dummy
 /usr/sbin/ip addr add 10.255.255.2/24 dev swe-dummy
 /usr/sbin/ip link set swe-dummy up
 /usr/sbin/ip addr
 /usr/sbin/ip route
 .tools/bin/uv run --frozen --no-sync python -c '\''import socket,time; s=socket.socket(); s.settimeout(0.2); start=time.monotonic()
try: s.connect(("10.255.255.1",80)); raise RuntimeError("Unexpected connection")
except socket.timeout: print("PASS: isolated connection timeout",time.monotonic()-start)'\''
'
