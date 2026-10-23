#!/bin/bash
#SBATCH --job-name=role-training
#SBATCH --account=gpu
#SBATCH --partition=ber-gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=vca-gpu-0701-03
#SBATCH --config=${1:?Pass a training configuration JSON}
shift
.tools/bin/uv run --project "$runtime" --frozen --no-sync torchrun --standalone --nproc-per-node=2 \
 --module role_probe_training.training.train --config "$config" "$@"
