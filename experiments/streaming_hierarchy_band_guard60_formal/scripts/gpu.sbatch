#!/bin/bash
#SBATCH --account=project_462001687
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:8
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --job-name=bg60-fullbench
#SBATCH --output=logs/gpu-%j.out
#SBATCH --error=logs/gpu-%j.err
#SBATCH --no-requeue
set -euo pipefail
cd '/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype/iterations/streaming_hierarchy_band_guard60_formal'
bash scripts/runtime.sh implementations/worker.py --check
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
srun --nodes=4 --ntasks=32 --ntasks-per-node=8 --cpus-per-task=7 --gpus-per-task=mi250:1 \
  --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1 \
  --output='logs/worker-%j-%t.out' --error='logs/worker-%j-%t.err' \
  bash scripts/runtime.sh implementations/worker.py
python3 implementations/completion.py
