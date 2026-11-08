#!/bin/bash
#SBATCH --account=project_462001687
#SBATCH --partition=dev-g
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=mi250:8
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --job-name=tree-band60-dev64
#SBATCH --output=logs/gpu-%j.out
#SBATCH --error=logs/gpu-%j.err
#SBATCH --no-requeue
set -euo pipefail
cd '/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype/iterations/streaming_hierarchy_band_guard60_dev64'

bash scripts/runtime.sh implementations/worker.py --check
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
srun --mem=400G --gpu-bind=none --gres-flags=allow-task-sharing --cpu-bind=cores --kill-on-bad-exit=1 --output='logs/worker-%j-%t.out' --error='logs/worker-%j-%t.err' bash scripts/runtime.sh implementations/worker.py
python3 implementations/completion.py
# The account submission-QOS is currently saturated by unrelated queued jobs.
# Score after generation in this same allocation so this experiment consumes a
# single submitted-job slot. Model worker processes have exited before grading.
bash scripts/runtime.sh implementations/score_dev64.py prepare
bash scripts/score_code.sh
bash scripts/runtime.sh implementations/score_dev64.py finalize
