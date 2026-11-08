#!/bin/bash
#SBATCH --account=project_462001687
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=tree-band60-score
#SBATCH --output=logs/score-%j.out
#SBATCH --error=logs/score-%j.err
set -euo pipefail
cd '/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype/iterations/streaming_hierarchy_band_guard60_dev64'
bash scripts/runtime.sh implementations/score_dev64.py prepare
bash scripts/score_code.sh
bash scripts/runtime.sh implementations/score_dev64.py finalize
