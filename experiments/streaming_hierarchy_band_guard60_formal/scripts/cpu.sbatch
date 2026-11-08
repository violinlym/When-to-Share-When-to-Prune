#!/bin/bash
#SBATCH --account=project_462001687
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=bg60-fullscore
#SBATCH --output=logs/score-%j.out
#SBATCH --error=logs/score-%j.err
set -euo pipefail
cd '/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype/iterations/streaming_hierarchy_band_guard60_formal'
bash scripts/runtime.sh implementations/score.py prepare
bash scripts/score_code.sh mbpp_plus
bash scripts/score_code.sh humaneval
bash scripts/runtime.sh implementations/score.py finalize
