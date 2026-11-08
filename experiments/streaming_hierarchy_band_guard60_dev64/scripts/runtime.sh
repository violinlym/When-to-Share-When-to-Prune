#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME='/flash/project_462001687/yiminliu/ouro_motivation_accuracy/hf'
export PYTHONPATH='/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype/iterations/streaming_hierarchy_band_guard60_dev64/implementations'
cd '/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype/iterations/streaming_hierarchy_band_guard60_dev64'
exec singularity exec --bind /pfs,/scratch,/project,/users,/flash \
 /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
 /scratch/project_462001687/yiminliu/ouro_motivation_accuracy/venv/bin/python "$@"
