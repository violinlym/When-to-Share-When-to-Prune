#!/bin/bash
set -euo pipefail
root='/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype'
venv='/scratch/project_462001687/yiminliu/ouro_motivation_accuracy/venv'
extra='/scratch/project_462001687/yiminliu/dloop/env/huginn-evalplus-py312/lib/python3.12/site-packages'
work='/flash/project_462001687/yiminliu/coverkv_hierarchy_band_guard60_dev64_20261106/scoring'
singularity exec --cleanenv --containall --net --network none --no-home --no-mount hostfs,bind-paths,cwd \
 --bind "$work:/work" --bind '/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype/iterations/streaming_hierarchy_band_guard60_dev64/implementations:/scorer:ro' \
 --bind "$venv:$venv:ro" --bind "$extra:$extra:ro" \
 --env "PYTHONPATH=$extra,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,CUDA_VISIBLE_DEVICES=,HIP_VISIBLE_DEVICES=,ROCR_VISIBLE_DEVICES=,PYTHONDONTWRITEBYTECODE=1" \
 /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
 "$venv/bin/python" /scorer/code_only.py "$@"
