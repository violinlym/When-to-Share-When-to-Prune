#!/bin/bash
set -euo pipefail
root='/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype'
venv='/scratch/project_462001687/yiminliu/ouro_motivation_accuracy/venv'
extra='/scratch/project_462001687/yiminliu/dloop/env/huginn-evalplus-py312/lib/python3.12/site-packages'
task=$1
benchmark_store=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["store"])' '/pfs/lustrep3/users/yiminliu/Project/kv cache management for loop transformer/motivation/gpt/algorithm_prototype/iterations/streaming_hierarchy_band_guard60_formal/configs/protocol.json')
work="$benchmark_store/scoring/$task"
if [[ "$task" == humaneval ]]; then
 cache='/flash/project_462001687/yiminliu/context_preserving_full_humaneval_six_20261031/scoring/full/cache/evalplus'
elif [[ "$task" == mbpp_plus ]]; then
 cache='/flash/project_462001687/yiminliu/context_preserving_full_math_mbpp_six_20261031/scoring/full/cache/evalplus'
else
 exit 2
fi
singularity exec --cleanenv --containall --net --network none --no-home --no-mount hostfs,bind-paths,cwd \
 --bind "$work:/work" --bind "$root/implementations:/scorer:ro" --bind "$cache:/work/cache/evalplus:ro" \
 --bind "$venv:$venv:ro" --bind "$extra:$extra:ro" \
 --env "XDG_CACHE_HOME=/work/cache,PYTHONPATH=$extra,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,CUDA_VISIBLE_DEVICES=,HIP_VISIBLE_DEVICES=,ROCR_VISIBLE_DEVICES=,PYTHONDONTWRITEBYTECODE=1,SCORE_CPUS=${SLURM_CPUS_PER_TASK:-64}" \
 /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
 "$venv/bin/python" /scorer/score_parallel.py "$task"
