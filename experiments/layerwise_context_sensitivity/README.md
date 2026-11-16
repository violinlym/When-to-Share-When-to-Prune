# Layer-wise Context Loss Sensitivity Profiling

This experiment tests whether Ouro-1.4B's 24 physical layers differ in their
sensitivity to the same 25% context deletion. It changes no A1/A2 algorithm.

The 48 fixed prompts are the first 16 examples from each GSM8K, MATH500, and
MBPP+ group in `../anchor_route_kvonly/manifest64.json`. The local manifest
retains only prompts and token IDs, with gold answers omitted. The model runs
four loops with native BF16 KV and eager attention.

For each prompt, a single Full KV prefill collects attention peak per token,
layer, loop, and head. The worst-ranked eligible tokens for each physical layer
are masked after prefill. The recent `ceil(20% × prompt_length)` tokens remain
accessible. The same selected indices are blocked in all four loops, all heads,
and both K/V uses of that target layer; every other layer remains Full KV.
Actual cache tensors are left full length so cache positions, RoPE indices, and
the initial prompt computation are identical. This is a **sensitivity test**,
not a memory-reduction benchmark.

Full KV greedily selects a continuation of at most 32 tokens. Every variant is
teacher-forced with those IDs. The final prefill logits are excluded; each
comparison is for a prediction after a continuation token has been fed.
`per_sample.csv` contains equal-weight per-sample mean KL, valid token count,
actual deletion ratio, removed prefill attention mass and selection checksum.

## Reproduce

Use the same Python environment as the Ouro experiments (PyTorch,
Transformers supporting `DynamicLayer`, NumPy, and Matplotlib), with a local
Ouro-1.4B checkpoint. Override `--model` if the default local path differs.

```bash
python -m unittest -v test_core.py
CUDA_VISIBLE_DEVICES=4 python run_profile.py --smoke --ratio 0 --output results_ratio0
CUDA_VISIBLE_DEVICES=4 python run_profile.py --smoke --output results
CUDA_VISIBLE_DEVICES=4 python run_profile.py --worker 0 --workers 4
CUDA_VISIBLE_DEVICES=5 python run_profile.py --worker 1 --workers 4
CUDA_VISIBLE_DEVICES=6 python run_profile.py --worker 2 --workers 4
CUDA_VISIBLE_DEVICES=7 python run_profile.py --worker 3 --workers 4
python analyze.py
```

Each sample is written atomically to `results/formal/NNN.json`; completed
samples are skipped on restart. Analysis requires all 48 paired samples.
The scripts never inspect gold answers or future attention when choosing
tokens. The random seed is 20261116.

## Attention dispersion follow-up

This follow-up reruns only Full KV prefill on the same 48 prompts. It
reconstructs each original deletion set, checks its checksum and removed
attention mass against `per_sample.csv`, then measures token-mass entropy and
effective attention coverage. It does not run a new A1 policy or decode.

```bash
CUDA_VISIBLE_DEVICES=4 python attention_dispersion.py --worker 0 --workers 4
CUDA_VISIBLE_DEVICES=5 python attention_dispersion.py --worker 1 --workers 4
CUDA_VISIBLE_DEVICES=6 python attention_dispersion.py --worker 2 --workers 4
CUDA_VISIBLE_DEVICES=7 python attention_dispersion.py --worker 3 --workers 4
python analyze_attention_dispersion.py
```

The resulting interpretation and limitations are in
`ATTENTION_DISPERSION_REPORT.md`.

## Fixed 80% attention-mass threshold occupancy

This measurement uses the same peak-attention token order in both policies.
Fixed A1 retains `floor(0.75 × N)` roots per layer. The adaptive policy keeps
the recent 20% and then takes high-ranked eligible tokens until that layer's
aggregate prefill attention mass reaches 80%. This stage measures A1 root
counts and BF16 K/V payload only; A2, KL and task accuracy are not run.

```bash
python -m unittest -v test_core.py
CUDA_VISIBLE_DEVICES=4 python measure_tau80.py --smoke
CUDA_VISIBLE_DEVICES=4 python measure_tau80.py --worker 0 --workers 4
CUDA_VISIBLE_DEVICES=5 python measure_tau80.py --worker 1 --workers 4
CUDA_VISIBLE_DEVICES=6 python measure_tau80.py --worker 2 --workers 4
CUDA_VISIBLE_DEVICES=7 python measure_tau80.py --worker 3 --workers 4
python analyze_tau80.py
```

See `TAU80_OCCUPANCY_REPORT.md` for the global budget comparison and the
24-layer allocation table.

For the 90% threshold measurement on the same prompts, pass `--tau 0.9` to
each `measure_tau80.py` invocation above, then run
`python analyze_tau80.py --tau 0.9`. This writes separate `tau90_*` outputs
and `TAU90_OCCUPANCY_REPORT.md`, preserving the 80% results.
