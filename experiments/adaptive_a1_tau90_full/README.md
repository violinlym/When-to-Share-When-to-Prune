# Adaptive A1 (τ=0.9) + original A2, full GSM8K/MATH500

This experiment is isolated from `../anchor_route_kvonly/anchor75/`.
`base_cache.py`, `sparse_cache.py`, and `route_gain.py` are byte-for-byte
copies from baseline commit `950b76a`. `adaptive_cache.py` overrides the A1
Root choice and retains the original A2 allocation, K/V masks, and rightward
routing. The same Full KV prefill is used for paired fixed/adaptive generation.

The frozen manifest contains 1,319 GSM8K test prompts and 500 MATH500
prompts from the existing local full benchmark, including prompt text, token
IDs, gold answers and generation parameters. Its SHA256 is
`02fe555ba90a4fe2caeef35848fc063293140400c8735ddd81863e77cb095c40`.
The source dataset revision was not embedded in that legacy manifest; this
hash identifies the exact frozen evaluation inputs.

Only K/V payload is capped at 25% Full KV. Metadata is measured separately.
The original A2 chooses positive marginal route gains and may leave capacity
unused; actual payload is recorded per sample and per generation stage.
During decode, A1 historical attention mass gives each prefill query and each
observed decode query equal weight. Deleted tokens keep their past mass for
coverage accounting but cannot be restored. Any coverage failure is reported.

## Commands

Use the Ouro GPU Python environment and a local Ouro-1.4B checkpoint. The
default model path in the scripts can be overridden with `--model` in
`run_full.py`.

```bash
python -m unittest -v test_adaptive.py test_scoring.py
CUDA_VISIBLE_DEVICES=4 python run_full.py --smoke
for w in $(seq 0 15); do gpu=$((4 + w % 4)); CUDA_VISIBLE_DEVICES=$gpu python run_full.py --worker $w --workers 16 & done; wait
for w in $(seq 0 3); do gpu=$((4 + w)); CUDA_VISIBLE_DEVICES=$gpu python run_full_reference.py --worker $w --workers 4 & done; wait
python score_full.py
python make_report.py
```

Each sample/method result is written atomically to `results/formal/` and
skipped on restart. The 24 Full KV reference items whose historical manifest
hash differs are regenerated locally; exact-matching historical Full KV
records are reused. The scorer refuses incomplete full results.
