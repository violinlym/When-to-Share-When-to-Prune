# Anchor Route: KV-only Ouro prototypes

This directory contains the **new A1/A2/A3 algorithm prototypes**, separate from the earlier 60%-compression CoVerKV prototype and from the attention-pruning baselines. `anchor50/` targets 50% compression; `anchor75/` targets 75% compression. Both use native BF16 KV and charge the **KV payload** against the target. Metadata is measured and reported separately.

## Implemented algorithm

- **A1, root reservation:** Keep loop 4 as the root version of each live token. At 50% compression all tokens retain a root. At 75% compression, retain roots for approximately 75% of tokens per layer, selected by the maximum observed attention across the four loops, with the most recent 20% of tokens protected. Tokens without a root are masked in attention; their context is genuinely unavailable.
- **A2, asymmetric version competition:** K and V compete independently for extra slots across layers and tokens. For every possible directed replacement of one loop by a later loop, compute a relative squared KV difference and weight it by the token's mean observed attention. Starting from root-only masks, greedily add the available version with the largest positive reduction in the current rightward-route proxy loss. Recompute gains after each addition to the same group. This is a proxy-guided greedy allocator, not an exact global optimizer or a measured end-to-end loss. `sparse_cache.py` overrides the simpler allocation routine in `base_cache.py` and is the implementation loaded by `run_sparse.py`.
- **A3, logical routing:** For a live token and requested loop, use that loop's stored version if present; otherwise use its nearest stored version to the right. Compact K and V pools have separate slot masks and offsets. Deleted tokens return zero history and are excluded from attention by a mask. No version is reconstructed after it has been discarded.

The prototype recomputes its selection as decoding grows. Its current PyTorch implementation materializes routed history and uses eager attention; it demonstrates the algorithm and memory budget, **not** a hardware speedup or an optimized GPU kernel. It is deliberately nonuniform across K/V, layer, token, and loop.

## Run

The reference environment used PyTorch 2.4.0+cu121, Transformers 4.57.1, NumPy 1.26.4, and an Ouro-1.4B checkpoint compatible with `trust_remote_code=True`. CUDA is required for generation. From either `anchor50/` or `anchor75/`:

```bash
ln -s /absolute/path/to/Ouro-1.4B model
python test_sparse.py
python run_sparse.py --smoke
python run_sparse.py --worker 0 --workers 1
```

The shared `manifest64.json` has 64 sampled examples from each of GSM8K, MATH500, MBPP+, and HumanEval. The run script checks tokenization against this manifest. Results are written under the local `results/` directory and are ignored by Git. Use distinct `--worker` values for independent workers. The model link must point to the local checkpoint and is ignored by Git.

`SOURCE_HASHES.json` records the exact source snapshot copied from the tested local prototypes. The copied KV-only code passed CPU route and budget tests and one GPU smoke example per target. Earlier full-benchmark scores came from an older budget policy that charged metadata; they are **not** accuracy results for this KV-only snapshot. Run paired full benchmarks before making accuracy claims about it.
