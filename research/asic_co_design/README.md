# ASIC-oriented KV hierarchy co-design — exploratory branch

This directory adds CPU specifications and a research plan. It does **not** replace
the frozen 60%-compression prototype or claim measured ASIC gains.

- [DESIGN.md](DESIGN.md): motivation, algorithm/dataflow co-design, hardware costs,
  sparse attention interaction, experiments and falsification criteria (Chinese).
- `selector.py`: exact threshold selection for a **new integer score**, with
  certified mass intervals and stable hierarchy-preserving ties.
- `scoring.py`: sampled integer L1 + dyadic complete-linkage experimental metric.
- `analyze_predictability.py`: summary of existing real dense attention traces.
- `resource_model.py`: explicit storage calculations, not an ASIC simulator.
- `results/`: CPU parity evidence, dense profiling summary, storage scenarios.
- `evidence/`: selected historical Experiment 8 results, clearly marked synthetic.

Run from this directory with Python 3.11, no model/GPU/PyTorch required:

```sh
python3.11 -m unittest -v test_models.py
python3.11 resource_model.py
python3.11 analyze_predictability.py /path/to/per_prompt_predictions.csv
```

The selector receives valid mass bounds from its caller; it does not produce them
for free. Every threshold pass and certificate read is counted. Its current
binary-search implementation can be expensive and is a correctness baseline,
not the proposed final microarchitecture. Root exchange, full model integration,
real trace selector parity, quantization quality, sparse execution, cycle modeling,
RTL, synthesis and power analysis remain unimplemented/unvalidated here.

The sorted oracle exists only in tests. No hidden BF16 KV backup is introduced.
New scores can differ from the frozen floating-point allocator; integer-oracle
agreement is not evidence of unchanged model accuracy. No new GPU jobs submitted.
