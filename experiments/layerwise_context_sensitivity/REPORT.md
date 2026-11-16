# Layer-wise Context Loss Sensitivity Profiling

## Protocol

- Model: Ouro-1.4B, native BF16 KV, four loops × 24 physical layers; full-KV prefill for each sample.
- Data: first 16 fixed samples from each of GSM8K, MATH500 and MBPP+ in the existing `anchor_route_kvonly/manifest64.json`; `48` paired samples total.
- Manifest SHA256: `28dc76b5f8bf82c9a48f51b8a799ac5b0e3d18ad167c93a830173015ede43343`. Seed: `20261116`.
- Source manifest SHA256: `f86a62c59efc43de86b00a3b53e56867f5a96c9e62af3beb65a43a3a02002141`.
- Ouro BF16 weight SHA256: `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`; model code SHA256: `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`.
- Per target layer, peak prefill attention across all four loops, all heads and all prompt query positions ranks candidate tokens; the lowest 25% of original prompt length outside the protected recent 20% are masked for every loop and head.
- Full prefill is reused across variants. The full cache tensors remain at original positions. An additive attention mask makes selected historical tokens inaccessible during decode; no KV sharing or other compression runs.
- Full KV greedy continuation, up to 32 tokens, is frozen per sample. Each variant is teacher-forced on this same sequence. The prefill last-token prediction is excluded; KL is evaluated on logits after each continuation token is fed.
- FP32 log-softmax computes KL(Full || Pruned). Per-sample KL averages valid continuation predictions; layer means average samples equally. 95% percentile bootstrap CIs resample paired samples 10,000 times.
- CPU unit tests: 4 passed. GPU 25% smoke passed all runtime mask and loop assertions.
- Zero-pruning GPU smoke: layers 1, 13, 24; maximum KL `0`.

## Results

- Global paired layer-heterogeneity permutation test: variance of layer means `1.0441785e-05`; p = `0.00019996` (5,000 within-sample label permutations).
- Exploratory robustness test excluding layer 1: paired layer-heterogeneity p = `0.00019996`.
- Most sensitive: layer 1 (mean KL `0.0166816`); least sensitive: layer 19 (mean KL `0.000298092`).
- Paired top–bottom KL difference: `0.0163835`, 95% bootstrap CI [`0.0110957`, `0.0225896`]. This top–bottom pair was chosen after inspecting the means, so its CI is descriptive rather than selection-adjusted.
- Bootstrap probability of retaining these extreme ranks: top layer 1, `100.0%`; bottom layer 19, `81.5%`.
- Mean actual pruned fraction: `25.2310%` (rounding per prompt).
- Valid post-pruning prediction positions per sample: `32`–`32`. Actual per-sample pruned fraction: `24.32%`–`26.67%`.
- Across-layer correlation of mean deleted historical attention mass with mean KL: `0.873`. Layer 1 loses `51.0%` of its historical attention mass despite deleting about 25% of tokens; per-layer token sets differ, so this protocol combines layer response with selection quality.

### Sensitivity ranking (descending mean KL)

L1, L10, L8, L13, L11, L18, L15, L2, L20, L21, L16, L14, L24, L23, L9, L17, L22, L12, L7, L3, L4, L6, L5, L19

### Cross-dataset ranking consistency (Spearman rho)

| Pair | rho |
|---|---:|
| gsm8k vs math500 | 0.519 |
| gsm8k vs mbpp_plus | 0.568 |
| math500 vs mbpp_plus | 0.528 |

### Per-dataset extremes and paired heterogeneity

| Dataset | Highest-KL layer | Lowest-KL layer | Layer 1 rank | Permutation p |
|---|---:|---:|---:|---:|
| gsm8k | 13 | 19 | 15 | 0.00019996 |
| math500 | 1 | 3 | 1 | 0.00019996 |
| mbpp_plus | 1 | 19 | 1 | 0.00019996 |

**Interpretation:** The paired global test supports unequal mean KL across layers, and the result remains after excluding layer 1. Rank correlations across datasets are moderate rather than identical. Layer 1 is the overall maximum but its GSM8K rank is 15; therefore the exact most-sensitive layer is task dependent. The descriptive top–bottom paired CI excludes zero. Different token choices and deleted attention mass prevent attributing all of the observed KL spread to intrinsic layer sensitivity alone.

See `per_sample.csv`, `layerwise_summary.csv`, `layerwise_sensitivity.png` and `layerwise_heatmap.png` for all estimates. Rankings are descriptive; the 16 samples per dataset limit precision of dataset-specific ranks.
