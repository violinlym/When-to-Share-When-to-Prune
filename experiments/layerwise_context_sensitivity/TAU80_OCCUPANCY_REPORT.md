# Fixed τ=0.8 adaptive anchor occupancy

## Protocol

- Ouro-1.4B, four loops × 24 physical layers, native BF16 KV; the same fixed 48 prompts (16 each GSM8K, MATH500, MBPP+) and seed 20261116 as the prior profiling.
- Full KV prefill only. Attention peak over all loops, heads and prompt queries gives the **same token ranking** for both policies. Recent `ceil(20% × N)` tokens are always protected.
- Uniform policy keeps `floor(0.75 × N)` tokens per layer, matching the `anchor75` A1 root-count rule.
- Adaptive policy starts with the recent window, then adds eligible tokens in descending peak order until their *aggregate historical prefill attention mass* reaches at least 0.8 for that layer. The mass is averaged over all loops, heads and prefill queries. A last-32-query mass coverage is reported as a causal-position diagnostic.
- No global budget is imposed in this first measurement. A2, decode, Output KL and task accuracy are not run. The root-only KV payload assumes one BF16 K and one BF16 V slot per retained token and physical layer, shared by four loops; each root costs 8192 bytes. Full KV has four K/V versions.

## Global occupancy

- Uniform fixed-75 roots: **146,280**; root payload **1,198,325,760 bytes**, **18.69%** of Full KV.
- Unconstrained τ=0.8 roots: **85,207**; root payload **698,015,744 bytes**, **10.89%** of Full KV.
- Relative to the fixed-75 root budget: **58.25%** of anchors, a difference of **-61,073** roots across the 48 prompts.
- Under a hypothetical **25% of Full KV** total payload cap, root occupancy alone would leave **6.31% of Full KV** for A2 after fixed-75 roots, versus **14.11%** after τ=0.8 roots. A2 was not run, so these are capacities rather than measured A2 payload.
- Among 1152 layer–sample pairs, τ=0.8 uses more roots than fixed-75 in **32**, fewer in **1118**, and the same in **2**.
- Every adaptive layer–sample pair meets τ=0.8. Its mean historical attention-mass coverage is **81.38%**; the uniform fixed-75 mean is **92.59%**. These are **different budgets**, so they do not establish which policy is better under equal KV capacity.

## Layer-wise allocation

- Mean adaptive ratio spans **23.04%** (L19) to **76.83%** (L2); uniform retention is approximately 75% in every layer.
- Across layer means, attention entropy versus required τ=0.8 root ratio has Spearman rho **0.994**.

| Layer | Adaptive roots / tokens | 95% bootstrap CI | Fixed roots / tokens | Adaptive mass coverage | Fixed mass coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 53.89% | [53.13%, 54.65%] | 74.51% | 80.29% | 89.44% |
| 2 | 76.83% | [73.49%, 80.03%] | 74.51% | 81.84% | 79.65% |
| 3 | 52.81% | [51.50%, 54.28%] | 74.51% | 80.77% | 89.85% |
| 4 | 56.66% | [55.33%, 58.01%] | 74.51% | 80.45% | 88.43% |
| 5 | 32.92% | [31.18%, 34.93%] | 74.51% | 80.49% | 93.40% |
| 6 | 36.65% | [34.98%, 38.29%] | 74.51% | 80.29% | 94.48% |
| 7 | 58.11% | [56.84%, 59.45%] | 74.51% | 80.35% | 89.02% |
| 8 | 50.09% | [47.86%, 52.24%] | 74.51% | 80.46% | 92.60% |
| 9 | 58.15% | [56.55%, 59.73%] | 74.51% | 80.42% | 88.27% |
| 10 | 41.70% | [39.22%, 44.10%] | 74.51% | 80.31% | 93.92% |
| 11 | 51.23% | [48.99%, 53.32%] | 74.51% | 80.31% | 90.24% |
| 12 | 45.80% | [44.03%, 47.51%] | 74.51% | 80.34% | 92.17% |
| 13 | 39.84% | [37.41%, 42.16%] | 74.51% | 80.40% | 93.98% |
| 14 | 48.20% | [46.53%, 49.76%] | 74.51% | 80.40% | 92.66% |
| 15 | 44.60% | [42.70%, 46.50%] | 74.51% | 80.36% | 92.54% |
| 16 | 42.43% | [41.01%, 43.82%] | 74.51% | 80.48% | 93.21% |
| 17 | 30.57% | [29.31%, 31.86%] | 74.51% | 80.43% | 95.12% |
| 18 | 28.03% | [27.00%, 29.03%] | 74.51% | 81.39% | 96.15% |
| 19 | 23.04% | [22.46%, 23.65%] | 74.51% | 84.04% | 96.80% |
| 20 | 26.21% | [25.31%, 27.11%] | 74.51% | 83.13% | 97.01% |
| 21 | 23.35% | [22.69%, 24.06%] | 74.51% | 84.61% | 96.02% |
| 22 | 23.07% | [22.44%, 23.71%] | 74.51% | 86.72% | 97.49% |
| 23 | 25.53% | [24.80%, 26.30%] | 74.51% | 83.02% | 95.79% |
| 24 | 36.14% | [34.46%, 37.87%] | 74.51% | 81.90% | 93.99% |

## Dataset occupancy

| Dataset | τ=0.8 roots / fixed-75 roots | τ=0.8 root payload / Full KV | Fixed root payload / Full KV |
|---|---:|---:|---:|
| gsm8k | 60.61% | 11.35% | 18.72% |
| math500 | 51.69% | 9.65% | 18.66% |
| mbpp_plus | 58.42% | 10.85% | 18.58% |

## Budget interpretation

The 80% threshold is a **per-layer minimum mass coverage**, not a fixed global KV budget. If its total root count exceeds the fixed-75 budget, the threshold and budget cannot both be satisfied without relaxing a layer threshold. If it falls below the budget, the unused root slots must be reassigned before any equal-budget comparison. A2 would then receive only the remaining KV slots. This experiment measures neither A2 occupancy nor output quality.

See `tau80_per_sample.csv`, `tau80_layerwise.csv`, and `tau80_layerwise.png`.
