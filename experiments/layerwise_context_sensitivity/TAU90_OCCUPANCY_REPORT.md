# Fixed τ=0.90 adaptive anchor occupancy

## Protocol

- Ouro-1.4B, four loops × 24 physical layers, native BF16 KV; the same fixed 48 prompts (16 each GSM8K, MATH500, MBPP+) and seed 20261116 as the prior profiling.
- Full KV prefill only. Attention peak over all loops, heads and prompt queries gives the **same token ranking** for both policies. Recent `ceil(20% × N)` tokens are always protected.
- Uniform policy keeps `floor(0.75 × N)` tokens per layer, matching the `anchor75` A1 root-count rule.
- Adaptive policy starts with the recent window, then adds eligible tokens in descending peak order until their *aggregate historical prefill attention mass* reaches at least 0.90 for that layer. The mass is averaged over all loops, heads and prefill queries. A last-32-query mass coverage is reported as a causal-position diagnostic.
- No global budget is imposed in this first measurement. A2, decode, Output KL and task accuracy are not run. The root-only KV payload assumes one BF16 K and one BF16 V slot per retained token and physical layer, shared by four loops; each root costs 8192 bytes. Full KV has four K/V versions.

## Global occupancy

- Uniform fixed-75 roots: **146,280**; root payload **1,198,325,760 bytes**, **18.69%** of Full KV.
- Unconstrained τ=0.90 roots: **126,890**; root payload **1,039,482,880 bytes**, **16.21%** of Full KV.
- Relative to the fixed-75 root budget: **86.74%** of anchors, a difference of **-19,390** roots across the 48 prompts.
- Relative to the earlier τ=0.80 run, τ=0.90 retains **+41,683** more roots; τ=0.80 used **85,207** roots on these prompts.
- Under a hypothetical **25% of Full KV** total payload cap, root occupancy alone would leave **6.31% of Full KV** for A2 after fixed-75 roots, versus **8.79%** after τ=0.90 roots. A2 was not run, so these are capacities rather than measured A2 payload.
- Among 1152 layer–sample pairs, τ=0.90 uses more roots than fixed-75 in **245**, fewer in **850**, and the same in **57**.
- Every adaptive layer–sample pair meets τ=0.90. Its mean historical attention-mass coverage is **90.52%**; the uniform fixed-75 mean is **92.59%**. These are **different budgets**, so they do not establish which policy is better under equal KV capacity.

## Layer-wise allocation

- Mean adaptive ratio spans **37.00%** (L22) to **86.65%** (L2); uniform retention is approximately 75% in every layer.
- Across layer means, attention entropy versus required τ=0.90 root ratio has Spearman rho **0.990**.

| Layer | Adaptive roots / tokens | 95% bootstrap CI | Fixed roots / tokens | Adaptive mass coverage | Fixed mass coverage |
|---:|---:|---:|---:|---:|---:|
| 1 | 76.53% | [75.98%, 77.11%] | 74.51% | 90.25% | 89.44% |
| 2 | 86.65% | [84.42%, 89.04%] | 74.51% | 93.46% | 79.65% |
| 3 | 75.31% | [74.60%, 76.20%] | 74.51% | 90.45% | 89.85% |
| 4 | 78.24% | [77.09%, 79.42%] | 74.51% | 90.39% | 88.43% |
| 5 | 63.49% | [61.85%, 65.36%] | 74.51% | 90.56% | 93.40% |
| 6 | 60.19% | [58.72%, 61.64%] | 74.51% | 90.38% | 94.48% |
| 7 | 77.01% | [76.04%, 78.07%] | 74.51% | 90.57% | 89.02% |
| 8 | 69.00% | [67.41%, 70.55%] | 74.51% | 90.40% | 92.60% |
| 9 | 78.35% | [76.86%, 79.81%] | 74.51% | 90.31% | 88.27% |
| 10 | 63.99% | [62.43%, 65.52%] | 74.51% | 90.60% | 93.92% |
| 11 | 74.03% | [72.50%, 75.43%] | 74.51% | 90.28% | 90.24% |
| 12 | 69.05% | [67.42%, 70.57%] | 74.51% | 90.33% | 92.17% |
| 13 | 62.05% | [60.28%, 63.71%] | 74.51% | 90.22% | 93.98% |
| 14 | 67.95% | [66.80%, 69.03%] | 74.51% | 90.29% | 92.66% |
| 15 | 67.75% | [66.18%, 69.30%] | 74.51% | 90.23% | 92.54% |
| 16 | 65.73% | [64.48%, 66.97%] | 74.51% | 90.29% | 93.21% |
| 17 | 57.54% | [55.96%, 59.10%] | 74.51% | 90.67% | 95.12% |
| 18 | 51.50% | [49.42%, 53.47%] | 74.51% | 90.62% | 96.15% |
| 19 | 45.38% | [43.70%, 47.00%] | 74.51% | 90.82% | 96.80% |
| 20 | 46.61% | [44.10%, 49.13%] | 74.51% | 90.21% | 97.01% |
| 21 | 42.92% | [40.33%, 45.39%] | 74.51% | 90.19% | 96.02% |
| 22 | 37.00% | [35.91%, 38.05%] | 74.51% | 90.67% | 97.49% |
| 23 | 49.74% | [48.61%, 50.82%] | 74.51% | 90.20% | 95.79% |
| 24 | 60.07% | [58.75%, 61.32%] | 74.51% | 90.20% | 93.99% |

## Dataset occupancy

| Dataset | τ=0.90 roots / fixed-75 roots | τ=0.90 root payload / Full KV | Fixed root payload / Full KV |
|---|---:|---:|---:|
| gsm8k | 88.65% | 16.60% | 18.72% |
| math500 | 80.69% | 15.06% | 18.66% |
| mbpp_plus | 88.54% | 16.45% | 18.58% |

## Budget interpretation

The 90% threshold is a **per-layer minimum mass coverage**, not a fixed global KV budget. If its total root count exceeds the fixed-75 budget, the threshold and budget cannot both be satisfied without relaxing a layer threshold. If it falls below the budget, the unused root slots must be reassigned before any equal-budget comparison. A2 would then receive only the remaining KV slots. This experiment measures neither A2 occupancy nor output quality.

See `tau90_per_sample.csv`, `tau90_layerwise.csv`, and `tau90_layerwise.png`.
