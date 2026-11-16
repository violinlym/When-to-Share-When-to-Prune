# Attention dispersion and context loss

## Question and protocol

Does a layer with larger output KL under the same 25% token deletion have a more diffuse attention distribution? The same 48 prompts and native BF16 Ouro full-prefill protocol from `REPORT.md` were rerun on GPUs 4–7. Deletion masks were reconstructed from the original peak-attention rule and checked against every original selection checksum and deleted-mass value. No decode or A1 policy was changed.

A layer-level distribution is the attention mass averaged over four loops, all heads, and either all prefill queries or the last 32 prefill queries. Normalized entropy is H(p)/log(N); effective token fraction is exp(H(p))/N. Deleted mass is the attention probability on the original 25%-deleted token set; effective attention coverage is one minus deleted mass. These are descriptive prefill proxies for later decode attention, not direct measurements of future decode attention.

## Paired results

| Metric | Spearman rho across 24 layer means | 95% paired-bootstrap CI | rho without L1 | Median within-sample rho |
|---|---:|---:|---:|---:|
| global_normalized_entropy | 0.148 | [0.043, 0.303] | 0.070 | 0.154 |
| tail32_normalized_entropy | 0.223 | [0.097, 0.356] | 0.131 | 0.187 |
| global_deleted_mass | 0.113 | [-0.008, 0.235] | -0.008 | 0.087 |
| tail32_deleted_mass | 0.157 | [0.030, 0.291] | 0.042 | 0.160 |

- All layers delete about 25% of tokens, but mean removed historical attention mass spans **2.5%–51.0%** across layers; last-32-query removed mass spans **1.8%–43.4%**.
- Across layer means, entropy tracks removed attention mass strongly (Spearman rho `0.966` for all-query attention; `0.959` for last-32-query attention).
- Removed historical mass vs KL has Pearson r `0.873` but falls to `-0.007` after excluding layer 1; Spearman and per-task correlations in the tables show that this apparent linear relation is dominated by layer 1.
- Layer 1: KL `0.0166816`, global entropy `0.651`, removed historical mass `51.0%`; layer 19: KL `0.000298092`, global entropy `0.365`, removed historical mass `3.2%`.

### Dataset-specific rank correlations

| Dataset | All-query entropy vs KL | Last-32 entropy vs KL | Deleted mass vs KL |
|---|---:|---:|---:|
| gsm8k | -0.036 | 0.037 | -0.163 |
| math500 | 0.149 | 0.205 | 0.064 |
| mbpp_plus | 0.225 | 0.211 | 0.212 |

## Interpretation

The same **token coverage ratio** produces different **attention-mass coverage**. This establishes the coverage mismatch proposed in the motivation under the fixed 25% mask protocol. Entropy predicts attention-mass loss, but its rank association with output KL is weak, gets weaker without layer 1, and is absent on GSM8K. Thus the stronger claim that high-KL layers are consistently more diffuse is **not established** by these data. The attention-peak selection rule, layer response, and positional/causal effects all influence KL. This one-budget profiling does **not** show that transferring anchor slots from concentrated layers to diffuse layers improves output quality. That allocation claim requires a controlled fixed-total-budget intervention at multiple pruning ratios.

See `attention_dispersion_per_sample.csv`, `attention_dispersion_layerwise.csv`, `attention_dispersion_correlations.csv`, and `attention_dispersion_vs_kl.png`.
