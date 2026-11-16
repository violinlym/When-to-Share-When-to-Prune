"""Summarize unconstrained tau=.8 A1 anchor occupancy versus fixed 75%."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from measure_tau80 import HERE, TAU, FIXED, BYTES_PER_ROOT
from run_profile import SEED


DATASETS=('gsm8k','math500','mbpp_plus')


def ci(values,rng,draws=10000):
    x=np.asarray(values,dtype=float)
    sampled=x[rng.integers(0,len(x),(draws,len(x)))].mean(1)
    return float(np.quantile(sampled,.025)),float(np.quantile(sampled,.975))


def main():
    root=HERE/'results'/'tau80_anchors'
    files=sorted(root.glob('[0-9][0-9][0-9].json'))
    if len(files)!=48 or [int(p.stem) for p in files]!=list(range(48)):
        raise SystemExit(f'Expected 48 completed samples, found {len(files)}')
    rows=[]
    for file in files:
        group=json.loads(file.read_text())
        if len(group)!=24 or [x['layer'] for x in group]!=list(range(1,25)):
            raise ValueError(f'Invalid layer group {file}')
        rows.extend(group)
    if any(x['adaptive_mass_coverage'] < TAU-1e-9 for x in rows):
        raise ValueError('Attention coverage threshold violated')
    with (HERE/'tau80_per_sample.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    rng=np.random.default_rng(SEED)
    layerwise=[]
    for layer in range(1,25):
        group=[x for x in rows if x['layer']==layer]
        ratios=np.array([x['adaptive_ratio'] for x in group])
        low,high=ci(ratios,rng)
        layerwise.append(dict(layer=layer,mean_adaptive_ratio=float(ratios.mean()),
           ci95_ratio_low=low,ci95_ratio_high=high,
           mean_fixed_ratio=float(np.mean([x['fixed_ratio'] for x in group])),
           mean_adaptive_mass_coverage=float(np.mean([x['adaptive_mass_coverage'] for x in group])),
           mean_fixed_mass_coverage=float(np.mean([x['fixed_mass_coverage'] for x in group])),
           mean_adaptive_tail32_coverage=float(np.mean([x['adaptive_tail32_coverage'] for x in group])),
           mean_fixed_tail32_coverage=float(np.mean([x['fixed_tail32_coverage'] for x in group])),
           fraction_samples_above_fixed=float(np.mean([x['adaptive_roots']>x['fixed_roots'] for x in group])),
           pooled_adaptive_roots=sum(x['adaptive_roots'] for x in group),
           pooled_fixed_roots=sum(x['fixed_roots'] for x in group)))
    with (HERE/'tau80_layerwise.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=layerwise[0].keys(),lineterminator='\n');w.writeheader();w.writerows(layerwise)

    x=np.arange(1,25)
    ar=np.array([r['mean_adaptive_ratio'] for r in layerwise])
    ac=np.array([r['mean_adaptive_mass_coverage'] for r in layerwise])
    fc=np.array([r['mean_fixed_mass_coverage'] for r in layerwise])
    fig,(ax,ax2)=plt.subplots(2,1,sharex=True,figsize=(13,7))
    ax.bar(x,ar,color='#356d9a',label='τ=0.8 adaptive')
    ax.errorbar(x,ar,yerr=[ar-np.array([r['ci95_ratio_low'] for r in layerwise]),
                         np.array([r['ci95_ratio_high'] for r in layerwise])-ar],fmt='none',color='black',capsize=2,lw=.7)
    ax.axhline(FIXED,color='#b14f46',ls='--',label='Fixed 75% token retention')
    ax.set(ylabel='Anchor ratio',ylim=(0,1),title='Unconstrained τ=0.8 attention-mass anchoring')
    ax.legend(loc='lower right')
    ax2.plot(x,ac,'o-',label='τ=0.8 adaptive')
    ax2.plot(x,fc,'s--',label='Fixed 75% token retention')
    ax2.axhline(TAU,color='gray',ls=':',label='Attention mass threshold')
    ax2.set(xlabel='Physical Layer (1–24)',ylabel='Historical attention mass coverage',ylim=(0,1.02))
    ax2.set_xticks(x);ax2.legend(loc='lower right')
    fig.tight_layout();fig.savefig(HERE/'tau80_layerwise.png',dpi=200);plt.close(fig)

    total_adaptive=sum(x['adaptive_roots'] for x in rows)
    total_fixed=sum(x['fixed_roots'] for x in rows)
    total_full=sum(x['full_kv_bytes'] for x in rows)
    pooled_adaptive=total_adaptive*BYTES_PER_ROOT/total_full
    pooled_fixed=total_fixed*BYTES_PER_ROOT/total_full
    above=sum(x['adaptive_roots']>x['fixed_roots'] for x in rows)
    below=sum(x['adaptive_roots']<x['fixed_roots'] for x in rows)
    max_layer=max(layerwise,key=lambda x:x['mean_adaptive_ratio'])
    min_layer=min(layerwise,key=lambda x:x['mean_adaptive_ratio'])
    prior=list(csv.DictReader((HERE/'attention_dispersion_layerwise.csv').open()))
    entropy=np.array([float(x['global_entropy']) for x in prior])
    dispersion_rho=float(spearmanr(entropy,ar).statistic)
    lines=['# Fixed τ=0.8 adaptive anchor occupancy', '',
       '## Protocol', '',
       '- Ouro-1.4B, four loops × 24 physical layers, native BF16 KV; the same fixed 48 prompts (16 each GSM8K, MATH500, MBPP+) and seed 20261116 as the prior profiling.',
       '- Full KV prefill only. Attention peak over all loops, heads and prompt queries gives the **same token ranking** for both policies. Recent `ceil(20% × N)` tokens are always protected.',
       '- Uniform policy keeps `floor(0.75 × N)` tokens per layer, matching the `anchor75` A1 root-count rule.',
       '- Adaptive policy starts with the recent window, then adds eligible tokens in descending peak order until their *aggregate historical prefill attention mass* reaches at least 0.8 for that layer. The mass is averaged over all loops, heads and prefill queries. A last-32-query mass coverage is reported as a causal-position diagnostic.',
       '- No global budget is imposed in this first measurement. A2, decode, Output KL and task accuracy are not run. The root-only KV payload assumes one BF16 K and one BF16 V slot per retained token and physical layer, shared by four loops; each root costs 8192 bytes. Full KV has four K/V versions.',
       '', '## Global occupancy', '',
       f'- Uniform fixed-75 roots: **{total_fixed:,}**; root payload **{total_fixed*BYTES_PER_ROOT:,} bytes**, **{pooled_fixed:.2%}** of Full KV.',
       f'- Unconstrained τ=0.8 roots: **{total_adaptive:,}**; root payload **{total_adaptive*BYTES_PER_ROOT:,} bytes**, **{pooled_adaptive:.2%}** of Full KV.',
       f'- Relative to the fixed-75 root budget: **{total_adaptive/total_fixed:.2%}** of anchors, a difference of **{total_adaptive-total_fixed:+,}** roots across the 48 prompts.',
       f'- Under a hypothetical **25% of Full KV** total payload cap, root occupancy alone would leave **{.25-pooled_fixed:.2%} of Full KV** for A2 after fixed-75 roots, versus **{.25-pooled_adaptive:.2%}** after τ=0.8 roots. A2 was not run, so these are capacities rather than measured A2 payload.',
       f'- Among 1152 layer–sample pairs, τ=0.8 uses more roots than fixed-75 in **{above}**, fewer in **{below}**, and the same in **{1152-above-below}**.',
       f'- Every adaptive layer–sample pair meets τ=0.8. Its mean historical attention-mass coverage is **{np.mean([x["adaptive_mass_coverage"] for x in rows]):.2%}**; the uniform fixed-75 mean is **{np.mean([x["fixed_mass_coverage"] for x in rows]):.2%}**. These are **different budgets**, so they do not establish which policy is better under equal KV capacity.',
       '', '## Layer-wise allocation', '',
       f'- Mean adaptive ratio spans **{min_layer["mean_adaptive_ratio"]:.2%}** (L{min_layer["layer"]}) to **{max_layer["mean_adaptive_ratio"]:.2%}** (L{max_layer["layer"]}); uniform retention is approximately 75% in every layer.',
       f'- Across layer means, attention entropy versus required τ=0.8 root ratio has Spearman rho **{dispersion_rho:.3f}**.',
       '', '| Layer | Adaptive roots / tokens | 95% bootstrap CI | Fixed roots / tokens | Adaptive mass coverage | Fixed mass coverage |',
       '|---:|---:|---:|---:|---:|---:|']
    for r in layerwise:
        lines.append(f'| {r["layer"]} | {r["mean_adaptive_ratio"]:.2%} | [{r["ci95_ratio_low"]:.2%}, {r["ci95_ratio_high"]:.2%}] | {r["mean_fixed_ratio"]:.2%} | {r["mean_adaptive_mass_coverage"]:.2%} | {r["mean_fixed_mass_coverage"]:.2%} |')
    lines += ['', '## Dataset occupancy', '',
              '| Dataset | τ=0.8 roots / fixed-75 roots | τ=0.8 root payload / Full KV | Fixed root payload / Full KV |',
              '|---|---:|---:|---:|']
    for task in ('gsm8k','math500','mbpp_plus'):
        subset=[x for x in rows if x['dataset']==task]
        aa=sum(x['adaptive_roots'] for x in subset)
        ff=sum(x['fixed_roots'] for x in subset)
        full=sum(x['full_kv_bytes'] for x in subset)
        lines.append(f'| {task} | {aa/ff:.2%} | {aa*BYTES_PER_ROOT/full:.2%} | {ff*BYTES_PER_ROOT/full:.2%} |')
    lines += ['', '## Budget interpretation', '',
              'The 80% threshold is a **per-layer minimum mass coverage**, not a fixed global KV budget. '
              'If its total root count exceeds the fixed-75 budget, the threshold and budget cannot both be satisfied without relaxing a layer threshold. '
              'If it falls below the budget, the unused root slots must be reassigned before any equal-budget comparison. '
              'A2 would then receive only the remaining KV slots. This experiment measures neither A2 occupancy nor output quality.',
              '', 'See `tau80_per_sample.csv`, `tau80_layerwise.csv`, and `tau80_layerwise.png`.']
    (HERE/'TAU80_OCCUPANCY_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'roots adaptive {total_adaptive}, fixed {total_fixed}, payload/full {pooled_adaptive:.3%} vs {pooled_fixed:.3%}')


if __name__=='__main__':
    main()
