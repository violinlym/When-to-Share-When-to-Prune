"""Test whether full-prefill attention dispersion explains paired pruning KL."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from run_profile import HERE, SEED


FIELDS=('global_normalized_entropy','global_effective_token_fraction','global_top25_mass',
        'global_deleted_mass','tail32_normalized_entropy','tail32_effective_token_fraction',
        'tail32_top25_mass','tail32_deleted_mass')
DATASETS=('gsm8k','math500','mbpp_plus')


def rho(x,y):
    return float(spearmanr(x,y).statistic)


def bootstrap_layer_rho(x,y,seed=SEED,draws=10000):
    """Resample paired prompts, then correlate the 24 layer means."""
    rng=np.random.default_rng(seed)
    boot=np.empty(draws)
    for b in range(draws):
        sample=rng.integers(0,len(x),len(x))
        boot[b]=rho(x[sample].mean(0),y[sample].mean(0))
    return float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def main():
    root=HERE/'results'/'attention_dispersion'
    files=sorted(root.glob('[0-9][0-9][0-9].json'))
    if len(files)!=48:
        raise SystemExit(f'Need 48 complete prefill profiles; found {len(files)}')
    rows=[]
    for index,path in enumerate(files):
        if int(path.stem)!=index:
            raise ValueError('Incomplete sample indices')
        group=json.loads(path.read_text())
        if len(group)!=24 or [x['layer'] for x in group]!=list(range(1,25)):
            raise ValueError(f'Incomplete layers: {path}')
        rows.extend(group)
    with (HERE/'attention_dispersion_per_sample.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n')
        w.writeheader();w.writerows(rows)
    k=np.array([[float(rows[24*i+l]['mean_kl']) for l in range(24)] for i in range(48)])
    values={name:np.array([[float(rows[24*i+l][name]) for l in range(24)] for i in range(48)]) for name in FIELDS}
    datasets=[rows[24*i]['dataset'] for i in range(48)]
    layer_mean_k=k.mean(0)
    result=[]
    for name in FIELDS:
        x=values[name]
        per_sample=np.array([rho(x[i],k[i]) for i in range(48)])
        layer_rho=rho(x.mean(0),layer_mean_k)
        low,high=bootstrap_layer_rho(x,k)
        result.append(dict(metric=name,layer_mean_spearman=layer_rho,ci95_low=low,ci95_high=high,
                           excluding_layer1_spearman=rho(x.mean(0)[1:],layer_mean_k[1:]),
                           median_within_sample_spearman=float(np.median(per_sample)),
                           positive_sample_fraction=float(np.mean(per_sample>0))))
    with (HERE/'attention_dispersion_correlations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=result[0].keys(),lineterminator='\n');w.writeheader();w.writerows(result)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric,title in zip(axes,('global_normalized_entropy','tail32_normalized_entropy'),
                               ('All prefill queries','Last 32 prefill queries')):
        x=values[metric].mean(0)
        ax.scatter(x,layer_mean_k,c=np.arange(1,25),cmap='viridis',s=42)
        for layer in (1,8,10,13,19):
            ax.annotate(str(layer),(x[layer-1],layer_mean_k[layer-1]),fontsize=8,xytext=(4,4),textcoords='offset points')
        ax.set(xlabel='Normalized attention entropy',ylabel='Mean output KL',title=title)
        ax.set_yscale('log')
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'attention_dispersion_vs_kl.png',dpi=200);plt.close(fig)

    global_mass=values['global_deleted_mass'].mean(0)
    tail_mass=values['tail32_deleted_mass'].mean(0)
    entropy=values['global_normalized_entropy'].mean(0)
    tail_entropy=values['tail32_normalized_entropy'].mean(0)
    entropy_mass_rho=rho(entropy,global_mass)
    tail_entropy_mass_rho=rho(tail_entropy,tail_mass)
    mass_kl_pearson=float(np.corrcoef(global_mass,layer_mean_k)[0,1])
    mass_kl_pearson_no_l1=float(np.corrcoef(global_mass[1:],layer_mean_k[1:])[0,1])
    summary=[]
    for l in range(24):
        summary.append(dict(layer=l+1,mean_kl=layer_mean_k[l],global_entropy=entropy[l],
                            tail32_entropy=tail_entropy[l],global_deleted_mass=global_mass[l],
                            tail32_deleted_mass=tail_mass[l]))
    with (HERE/'attention_dispersion_layerwise.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys(),lineterminator='\n');w.writeheader();w.writerows(summary)

    def entry(metric):
        return next(x for x in result if x['metric']==metric)
    lines=['# Attention dispersion and context loss', '',
           '## Question and protocol', '',
           'Does a layer with larger output KL under the same 25% token deletion have a more diffuse attention distribution? '
           'The same 48 prompts and native BF16 Ouro full-prefill protocol from `REPORT.md` were rerun on GPUs 4–7. '
           'Deletion masks were reconstructed from the original peak-attention rule and checked against every original selection checksum and deleted-mass value. '
           'No decode or A1 policy was changed.', '',
           'A layer-level distribution is the attention mass averaged over four loops, all heads, and either all prefill queries or the last 32 prefill queries. '
           'Normalized entropy is H(p)/log(N); effective token fraction is exp(H(p))/N. '
           'Deleted mass is the attention probability on the original 25%-deleted token set; effective attention coverage is one minus deleted mass. '
           'These are descriptive prefill proxies for later decode attention, not direct measurements of future decode attention.', '',
           '## Paired results', '',
           '| Metric | Spearman rho across 24 layer means | 95% paired-bootstrap CI | rho without L1 | Median within-sample rho |',
           '|---|---:|---:|---:|---:|']
    for name in ('global_normalized_entropy','tail32_normalized_entropy',
                 'global_deleted_mass','tail32_deleted_mass'):
        x=entry(name)
        lines.append(f'| {name} | {x["layer_mean_spearman"]:.3f} | [{x["ci95_low"]:.3f}, {x["ci95_high"]:.3f}] | {x["excluding_layer1_spearman"]:.3f} | {x["median_within_sample_spearman"]:.3f} |')
    lines += ['',f'- All layers delete about 25% of tokens, but mean removed historical attention mass spans **{global_mass.min():.1%}–{global_mass.max():.1%}** across layers; last-32-query removed mass spans **{tail_mass.min():.1%}–{tail_mass.max():.1%}**.',
              f'- Across layer means, entropy tracks removed attention mass strongly (Spearman rho `{entropy_mass_rho:.3f}` for all-query attention; `{tail_entropy_mass_rho:.3f}` for last-32-query attention).',
              f'- Removed historical mass vs KL has Pearson r `{mass_kl_pearson:.3f}` but falls to `{mass_kl_pearson_no_l1:.3f}` after excluding layer 1; Spearman and per-task correlations in the tables show that this apparent linear relation is dominated by layer 1.',
              f'- Layer 1: KL `{layer_mean_k[0]:.6g}`, global entropy `{entropy[0]:.3f}`, removed historical mass `{global_mass[0]:.1%}`; layer 19: KL `{layer_mean_k[18]:.6g}`, global entropy `{entropy[18]:.3f}`, removed historical mass `{global_mass[18]:.1%}`.',
              '', '### Dataset-specific rank correlations', '',
              '| Dataset | All-query entropy vs KL | Last-32 entropy vs KL | Deleted mass vs KL |',
              '|---|---:|---:|---:|']
    for task in DATASETS:
        ix=[i for i,v in enumerate(datasets) if v==task]
        lines.append(f'| {task} | {rho(values["global_normalized_entropy"][ix].mean(0),k[ix].mean(0)):.3f} | {rho(values["tail32_normalized_entropy"][ix].mean(0),k[ix].mean(0)):.3f} | {rho(values["global_deleted_mass"][ix].mean(0),k[ix].mean(0)):.3f} |')
    lines += ['', '## Interpretation', '',
              'The same **token coverage ratio** produces different **attention-mass coverage**. '
              'This establishes the coverage mismatch proposed in the motivation under the fixed 25% mask protocol. '
              'Entropy predicts attention-mass loss, but its rank association with output KL is weak, gets weaker without layer 1, and is absent on GSM8K. '
              'Thus the stronger claim that high-KL layers are consistently more diffuse is **not established** by these data. '
              'The attention-peak selection rule, layer response, and positional/causal effects all influence KL. '
              'This one-budget profiling does **not** show that transferring anchor slots from concentrated layers to diffuse layers improves output quality. '
              'That allocation claim requires a controlled fixed-total-budget intervention at multiple pruning ratios.', '',
              'See `attention_dispersion_per_sample.csv`, `attention_dispersion_layerwise.csv`, '
              '`attention_dispersion_correlations.csv`, and `attention_dispersion_vs_kl.png`.']
    (HERE/'ATTENTION_DISPERSION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('entropy rho',entry('global_normalized_entropy')['layer_mean_spearman'],
          'mass rho',entry('global_deleted_mass')['layer_mean_spearman'])


if __name__=='__main__':
    main()
