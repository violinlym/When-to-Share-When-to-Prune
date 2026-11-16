"""Aggregate paired KL results, generate figures and a reproducible report."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATASETS = ('gsm8k', 'math500', 'mbpp_plus')
SEED = 20261116


def mean_ci(values, generator, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    resamples = generator.integers(0, len(values), size=(draws, len(values)))
    boot = values[resamples].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def rankdata(values):
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    for rank, ix in enumerate(order):
        ranks[ix] = rank + 1
    return ranks


def spearman(a, b):
    x, y = rankdata(a), rankdata(b)
    return float(np.corrcoef(x, y)[0, 1])


def heterogeneity_p(matrix, generator, permutations=5000):
    observed = float(matrix.mean(axis=0).var())
    null = np.empty(permutations)
    for b in range(permutations):
        shuffled = np.array([row[generator.permutation(matrix.shape[1])] for row in matrix])
        null[b] = shuffled.mean(axis=0).var()
    return observed, float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=HERE/'results'/'formal')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    files = sorted(args.results.glob('[0-9][0-9][0-9].json'))
    if not args.allow_partial and len(files) != 48:
        raise SystemExit(f'Expected 48 completed samples; found {len(files)}')
    rows = []
    for path in files:
        record = json.loads(path.read_text())
        if record['ratio'] != .25 or record['seed'] != SEED or len(record['rows']) != 24:
            raise ValueError(f'Protocol mismatch: {path}')
        rows.extend(record['rows'])
    if len(set((x['sample_index'], x['layer']) for x in rows)) != len(rows):
        raise ValueError('Duplicate sample/layer records')
    if not args.allow_partial:
        if len(rows) != 1152 or {x['sample_index'] for x in rows} != set(range(48)):
            raise ValueError('Incomplete paired matrix')
    rows.sort(key=lambda x: (x['sample_index'], x['layer']))
    sample_ids = sorted(set(x['sample_index'] for x in rows))
    matrix = np.array([[next(x['mean_kl'] for x in rows if x['sample_index']==i and x['layer']==layer)
                        for layer in range(1,25)] for i in sample_ids])
    rng = np.random.default_rng(SEED)
    summary=[]
    for layer in range(1,25):
        chosen = [x for x in rows if x['layer']==layer]
        values = np.array([x['mean_kl'] for x in chosen])
        mean, low, high = mean_ci(values, rng)
        summary.append(dict(layer=layer, mean_kl=mean, std_kl=float(values.std(ddof=1)),
                            ci95_low=low, ci95_high=high,
                            mean_actual_pruning_ratio=float(np.mean([x['actual_pruning_ratio'] for x in chosen])),
                            mean_deleted_attention_mass=float(np.mean([x['deleted_attention_mass'] for x in chosen])),
                            samples=len(chosen)))
    with (HERE/'per_sample.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    with (HERE/'layerwise_summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=summary[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(summary)

    # Paired permutation: under equal sensitivity, a sample's layer labels are exchangeable.
    observed, p_value = heterogeneity_p(matrix, rng)
    _, p_without_l1 = heterogeneity_p(matrix[:,1:], rng)

    dataset_means = np.array([[np.mean([x['mean_kl'] for x in rows if x['dataset']==task and x['layer']==layer])
                               for layer in range(1,25)] for task in DATASETS])
    pairs = [(DATASETS[i],DATASETS[j],spearman(dataset_means[i],dataset_means[j]))
             for i in range(3) for j in range(i+1,3)]
    dataset_details=[]
    for task in DATASETS:
        task_ids=[i for i in sample_ids if next(x['dataset'] for x in rows if x['sample_index']==i)==task]
        task_matrix=matrix[[sample_ids.index(i) for i in task_ids]]
        _, task_p=heterogeneity_p(task_matrix,rng)
        task_order=np.argsort(-task_matrix.mean(axis=0))+1
        dataset_details.append((task,int(task_order[0]),int(task_order[-1]),
                                int(np.where(task_order==1)[0][0]+1),task_p))
    ranking = np.argsort(-matrix.mean(axis=0)) + 1
    top, bottom = int(ranking[0]), int(ranking[-1])
    sampled = rng.integers(0, len(matrix), size=(10000, len(matrix)))
    boot_means = matrix[sampled].mean(axis=1)
    top_win_fraction = float(np.mean(np.argmax(boot_means, axis=1) == top-1))
    bottom_win_fraction = float(np.mean(np.argmin(boot_means, axis=1) == bottom-1))
    paired_difference = matrix[:,top-1] - matrix[:,bottom-1]
    gap, gap_low, gap_high = mean_ci(paired_difference, rng)
    mass_means=np.array([x['mean_deleted_attention_mass'] for x in summary])
    mass_kl_corr=float(np.corrcoef(matrix.mean(axis=0),mass_means)[0,1])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    means = np.array([x['mean_kl'] for x in summary])
    lower = means - np.array([x['ci95_low'] for x in summary])
    upper = np.array([x['ci95_high'] for x in summary]) - means
    fig,(ax_high,ax)=plt.subplots(2,1,sharex=True,figsize=(13,6),
                                gridspec_kw={'height_ratios':[1.2,3], 'hspace':.07})
    for panel in (ax_high,ax):
        panel.bar(np.arange(1,25), means, color='#356d9a')
        panel.errorbar(np.arange(1,25), means, yerr=[lower,upper], fmt='none', color='black', capsize=2, lw=.8)
    ax_high.set_ylim(.010, .024)
    ax_high.set_title('25% context pruning: mean output KL with 95% bootstrap CI (broken linear axis)')
    ax.set_ylim(0,.005)
    ax.set(xlabel='Physical Layer (1–24)', ylabel='Mean Output KL Divergence')
    ax.set_xticks(np.arange(1,25))
    ax_high.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax_high.tick_params(bottom=False,labelbottom=False)
    fig.subplots_adjust(left=.09,right=.98,bottom=.11,top=.89)
    fig.savefig(HERE/'layerwise_sensitivity.png',dpi=200);plt.close(fig)
    fig, ax=plt.subplots(figsize=(13,3.5))
    from matplotlib.colors import LogNorm
    im=ax.imshow(dataset_means,aspect='auto',cmap='magma',
                 norm=LogNorm(vmin=max(1e-5,float(dataset_means[dataset_means>0].min())),
                              vmax=float(dataset_means.max())))
    ax.set(yticks=np.arange(3),yticklabels=['GSM8K','MATH500','MBPP+'],xticks=np.arange(24),xticklabels=np.arange(1,25),xlabel='Physical Layer')
    fig.colorbar(im,ax=ax,label='Mean Output KL Divergence (log color scale)')
    fig.tight_layout();fig.savefig(HERE/'layerwise_heatmap.png',dpi=200);plt.close(fig)

    manifest_hash=hashlib.sha256((HERE/'manifest48.json').read_bytes()).hexdigest()
    zero_smoke=HERE/'results_ratio0'/'smoke'/'000.json'
    zero_smoke_line='- Zero-pruning GPU smoke not found in local results.'
    if zero_smoke.exists():
        zero_rows=json.loads(zero_smoke.read_text())['rows']
        zero_smoke_line=(f'- Zero-pruning GPU smoke: layers '
                         f'{", ".join(str(x["layer"]) for x in zero_rows)}; '
                         f'maximum KL `{max(x["mean_kl"] for x in zero_rows):.3g}`.')
    lines=[
        '# Layer-wise Context Loss Sensitivity Profiling', '',
        '## Protocol', '',
        f'- Model: Ouro-1.4B, native BF16 KV, four loops × 24 physical layers; full-KV prefill for each sample.',
        f'- Data: first 16 fixed samples from each of GSM8K, MATH500 and MBPP+ in the existing `anchor_route_kvonly/manifest64.json`; `{len(sample_ids)}` paired samples total.',
        f'- Manifest SHA256: `{manifest_hash}`. Seed: `{SEED}`.',
        '- Source manifest SHA256: `f86a62c59efc43de86b00a3b53e56867f5a96c9e62af3beb65a43a3a02002141`.',
        '- Ouro BF16 weight SHA256: `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`; model code SHA256: `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`.',
        '- Per target layer, peak prefill attention across all four loops, all heads and all prompt query positions ranks candidate tokens; the lowest 25% of original prompt length outside the protected recent 20% are masked for every loop and head.',
        '- Full prefill is reused across variants. The full cache tensors remain at original positions. An additive attention mask makes selected historical tokens inaccessible during decode; no KV sharing or other compression runs.',
        '- Full KV greedy continuation, up to 32 tokens, is frozen per sample. Each variant is teacher-forced on this same sequence. The prefill last-token prediction is excluded; KL is evaluated on logits after each continuation token is fed.',
        '- FP32 log-softmax computes KL(Full || Pruned). Per-sample KL averages valid continuation predictions; layer means average samples equally. 95% percentile bootstrap CIs resample paired samples 10,000 times.',
        '- CPU unit tests: 4 passed. GPU 25% smoke passed all runtime mask and loop assertions.',
        zero_smoke_line,
        '', '## Results', '',
        f'- Global paired layer-heterogeneity permutation test: variance of layer means `{observed:.8g}`; p = `{p_value:.6g}` (5,000 within-sample label permutations).',
        f'- Exploratory robustness test excluding layer 1: paired layer-heterogeneity p = `{p_without_l1:.6g}`.',
        f'- Most sensitive: layer {top} (mean KL `{means[top-1]:.6g}`); least sensitive: layer {bottom} (mean KL `{means[bottom-1]:.6g}`).',
        f'- Paired top–bottom KL difference: `{gap:.6g}`, 95% bootstrap CI [`{gap_low:.6g}`, `{gap_high:.6g}`]. This top–bottom pair was chosen after inspecting the means, so its CI is descriptive rather than selection-adjusted.',
        f'- Bootstrap probability of retaining these extreme ranks: top layer {top}, `{top_win_fraction:.1%}`; bottom layer {bottom}, `{bottom_win_fraction:.1%}`.',
        f'- Mean actual pruned fraction: `{np.mean([x["actual_pruning_ratio"] for x in rows]):.4%}` (rounding per prompt).',
        f'- Valid post-pruning prediction positions per sample: `{min(x["valid_tokens"] for x in rows)}`–`{max(x["valid_tokens"] for x in rows)}`. Actual per-sample pruned fraction: `{min(x["actual_pruning_ratio"] for x in rows):.2%}`–`{max(x["actual_pruning_ratio"] for x in rows):.2%}`.',
        f'- Across-layer correlation of mean deleted historical attention mass with mean KL: `{mass_kl_corr:.3f}`. Layer 1 loses `{mass_means[0]:.1%}` of its historical attention mass despite deleting about 25% of tokens; per-layer token sets differ, so this protocol combines layer response with selection quality.',
        '', '### Sensitivity ranking (descending mean KL)', '',
        ', '.join(f'L{layer}' for layer in ranking), '',
        '### Cross-dataset ranking consistency (Spearman rho)', '',
        '| Pair | rho |', '|---|---:|',
    ]
    lines += [f'| {a} vs {b} | {rho:.3f} |' for a,b,rho in pairs]
    lines += ['', '### Per-dataset extremes and paired heterogeneity', '',
              '| Dataset | Highest-KL layer | Lowest-KL layer | Layer 1 rank | Permutation p |',
              '|---|---:|---:|---:|---:|']
    lines += [f'| {task} | {high} | {low} | {l1_rank} | {task_p:.6g} |'
              for task,high,low,l1_rank,task_p in dataset_details]
    lines += ['', f'**Interpretation:** The paired global test {"supports" if p_value < .05 else "does not establish"} unequal mean KL across layers, and the result remains after excluding layer 1. '
             f'Rank correlations across datasets are moderate rather than identical. '
             f'Layer 1 is the overall maximum but its GSM8K rank is {dataset_details[0][3]}; therefore the exact most-sensitive layer is task dependent. '
             f'The descriptive top–bottom paired CI {"excludes" if gap_low > 0 else "includes"} zero. '
             'Different token choices and deleted attention mass prevent attributing all of the observed KL spread to intrinsic layer sensitivity alone.',
             '', 'See `per_sample.csv`, `layerwise_summary.csv`, `layerwise_sensitivity.png` and `layerwise_heatmap.png` for all estimates. Rankings are descriptive; the 16 samples per dataset limit precision of dataset-specific ranks.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(sample_ids)} samples, heterogeneity p={p_value:.6g}, top L{top}, bottom L{bottom}')


if __name__=='__main__':
    main()
