"""Build full-benchmark plots and audited report after complete scoring."""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers

from run_full import HERE, MODEL, F, SLOT


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def load_csv(name):
    with (HERE/name).open(newline='') as f:return list(csv.DictReader(f))


def main():
    summary=load_csv('accuracy_summary.csv')
    paired=load_csv('paired_stats.csv')
    docs=json.loads((HERE/'manifest1819.json').read_text())
    records={method:[] for method in ('fixed75','adaptive90')}
    for i in range(1819):
        for method in records:
            path=HERE/'results'/'formal'/f'{i:04d}_{method}.json'
            raw=json.loads(path.read_text())
            if raw['manifest_index']!=i or raw['task']!=docs[i]['task']:
                raise ValueError(f'Record mismatch {path}')
            records[method].append(raw)
    if any(len(v)!=1819 for v in records.values()):raise ValueError('Incomplete benchmark')
    layer=np.arange(1,25)
    fig,ax=plt.subplots(figsize=(13,5))
    for method,color in (('fixed75','#bd6751'),('adaptive90','#356d9a')):
        raw=records[method]
        prefill=np.array([[x['initial']['roots_by_layer'][l]/x['prompt_tokens'] for l in range(24)] for x in raw])
        final=np.array([[x['final']['roots_by_layer'][l]/(x['final_full_bytes']//F) for l in range(24)] for x in raw])
        ax.plot(layer,prefill.mean(0),'o-',color=color,label=f'{method} prefill')
        ax.plot(layer,final.mean(0),'--',color=color,alpha=.75,label=f'{method} final')
    ax.set(xlabel='Physical layer (1–24)',ylabel='Mean Root coverage ratio',ylim=(0,1),
           title='Layer-wise A1 Root allocation over 1819 paired prompts')
    ax.set_xticks(layer);ax.legend(ncol=2)
    fig.tight_layout();fig.savefig(HERE/'layerwise_anchor_coverage.png',dpi=200);plt.close(fig)

    budget_rows=[]
    for method in ('fixed75','adaptive90'):
        for phase in ('initial','final'):
            raw=records[method]
            total_full=sum(x[f'{phase}_full_bytes'] for x in raw)
            roots=sum(x[phase]['root_slots']*2*SLOT for x in raw)
            extra=sum(x[phase]['extra_kv_version_slots']*SLOT for x in raw)
            payload=sum(x[phase]['payload_bytes'] for x in raw)
            if roots+extra!=payload:raise ValueError('A1/A2 payload accounting mismatch')
            budget_rows.append(dict(method=method,phase=phase,root_fraction=roots/total_full,
                                    a2_fraction=extra/total_full,payload_fraction=payload/total_full,
                                    unused_to_25pct=.25-payload/total_full,
                                    underfilled_samples=sum(x[phase]['payload_fraction']<.25-1e-10 for x in raw)))
    fig,ax=plt.subplots(figsize=(8,4.5))
    xx=np.arange(4)
    root=np.array([x['root_fraction'] for x in budget_rows])
    a2=np.array([x['a2_fraction'] for x in budget_rows])
    slack=np.array([max(0,x['unused_to_25pct']) for x in budget_rows])
    ax.bar(xx,root,label='A1 K/V Roots',color='#356d9a')
    ax.bar(xx,a2,bottom=root,label='A2 extra K/V versions',color='#61a88a')
    ax.bar(xx,slack,bottom=root+a2,label='Unused budget',color='#ddd2a2')
    ax.axhline(.25,color='black',ls='--',lw=.8)
    ax.set(xticks=xx,xticklabels=[f'{x["method"]}\n{x["phase"]}' for x in budget_rows],
           ylabel='Fraction of Full KV payload',ylim=(0,.28),title='A1/A2 budget distribution')
    ax.legend();fig.tight_layout();fig.savefig(HERE/'kv_budget_distribution.png',dpi=200);plt.close(fig)

    def accuracy(task,method):
        x=next(x for x in summary if x['task']==task and x['method']==method)
        return f'{int(x["correct"])}/{int(x["items"])} ({float(x["accuracy"]):.2%})'
    impl_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    manifest_hash=digest(HERE/'manifest1819.json')
    model_hash=digest(MODEL/'model.safetensors')
    model_code_hash=digest(MODEL/'modeling_ouro.py')
    a2_hash=digest(HERE/'route_gain.py')
    baseline_a2_hash=digest(HERE.parent/'anchor_route_kvonly'/'anchor75'/'route_gain.py')
    if a2_hash!=baseline_a2_hash:raise ValueError('A2 source changed')
    capacity_equal=all(abs(records['fixed75'][i][phase]['payload_fraction']-
                           records['adaptive90'][i][phase]['payload_fraction'])<1e-12
                       for i in range(1819) for phase in ('initial','final'))
    violations=sum(x['coverage_violation_layer_steps'] or 0 for x in records['adaptive90'])
    budget_violations=sum(x['budget_violation_count'] for raw in records.values() for x in raw)
    resurrections=sum(x['root_resurrection_count'] for raw in records.values() for x in raw)
    min_coverage=min(x['minimum_observed_coverage'] for x in records['adaptive90'])
    initial_roots=np.array([[x['initial']['root_slots'] for x in records[m]] for m in ('fixed75','adaptive90')])
    initial_extra=np.array([[x['initial']['extra_kv_version_slots'] for x in records[m]] for m in ('fixed75','adaptive90')])
    final_roots=np.array([[x['final']['root_slots'] for x in records[m]] for m in ('fixed75','adaptive90')])
    final_extra=np.array([[x['final']['extra_kv_version_slots'] for x in records[m]] for m in ('fixed75','adaptive90')])
    lines=['# Adaptive A1 τ=0.9 + original A2: full benchmark', '',
           '## Protocol and provenance', '',
           '- Baseline code commit: `950b76a`; implementation code commit at report generation: `'+impl_commit+'`.',
           '- Ouro-1.4B, native BF16, four loops × 24 layers, greedy decoding. Fixed and adaptive methods use the same Full KV prefill, tokenizer, prompt IDs, generation limits and stop strings for each item.',
           f'- Frozen input manifest: 1319 GSM8K test prompts and 500 MATH500 prompts, SHA256 `{manifest_hash}`. Each item carries task/doc ID, prompt, token IDs, gold answer and source-input SHA256. All source problems and answers were checked against pinned `openai/gsm8k` test revision `740312add88f781978c0658806c59bc2815b9866` and `HuggingFaceH4/MATH-500` test revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`; see `DATA_PROVENANCE.json`.',
           f'- Model weight SHA256 `{model_hash}`; Ouro model code SHA256 `{model_code_hash}`. PyTorch `{torch.__version__}`, Transformers `{transformers.__version__}`.',
           '- Fixed A1: original `anchor75` peak ranking, 20% recent protection, `floor(0.75*N)` K4/V4 Roots per layer. Adaptive A1: identical peak ranking and recent protection, adding roots until 90% historical attention mass per layer. A1 updates use only past observations, weighted by actual prefill/decode query counts.',
           f'- Original A2: same six directed relative-squared K/V costs, access weighting and positive marginal route-gain greedy allocation; `route_gain.py` SHA256 `{a2_hash}` matches the baseline byte-for-byte. Rightward routing and K/V independent masks are likewise inherited unchanged.',
           '- 25% Full KV **payload** cap at every update; metadata measured separately. Deleted roots or versions cannot be restored. The generated raw JSON files are atomically written and resumable.',
           '- GSM8K scorer: last explicit `The answer is` or `####` number in the generated text, compared with gold using Decimal. MATH500 scorer: copied project `scorer_v2.py` balanced last-box extraction plus frozen `math_normalize.py` equivalence; no method-specific scoring. Scorer SHA256: `'+digest(HERE/'score_full.py')+'`; MATH scorer SHA256: `'+digest(HERE/'scorer_v2.py')+'`.',
           '- Full KV reference: 1795 exact-protocol historical records reused after matching task/doc ID, prompt, token IDs, gold, generation parameters and source-input SHA256; 24 records with old manifest hashes regenerated under this manifest.',
           '', '## Accuracy', '',
           '| Method | GSM8K 1319 | MATH500 500 |', '|---|---:|---:|']
    for method in ('full','fixed75','adaptive90'):
        lines.append(f'| {method} | {accuracy("gsm8k",method)} | {accuracy("math500",method)} |')
    lines += ['', '| Task | Adaptive − Fixed | Paired 95% bootstrap CI | Adaptive-only correct | Fixed-only correct | Exact McNemar p |',
              '|---|---:|---:|---:|---:|---:|']
    for x in paired:
        lines.append(f'| {x["task"]} | {float(x["delta_accuracy"]):+.2%} | [{float(x["ci95_low"]):+.2%}, {float(x["ci95_high"]):+.2%}] | {x["adaptive_only_correct"]} | {x["fixed_only_correct"]} | {float(x["mcnemar_exact_p"]):.4g} |')
    lines += ['', '## KV allocation', '',
              '| Method | Stage | A1 Root fraction of Full | A2 extra-version fraction | Total payload fraction | Underfilled items |',
              '|---|---|---:|---:|---:|---:|']
    for x in budget_rows:
        lines.append(f'| {x["method"]} | {x["phase"]} | {x["root_fraction"]:.3%} | {x["a2_fraction"]:.3%} | {x["payload_fraction"]:.3%} | {x["underfilled_samples"]} |')
    lines += ['',f'- Identical per-item prefill and final KV payload for both methods: **{capacity_equal}**.',
              f'- Adaptive A1 releases a mean of **{np.mean(initial_roots[0]-initial_roots[1]):.1f}** physical-layer Root tokens per prompt at prefill; A2 adds **{np.mean(initial_extra[1]-initial_extra[0]):.1f}** more K/V version slots. At final decode, the corresponding means are **{np.mean(final_roots[0]-final_roots[1]):.1f}** Roots and **{np.mean(final_extra[1]-final_extra[0]):.1f}** extra K/V slots.',
              f'- Decode: observed layer-step attention-coverage violations `{violations}`; minimum observed adaptive coverage `{min_coverage:.5f}`. Budget violations `{budget_violations}`; root resurrection events `{resurrections}`.',
              '', '### Per-layer Root allocation', '',
              '| Layer | Fixed prefill | Adaptive prefill | Fixed final | Adaptive final | Adaptive prefill attention mass coverage |',
              '|---:|---:|---:|---:|---:|---:|']
    for layer in range(24):
        def ratio(method,phase):
            xs=records[method]
            return np.mean([x[phase]['roots_by_layer'][layer]/(x['prompt_tokens'] if phase=='initial' else x['final_full_bytes']//F) for x in xs])
        mass=np.mean([x['initial']['historical_attention_mass_coverage_by_layer'][layer] for x in records['adaptive90']])
        lines.append(f'| {layer+1} | {ratio("fixed75","initial"):.2%} | {ratio("adaptive90","initial"):.2%} | {ratio("fixed75","final"):.2%} | {ratio("adaptive90","final"):.2%} | {mass:.2%} |')
    lines += ['', '## Reproduce', '',
              '```bash',
              'python -m unittest -v test_adaptive.py test_scoring.py',
              'CUDA_VISIBLE_DEVICES=4 python run_full.py --smoke',
              '# Launch 16 independent workers, assigning worker w to GPU 4+(w mod 4).',
              'for w in $(seq 0 15); do gpu=$((4 + w % 4)); CUDA_VISIBLE_DEVICES=$gpu python run_full.py --worker $w --workers 16 & done; wait',
              'for w in $(seq 0 3); do gpu=$((4 + w)); CUDA_VISIBLE_DEVICES=$gpu python run_full_reference.py --worker $w --workers 4 & done; wait',
              'python score_full.py',
              'python make_report.py',
              '```', '',
              'Raw generation is retained locally under `results/` and excluded from Git; `gsm8k_full_results.csv` and `math500_full_results.csv` include source paths and SHA256 for every scored record.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('capacity equal',capacity_equal,'violations',violations,'budget',budget_violations)


if __name__=='__main__':main()
