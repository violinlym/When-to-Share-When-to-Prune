"""Summarize existing dense traces, never simulate a sparse model from recall."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values, p):
    v = sorted(values)
    x = (len(v) - 1) * p
    a = int(x)
    return v[a] + (v[min(a + 1, len(v) - 1)] - v[a]) * (x - a)


def run(source, output):
    grouped = defaultdict(list)
    with source.open() as f:
        for row in csv.DictReader(f):
            if row['predictor'] == 'previous' and row['metric'] in (
                    'topk_recall', 'attention_mass_recall', 'history_mass_recall'):
                key = (int(row['target_loop']), float(row['top_ratio']), row['metric'], int(row['prompt']))
                grouped[key].append(float(row['mean']))
    summaries, cells = defaultdict(list), defaultdict(list)
    for (loop, ratio, metric, prompt), vals in grouped.items():
        summaries[loop, ratio, metric].append(statistics.mean(vals))
        cells[loop, ratio, metric].extend(vals)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'predictability_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, lineterminator='\n', fieldnames=['loop', 'top_fraction', 'metric', 'prompts',
            'prompt_median', 'prompt_min', 'prompt_max', 'head_layer_prompt_p10'])
        writer.writeheader()
        for key, vals in sorted(summaries.items()):
            loop, ratio, metric = key
            writer.writerow(dict(loop=loop, top_fraction=ratio, metric=metric, prompts=len(vals),
                prompt_median=statistics.median(vals), prompt_min=min(vals), prompt_max=max(vals),
                head_layer_prompt_p10=quantile(cells[key], .1)))
    (output / 'predictability_provenance.json').write_text(json.dumps({
        'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'evidence': 'REAL_DENSE_PROFILING; NOT SPARSE_EXECUTION',
        'scope': '32 GSM prompts;32 decode queries each;4 sampled layers (1-based 6,12,18,24);16 heads',
        'aggregation': 'Mean over sampled head/layer cells within each prompt, then median over prompts. p10 is of cell query means, not individual queries.',
        'limitations': 'Short dense reference trajectories, shared demonstrations, no causal sparse rollout, no test of compressed-cache interaction.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    args = p.parse_args()
    run(args.source, args.output)
