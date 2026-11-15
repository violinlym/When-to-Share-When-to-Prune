"""Run an Ouro attention-pruning adapter on a JSONL prompt file."""
import argparse
import json
from pathlib import Path

import torch

from engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).parents[1] / 'configs' / 'ouro.example.json')
    parser.add_argument('--input', type=Path, required=True, help='JSONL with id, context, max_new_tokens, and optional stop strings')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', type=str, help='Override models.ouro.path from the config')
    parser.add_argument('--methods', nargs='+', choices=('h2o', 'snapkv_d', 'rkv', 'triattention'),
                        default=('h2o', 'snapkv_d', 'rkv', 'triattention'))
    parser.add_argument('--loop-policy', choices=('bound', 'loop'), default='bound')
    parser.add_argument('--snapkv-window-policy', choices=('budget-aware', 'original'), default='budget-aware',
                        help='Updated SnapKV-D default; original reproduces min(128,capacity) recent protection')
    parser.add_argument('--limit', type=int, help='Optional prefix limit for a local smoke run')
    parser.add_argument('--tri-calibration', type=Path, help='TriAttention calibration .pt from calibrate_ouro.py')
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.model_path:
        cfg['models']['ouro']['path'] = args.model_path
    if not torch.cuda.is_available():
        raise RuntimeError('This runner requires a CUDA device (NVIDIA)')
    engine = Engine(cfg, 'ouro')
    if 'triattention' in args.methods:
        if not args.tri_calibration:
            raise ValueError('--tri-calibration is required when running TriAttention')
        cal = torch.load(args.tri_calibration, map_location='cpu', weights_only=True)
        engine.calibration = {k: v.to('cuda') for k, v in cal.items() if torch.is_tensor(v)}

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as out:
        for index, row in enumerate(rows):
            doc = {
                'context': row['context'],
                'context_ids': engine.tok.encode(row['context'], add_special_tokens=False),
                'generation_kwargs': {
                    'max_gen_toks': int(row.get('max_new_tokens', 512)),
                    'until': row.get('stop', []),
                    'do_sample': False,
                },
            }
            source = engine.prepare(doc)
            for method in args.methods:
                condition = {
                    'algorithm': method,
                    'method': args.loop_policy,
                    'ratio': cfg['budget']['ratio'],
                }
                if method == 'snapkv_d' and args.snapkv_window_policy == 'budget-aware':
                    condition['snapkv_recent_cap_fraction'] = .5
                result, initial, final = engine.generate(doc, source, condition)
                record = {
                    'id': row.get('id', index), 'method': method, 'loop_policy': args.loop_policy,
                    'snapkv_window_policy': args.snapkv_window_policy if method == 'snapkv_d' else None,
                    'compression_ratio': cfg['budget']['compression_ratio'],
                    'generated_text': result['generated_text'],
                    'generated_tokens': result['generated_tokens'], 'stop_reason': result['stop_reason'],
                    'prefill_cache': result['prefill_cache'], 'final_cache': result['final_cache'],
                    'prefill_coverage': result['prefill_coverage'], 'final_coverage': result['final_coverage'],
                    'generation_seconds': result['generation_seconds'],
                    'peak_allocated_bytes': result['peak_allocated_bytes'],
                    'peak_reserved_bytes': result['peak_reserved_bytes'],
                }
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                out.flush()
            del source


if __name__ == '__main__':
    main()
