"""Create per-layer/loop/head TriAttention query statistics from plain text JSONL."""
import argparse
import json
from pathlib import Path

import torch

from engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).parents[1] / 'configs' / 'ouro.example.json')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--texts', type=Path, required=True, help='JSONL with a text field; use calibration/training text only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.model_path:
        cfg['models']['ouro']['path'] = args.model_path
    engine = Engine(cfg, 'ouro')
    mc = cfg['models']['ouro']
    shape = (mc['layers'], mc['loops'], mc['query_heads'], mc['head_dim'] // 2)
    sums = {key: torch.zeros(shape, dtype=torch.float64) for key in ('q_real', 'q_imag', 'q_abs')}
    counts = torch.zeros(shape[:2], dtype=torch.long)
    calls = [0]

    def capture(layer):
        def hook(module, inputs, output):
            r, l = divmod(calls[0], mc['layers'])
            if l != layer or r >= mc['loops']:
                raise RuntimeError('Unexpected Ouro layer/loop forward order')
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            q = tensor.reshape(-1, mc['query_heads'], mc['head_dim']).float()
            qr, qi = q.chunk(2, dim=-1)
            values = {'q_real': qr, 'q_imag': qi, 'q_abs': torch.sqrt(qr * qr + qi * qi)}
            for key, value in values.items():
                sums[key][l, r] += value.sum(0).detach().cpu().double()
            counts[l, r] += q.shape[0]
            calls[0] += 1
        return hook

    handles = [layer.self_attn.q_proj.register_forward_hook(capture(i))
               for i, layer in enumerate(engine.model.model.layers)]
    rows = [json.loads(line) for line in args.texts.read_text().splitlines() if line.strip()][:args.limit]
    if not rows:
        raise ValueError('No calibration texts found')
    with torch.inference_mode():
        for index, row in enumerate(rows):
            ids = engine.tok.encode(row['text'], add_special_tokens=False)
            if not ids:
                continue
            if len(ids) > engine.model.config.max_position_embeddings:
                raise ValueError('Calibration text exceeds model context; preselect complete shorter texts')
            calls[0] = 0
            engine.model(torch.tensor([ids], device='cuda'), use_cache=False,
                         exit_at_step=mc['loops'] - 1, use_weighted_exit=False, logits_to_keep=1)
            if calls[0] != mc['layers'] * mc['loops']:
                raise RuntimeError('Did not observe every layer in every loop')
            print('calibrated', index, 'tokens', len(ids), flush=True)
    for handle in handles:
        handle.remove()
    payload = {key: (value / counts[..., None, None]).float() for key, value in sums.items()}
    payload['omega'] = engine.model.model.rotary_emb.inv_freq.detach().float().cpu()
    if tuple(payload['omega'].shape) != (mc['head_dim'] // 2,):
        raise RuntimeError('Unexpected RoPE frequency shape')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print('saved calibration:', args.output)


if __name__ == '__main__':
    main()
