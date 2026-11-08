"""CPU structural check for conservative Version-Tree/root exchange."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from price_exchange_cache import PriceExchangeCache
from test_joint import fake_attention


def main():
    torch.set_num_threads(1); torch.manual_seed(611)
    mc = dict(layers=4, loops=4, kv_heads=2, query_heads=2, head_dim=8)
    kv = [torch.randn(2, 4, 2, 20, 8) for _ in range(4)]
    scores = [torch.rand(4, 2, 20) for _ in range(4)]
    peaks = [row.clone() for row in scores]
    # Persistently unimportant old tokens only in deep layers.
    scores[2][..., :8] = 0.; scores[3][..., :8] = 0.
    peaks[2][..., :8] = 0.; peaks[3][..., :8] = 0.
    # Token0 has low cumulative mass but a historical attention spike: peak gate must veto pruning.
    peaks[2][..., 0] = 1.; peaks[3][..., 0] = 1.
    source = dict(prompt_tokens=20, kv=kv, scores=scores, peak_scores=peaks)
    cache = PriceExchangeCache(source, mc, .4)
    audit = cache.audit_hierarchy()
    assert audit['status'] == 'PASS' and audit['pruned_layer_token_roots'] > 0
    assert all(bool(cache.states[layer].ptr[0, 0, 0, 3] >= 0) for layer in (2, 3))
    assert cache.memory()['auxiliary_breakdown_bytes']['loop_head_peak_statistics'] > 0
    assert cache.coverage()['global_context_union'] == 1.
    before = cache.decisions()
    added = [torch.randn(2, 4, 2, 1, 8) for _ in range(4)]
    for loop in range(4):
        for layer in range(4):
            k, v = cache.update(added[layer][0, loop][None], added[layer][1, loop][None], loop * 4 + layer)
            q = torch.randn(1, 2, 1, 8)
            cache.attention(fake_attention, SimpleNamespace(layer_idx=layer), q, k, v, None)
    cache.finish()
    after = cache.decisions()
    assert (after[..., :before.shape[-2], :] >= before).all()
    audit2 = cache.audit_hierarchy()
    mem = cache.memory()
    full_components = mc['layers'] * 2 * 4 * mc['kv_heads'] * cache.n
    target_components = int(torch.ceil(torch.tensor(full_components * .4)).item())
    assert mem['payload_bytes'] == target_components * mc['head_dim'] * 4
    record = dict(status='PASS', initial_pruned_roots=audit['pruned_layer_token_roots'],
                  final_pruned_roots=audit2['pruned_layer_token_roots'],
                  current_length_retention=.4, recent_protection=True,
                  peak_veto=True,
                  deep_only=True, irreversible=True, global_context_union=cache.coverage()['global_context_union'])
    path = Path(__file__).resolve().parents[1] / 'results/price_exchange_cpu_checks.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
