"""Independent statistics, scalar sorting and native online route checks."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from granular_cache import GranularCache
from exposure_cache import ExposureCache, visible_importance
from test_joint import fake_attention


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1105)
    mc = dict(layers=2, loops=4, kv_heads=2, query_heads=2, head_dim=8)
    for n in [1, 4, 9]:
        weights = torch.rand(4, 2, n, n).tril()
        weights /= weights.sum(-1, keepdim=True)
        means = visible_importance(weights.sum(-2))
        expected = torch.stack([weights[:, :, token:, token].mean(-1) for token in range(n)], -1)
        torch.testing.assert_close(means, expected, rtol=2e-6, atol=1e-7)
    cases, steps = 0, 0
    for dtype in [torch.float32, torch.bfloat16]:
        for method in ['head_version', 'head_loop_version']:
            n = 7
            values = [torch.randn(2, 4, 2, n, 8, dtype=dtype) for _ in range(2)]
            weights = [torch.rand(4, 2, n, n).tril() for _ in range(2)]
            weights = [w / w.sum(-1, keepdim=True) for w in weights]
            masses = [w.sum(-2) for w in weights]
            source = dict(prompt_tokens=n, kv=values, scores=masses)
            cache = GranularCache(source, mc, .25, method)
            # Independent component loops and Python stable sort, not the allocator's threshold helper.
            priorities = []
            for layer in range(2):
                for kind in range(2):
                    for head in range(2):
                        for token in range(n):
                            for loop in range(3):
                                old = values[layer][kind, loop, head, token].float()
                                new = values[layer][kind, loop+1, head, token].float()
                                residual = (new-old).norm() / (old.norm()+1e-8)
                                if method == 'head_version':
                                    importance = (masses[layer] / torch.arange(n, 0, -1))[..., token].mean(0)[head]
                                else:
                                    importance = masses[layer][loop, head, token] / (n-token)
                                priorities.append(float(torch.minimum(residual*importance, importance)))
            count = 2*2*2*(4*((n+1)//2)-n)
            chosen = sorted(range(len(priorities)), key=lambda j: (-priorities[j], j))[:count]
            expected = torch.zeros(len(priorities), dtype=torch.bool)
            expected[chosen] = True
            actual = torch.stack([s.ptr[..., :3] >= 0 for s in cache.states]).reshape(-1)
            assert torch.equal(actual, expected)
            cases += 1
            cache.audit_source(source)
            history_mass = [m.clone() for m in masses]
            for step in range(8):
                before = cache.decisions()
                added = [torch.randn(2, 4, 2, 1, 8, dtype=dtype) for _ in range(2)]
                new_weights = [[] for _ in range(2)]
                for loop in range(4):
                    for layer in range(2):
                        k, v = cache.update(added[layer][0, loop][None], added[layer][1, loop][None], loop*2+layer)
                        expected = torch.empty(2, 2, cache.n, 8, dtype=dtype)
                        for kind in range(2):
                            for head in range(2):
                                for token in range(cache.n):
                                    representative = int(before[layer, kind, head, token, loop])
                                    expected[kind, head, token] = values[layer][kind, representative, head, token]
                        assert torch.equal(k[0, :, :-1], expected[0]) and torch.equal(v[0, :, :-1], expected[1])
                        query = torch.randn(1, 2, 1, 8, dtype=dtype)
                        _, w = cache.attention(fake_attention, SimpleNamespace(layer_idx=layer), query, k, v, None)
                        new_weights[layer].append(w[0, :, 0].float())
                cache.finish()
                values = [torch.cat((old, new), dim=3) for old, new in zip(values, added)]
                for layer in range(2):
                    history_mass[layer] = torch.cat((history_mass[layer], torch.zeros(4, 2, 1)), -1) + torch.stack(new_weights[layer])
                    assert torch.equal(cache.states[layer].mass, history_mass[layer])
                after = cache.decisions()
                assert (after[..., :before.shape[-2], :] >= before).all() and (after[..., 3] < 4).all()
                cache.audit_source(dict(kv=values))
                mem = cache.memory()
                assert mem['payload_bytes'] == 8*4*((cache.n+1)//2)*8*values[0].element_size()
                assert mem['auxiliary_breakdown_bytes']['scalar_attention_statistics'] == 2*4*2*cache.n*4
                steps += 1
        # Equal head/loop statistics reduce exactly to the existing pooled control.
        n = 6
        values = [torch.randn(2, 4, 2, n, 8, dtype=dtype) for _ in range(2)]
        mass = torch.tensor([1.,2.,3.,4.,5.,6.]).expand(4,2,n).clone()
        source = dict(prompt_tokens=n, kv=values, scores=[mass.clone(),mass.clone()])
        old = ExposureCache(source, mc, .25, 'exposure_version')
        for method in ['head_version','head_loop_version']:
            new = GranularCache(source, mc, .25, method)
            assert np.array_equal(old.decisions(), new.decisions())
    result = dict(status='PASS', independent_causal_mean_cases=3, independent_sort_cases=cases,
                  native_online_steps=steps, uniform_head_loop_reduction_cases=4,
                  exact_current_length_payload=True, attention_metadata_accounted=True)
    (Path(__file__).resolve().parents[1]/'results/granular_cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
