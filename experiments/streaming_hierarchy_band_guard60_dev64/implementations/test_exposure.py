"""Independent causal means and multi-step native-value / byte-budget checks."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from exposure_cache import ExposureCache, visible_importance
from test_joint import fake_attention


def main():
    torch.set_num_threads(1)
    torch.manual_seed(628)
    for n in [1, 2, 7, 16]:
        weights = torch.rand(4, 2, n, n).tril()
        weights /= weights.sum(-1, keepdim=True)
        mass = weights.sum(-2).mean((0, 1))
        expected = torch.stack([weights[:, :, token:, token].mean() for token in range(n)])
        torch.testing.assert_close(visible_importance(mass), expected, rtol=2e-6, atol=1e-7)
        # Appending one query increments every past token's exposure by one;
        # the newborn token has one observation, including its self-attention.
        new = torch.rand(4, 2, n + 1)
        new /= new.sum(-1, keepdim=True)
        enlarged = torch.cat((mass, torch.zeros(1))) + new.mean((0, 1))
        expected_new = torch.stack([
            torch.cat((weights[:, :, token:, token].reshape(-1), new[:, :, token].reshape(-1))).mean()
            for token in range(n)] + [new[:, :, -1].mean()])
        torch.testing.assert_close(visible_importance(enlarged), expected_new, rtol=2e-6, atol=1e-7)
    mc = dict(layers=2, loops=4, kv_heads=2, query_heads=2, head_dim=8)
    parity_cases, steps, pruning_cases = 0, 0, 0
    for dtype in [torch.float32, torch.bfloat16]:
        for n in [2, 3, 4, 7]:
            values = [torch.randn(2, 4, 2, n, 8, dtype=dtype) for _ in range(2)]
            mass = torch.arange(n, 0, -1, dtype=torch.float32).expand(4, 2, n).clone()
            source = dict(prompt_tokens=n, kv=values, scores=[mass.clone(), mass.clone()])
            version = ExposureCache(source, mc, .25, 'exposure_version')
            joint = ExposureCache(source, mc, .25, 'exposure_global')
            assert joint.coverage()['global_context_union'] == 1.
            assert np.array_equal(version.decisions(), joint.decisions())
            parity_cases += 1
        for method in ['exposure_version', 'exposure_global']:
            values = [torch.randn(2, 1, 2, 4, 8, dtype=dtype) * torch.tensor([-1, 1, -1, 1], dtype=dtype)[None, :, None, None, None] for _ in range(2)]
            mass = (torch.tensor([40., 10., .01, .001]) * torch.arange(4, 0, -1)).expand(4, 2, 4).clone()
            source = dict(prompt_tokens=4, kv=values, scores=[mass.clone(), mass.clone()])
            cache = ExposureCache(source, mc, .25, method)
            assert cache.coverage()['global_context_union'] == (1. if method == 'exposure_version' else .5)
            pruning_cases += int(method == 'exposure_global')
            cache.audit_source(source)
            for step in range(8):
                before = cache.decisions()
                added = [torch.randn(2, 4, 2, 1, 8, dtype=dtype) for _ in range(2)]
                for loop in range(4):
                    for layer in range(2):
                        k, v = cache.update(added[layer][0, loop][None], added[layer][1, loop][None], loop * 2 + layer)
                        expected = torch.zeros(2, 2, cache.n, 8, dtype=dtype)
                        for component in range(2):
                            for head in range(2):
                                for token in range(cache.n):
                                    representative = int(before[layer, component, head, token, loop])
                                    if representative < 4:
                                        expected[component, head, token] = values[layer][component, representative, head, token]
                        assert torch.equal(k[0, :, :-1], expected[0]) and torch.equal(v[0, :, :-1], expected[1])
                        query = torch.randn(1, 2, 1, 8, dtype=dtype)
                        _, weights = cache.attention(fake_attention, SimpleNamespace(layer_idx=layer), query, k, v, None)
                        dead = torch.from_numpy(before[layer, 0, :, :, 3] == 4)
                        assert (weights[0, :, 0, :-1][dead] == 0).all()
                cache.finish()
                values = [torch.cat((old, new), dim=3) for old, new in zip(values, added)]
                after = cache.decisions()
                assert (after[..., :before.shape[-2], :] >= before).all()
                alive = after[..., 3] < 4
                assert (alive == alive[0, 0, 0]).all()
                if method == 'exposure_version':
                    assert alive.all()
                cache.audit_source(dict(kv=values))
                memory = cache.memory()
                assert memory['payload_bytes'] == 8 * 4 * ((cache.n + 1) // 2) * 8 * values[0].element_size()
                steps += 1
    record = dict(status='PASS', causal_attention_mean_cases=8, no_pruning_policy_parity_cases=parity_cases,
                  native_online_steps=steps, pruning_fixtures=pruning_cases, exact_current_length_payload=True,
                  scope='Independent CPU engineering checks; no accuracy or performance claim')
    (Path(__file__).resolve().parents[1] / 'results/exposure_cpu_checks.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
