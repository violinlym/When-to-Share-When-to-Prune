"""CPU fixtures for global token roots and asymmetric version routing."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent / 'streaming_joint_allocation/implementations'))
import numpy as np
import torch
from global_cache import GlobalContextCache, select_global_roots
from streaming_cache import threshold_keep
from test_joint import fake_attention
from common import sha
from score import audit


def sorted_reference(roots, children, budget):
    cost = roots.shape[1]
    root_size = roots.numel()
    values = torch.cat((roots.reshape(-1), children.reshape(-1))).tolist()
    selected = set(sorted(range(len(values)), key=lambda i: (-values[i], i))[:budget])
    for token in range(roots.shape[0]):
        block = set(range(token * cost, (token + 1) * cost))
        if 0 < len(block & selected) < cost:
            selected.update(block)
            selected_roots = {i for i in selected if i < root_size}
            optional = sorted((i for i in selected if i >= root_size), key=lambda i: (-values[i], i))
            selected = selected_roots | set(optional[:budget-len(selected_roots)])
    return sorted(selected)


def check_selection(roots, children, budget):
    keep_roots, keep_children = select_global_roots(roots, children, budget)
    keep = torch.cat((keep_roots.flatten(), keep_children.flatten()))
    assert keep.nonzero().flatten().tolist() == sorted_reference(roots, children, budget)
    assert int(keep.sum()) == budget
    assert (keep_roots == keep_roots[:, :1]).all()
    assert not (keep_children & ~keep_roots[:, 0][None, :, None]).any()
    if keep_roots.all():
        assert torch.equal(keep_children, threshold_keep(children, budget-roots.numel()))
    return keep_roots, keep_children


def main():
    torch.set_num_threads(1)
    torch.manual_seed(916)
    cases = 0
    invariant_cases = 0
    for cost in [2, 4, 8]:
        for trial in range(20):
            root = torch.rand(7)
            children = root[None, :, None] * torch.rand(cost, 7, 3)
            if trial % 2 == 0:
                root = (root * 5).floor() / 5
                children = torch.minimum((children * 5).floor() / 5, root[None, :, None])
            for budget in [3 * cost, 7 * cost, 14 * cost]:
                roots, kept = check_selection(root[:, None].expand(-1, cost), children, budget)
                invariant_cases += int(roots.all())
                cases += 1
    # The cutoff splits the last root. Completing it must discard the LAST
    # tied child, preserving the original capped-version top-k rule.
    roots = torch.tensor([10., 9., 1.])[:, None].expand(-1, 2)
    children = torch.zeros(2, 3, 3)
    children[0, 0] = 2.
    kept_roots, kept_children = check_selection(roots, children, 8)
    assert kept_roots.all() and kept_children.flatten().nonzero().flatten().tolist() == [0, 1]
    cases += 1
    invariant_cases += 1
    mc = dict(layers=2, loops=4, kv_heads=2, query_heads=2, head_dim=8)
    steps = 0
    for dtype in [torch.float32, torch.bfloat16]:
        vectors = [torch.randn(2, 1, 2, 4, 8, dtype=dtype) * torch.tensor([-1, 1, -1, 1], dtype=dtype)[None, :, None, None, None]
                   for _ in range(2)]
        score = torch.tensor([40., 10., .01, .001], dtype=dtype).expand(4, 2, 4).clone()
        source = dict(prompt_tokens=4, kv=vectors, scores=[score.clone(), score.clone()])
        cache = GlobalContextCache(source, mc, .25)
        assert cache.coverage()['global_context_union'] == .5
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
                                    expected[component, head, token] = vectors[layer][component, representative, head, token]
                    assert torch.equal(k[0, :, :-1], expected[0]) and torch.equal(v[0, :, :-1], expected[1])
                    query = torch.randn(1, 2, 1, 8, dtype=dtype)
                    _, weights = cache.attention(fake_attention, SimpleNamespace(layer_idx=layer), query, k, v, None)
                    dead = torch.from_numpy(before[layer, 0, :, :, 3] == 4)
                    assert (weights[0, :, 0, :-1][dead] == 0).all()
            cache.finish()
            vectors = [torch.cat((old, new), dim=3) for old, new in zip(vectors, added)]
            after = cache.decisions()
            alive = after[..., 3] < 4
            assert (alive == alive[0, 0, 0]).all()
            assert (after[..., :before.shape[-2], :] >= before).all()
            cache.audit_source(dict(kv=vectors))
            memory = cache.memory()
            assert memory['payload_bytes'] == 8 * 4 * ((cache.n + 1) // 2) * 8 * vectors[0].element_size()
            steps += 1
    cache = GlobalContextCache(source, mc, .25)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'state.npz'
        np.savez_compressed(path, initial=cache.decisions(), final=cache.decisions())
        raw = dict(input_ids=list(range(4)), generated_tokens=1, prefill_cache=cache.memory(),
                   final_cache=cache.memory(), byte_history=[], decisions_path=str(path),
                   decisions_sha256=sha(path), prefill_coverage=cache.coverage(), final_coverage=cache.coverage(),
                   decision_format='loop_source_versions')
        audit(raw, dict(context_ids=list(range(4))), 'global_context_stable', mc)
    result = dict(status='PASS', independent_sorted_reference_cases=cases, online_steps=steps,
                  zero_pruning_equivalence_cases=invariant_cases, split_root_tie_repair=True,
                  global_root_closure=True, exact_current_length_budget=True, no_resurrection=True,
                  native_routing_and_masks=True, pruning_exercised=True, new_scoring_route_audit=True,
                  scope='CPU engineering fixtures, no quality or hardware-performance conclusion')
    (ROOT / 'results/cpu_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print('GLOBAL_CONTEXT_CPU_CHECKS_PASS', cases, steps)


if __name__ == '__main__':
    main()
