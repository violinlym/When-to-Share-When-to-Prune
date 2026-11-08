"""CPU engineering tests, including paired-root repair and fully empty layers."""
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT, sha, write
from joint_cache import JointCache, CappedVersionCache, select_tree
from score import audit


def sorted_reference(scores, budget):
    values = scores.flatten().tolist()
    selected = set(sorted(range(len(values)), key=lambda i: (-values[i], i))[:budget])
    for root in range(len(values)//8):
        a, b = root*8, root*8+1
        if (a in selected) != (b in selected):
            selected.update((a, b))
            victim = min((i for i in selected if i % 8 >= 2), key=lambda i: (values[i], i))
            selected.remove(victim)
    return sorted(selected)


def fake_attention(module, q, k, v, mask, **kwargs):
    logits = q.float()@k.float().transpose(-1, -2)/math.sqrt(q.shape[-1])
    if mask is not None:
        logits += mask.float()
    weights = logits.softmax(-1).to(q.dtype)
    return weights@v, weights


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    explicit = torch.tensor([[10., 10., 9., 8., 7., 6., 6., 1.], [5.5, 5.5, .1, .1, .1, .1, .1, .1]])
    for values in [explicit]+[torch.rand(12, 8) for _ in range(20)]:
        if values is not explicit:
            values[:, 1] = values[:, 0]
            values[:, 2:] = torch.minimum(values[:, 2:], values[:, :1])
        count = values.numel()//2
        actual = select_tree(values, count)
        assert actual.flatten().nonzero().flatten().tolist() == sorted_reference(values, count)
        assert torch.equal(actual[:, 0], actual[:, 1])
        assert not (actual[:, 2:] & ~actual[:, :1]).any()
    steps = 0
    mc = dict(layers=2, loops=4, kv_heads=2, query_heads=2, head_dim=128)
    for dtype in (torch.float32, torch.bfloat16):
        for cls in (JointCache, CappedVersionCache):
            original = [torch.randn(2, 4, 2, 4, 128, dtype=dtype) for _ in range(2)]
            scores = [torch.zeros(4, 2, 4, dtype=dtype), torch.full((4, 2, 4), 100., dtype=dtype)]
            source = dict(prompt_tokens=4, kv=original, scores=scores)
            cache = cls(source, mc, .25)
            cache.audit_source(source)
            if cls is JointCache:
                assert cache.states[0].values.shape[0] == 0
                assert cache.coverage()['local_any_loop_context'] < 1
            for step in range(6):
                before = cache.decisions()
                added = [torch.randn(2, 4, 2, 1, 128, dtype=dtype) for _ in range(2)]
                for loop in range(4):
                    for layer, state in enumerate(cache.states):
                        k, v = cache.update(added[layer][0, loop][None], added[layer][1, loop][None], loop*2+layer)
                        expected = torch.zeros(2, 2, cache.n, 128, dtype=dtype)
                        for component in range(2):
                            for head in range(2):
                                for token in range(cache.n):
                                    representative = int(before[layer, component, head, token, loop])
                                    if representative < 4:
                                        expected[component, head, token] = original[layer][component, representative, head, token]
                        assert torch.equal(k[0, :, :-1], expected[0]) and torch.equal(v[0, :, :-1], expected[1])
                        q = torch.randn(1, 2, 1, 128, dtype=dtype)
                        _, weights = cache.attention(fake_attention, SimpleNamespace(layer_idx=layer), q, k, v, None)
                        if cls is JointCache:
                            dead = torch.from_numpy(before[layer, 0, :, :, 3] == 4)
                            assert (weights[0, :, 0, :-1][dead] == 0).all()
                cache.finish()
                original = [torch.cat((old, new), dim=3) for old, new in zip(original, added)]
                after = cache.decisions()
                assert (after[..., :before.shape[-2], :] >= before).all()
                assert np.array_equal(after[:, 0, :, :, 3] < 4, after[:, 1, :, :, 3] < 4)
                cache.audit_source(dict(kv=original))
                assert cache.memory()['payload_ratio'] == ((cache.n+1)//2)/cache.n
                steps += 1
    # Validate the independent scoring audit on real tiny BF16 caches.
    for cls, method in [(JointCache, 'joint'), (CappedVersionCache, 'capped_version')]:
        cache = cls(source, mc, .25)
        data = cache.decisions()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            np.savez_compressed(path, initial=data, final=data)
            raw = dict(input_ids=list(range(4)), generated_tokens=1, prefill_cache=cache.memory(), final_cache=cache.memory(),
                byte_history=[], decisions_path=str(path), decisions_sha256=sha(path), prefill_coverage=cache.coverage(),
                final_coverage=cache.coverage(), decision_format='loop_source_versions')
            audit(raw, dict(context_ids=list(range(4))), method, mc)
    write(ROOT/'results/cpu_checks.json', dict(status='PASS', online_steps=steps, independent_sorted_reference_cases=21,
        root_pair_cutoff_repair=True, parent_child_closure=True, empty_layer_attention_mask=True, native_route_reconstruction=True,
        exact_current_length_global_budget=True, no_resurrection=True, joint_context_pruning_exercised=True,
        cap_only_control_retains_all_context=True, scope='CPU engineering fixtures; real-model quality unproven'))
    print('JOINT_CPU_CHECKS_PASS', steps)


if __name__ == '__main__':
    main()
