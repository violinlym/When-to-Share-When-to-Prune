"""Small independent oracle tests only; enumeration is never in the allocator."""
import itertools
import json
from pathlib import Path
import torch
from density_selection import subtree_density, select_density_contexts
from streaming_cache import threshold_keep


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1105)
    density_cases = 0
    for cost in [2, 4, 8]:
        for nchildren in range(7):
            for repeat in range(8):
                root = torch.rand(1, dtype=torch.float64) * cost
                child = torch.rand(1, nchildren, dtype=torch.float64)
                expected = max((float(root[0]) + sum(float(child[0,j]) for j in subset)) / (cost+len(subset))
                               for k in range(nchildren+1) for subset in itertools.combinations(range(nchildren), k))
                actual = subtree_density(root, child, cost)
                torch.testing.assert_close(actual, torch.tensor([expected], dtype=torch.float64), rtol=1e-12, atol=1e-12)
                density_cases += 1
    allocation_cases = zero_prune = 0
    for layers, heads in [(1,1),(2,2),(3,2)]:
        cost = layers * 2 * heads
        for n in [2,3,7,12]:
            for repeat in range(10):
                child = torch.rand(layers,2,heads,n,3)
                root = torch.rand(n)*cost
                budget = cost*4*((n+1)//2)
                keep, selected, density = select_density_contexts(root,child,cost,budget)
                assert (selected <= keep[None,None,None,:,None]).all()
                assert int(selected.sum())+int(keep.sum())*cost == budget
                # Independently retain the best available children after admission.
                flat = child.reshape(-1).tolist()
                available = keep[None,None,None,:,None].expand_as(child).reshape(-1).tolist()
                order = sorted([j for j,x in enumerate(available) if x], key=lambda j:(-flat[j],j))
                expect = torch.zeros(child.numel(),dtype=torch.bool)
                expect[order[:budget-int(keep.sum())*cost]] = True
                assert torch.equal(selected.reshape(-1), expect)
                if keep.all():
                    assert torch.equal(selected, threshold_keep(child,budget-n*cost))
                    zero_prune += 1
                allocation_cases += 1
    # A high-value child can justify its context even when base utility is low.
    child = torch.tensor([[[[[8.,0.,0.],[.1,.1,.1],[.1,.1,.1],[0.,0.,0.]]],
                           [[[0.,0.,0.],[.1,.1,.1],[.1,.1,.1],[0.,0.,0.]]]]])
    root = torch.tensor([.1,1.,1.,.01])
    keep, selected, density = select_density_contexts(root,child,2,8)
    assert keep[0] and selected[0,0,0,0,0] and not keep[-1]
    # An already deleted token can never reappear.
    dead_child = child.clone();dead_child[...,3,:] = -torch.inf
    dead_root = root.clone();dead_root[3] = -torch.inf
    keep, selected, _ = select_density_contexts(dead_root,dead_child,2,8)
    assert not keep[3] and not selected[...,3,:].any()
    # Audit a known integer-root rounding limitation, without requiring a bad
    # answer: future improved rounding may close this gap without failing tests.
    small = torch.zeros(1,2,1,2,3)
    small[0,0,0,0,0] = 8.
    utility = torch.tensor([2.,6.])
    keep, chosen, _ = select_density_contexts(utility,small,2,4)
    actual_objective = float(utility[keep].sum()+small[chosen].sum())
    oracle_objective = 10.  # root0 plus its value8 child and a zero-valued child
    assert actual_objective <= oracle_objective
    output = dict(status='PASS',density_oracle_cases=density_cases,allocation_cases=allocation_cases,
                  zero_pruning_original_tie_equivalence_cases=zero_prune,
                  root_child_admission_fixture=True,deleted_context_stays_unavailable=True,
                  integer_rounding_example=dict(actual_proxy_utility=actual_objective,
                      exact_best_proxy_utility=oracle_objective,independent_units='one artificial two-token fixture'),
                  scope='CPU scalar algorithm only; no model, accuracy or performance qualification')
    root_path=Path(__file__).resolve().parents[1]
    (root_path/'results').mkdir(exist_ok=True)
    (root_path/'results/cpu_checks.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
