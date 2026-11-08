"""Context admission by scalar benefit density; no sharing-pattern enumeration.

Root: one token's mandatory native last-loop K/V across layers/heads.
Children: its still-available additional native versions. Every child costs
one same-size KV component slot; the root costs C such slots.

All child utilities are cheap scalar proxies. This is a heuristic for the
integer global budget and says nothing by itself about downstream accuracy.
"""
import torch
from global_cache import select_global_roots
from streaming_cache import threshold_keep


def subtree_density(root_utility, children, root_cost):
    """Max root-plus-top-prefix density via one sort and prefix reduction.

children is [token, child]. Unavailable children are -inf. The root utility
already sums the benefits of its C base components; it is not a per-slot score.
For any cardinality, the highest-utility children form a sorted prefix, so no
subsets, sharing maps, model forwards or counterfactual outputs are constructed.
The exact maximum-density identity is distinct from the later budget heuristic.
"""
    assert root_cost > 0 and root_utility.ndim == 1 and children.ndim == 2
    assert root_utility.shape[0] == children.shape[0]
    ordered = children.sort(-1, descending=True).values
    sums = torch.cat((torch.zeros_like(root_utility[:, None]), ordered.cumsum(-1)), -1)
    costs = torch.arange(children.shape[-1]+1, device=children.device, dtype=children.dtype) + root_cost
    density = ((root_utility[:, None] + sums) / costs).amax(-1)
    return density


def select_density_contexts(root_utility, child_scores, root_cost, budget, protected=None):
    """Return global token roots and original-order optional version masks.

child_scores layout is [layer,kind,head,token,source_loop], preserving the
existing version-only allocator's tie order. Budget is native component slots.
The tested regime permits one base root for every currently live token.
"""
    assert child_scores.ndim == 5 and budget % root_cost == 0
    live = root_utility.isfinite()
    assert budget >= root_cost * int(live.sum()), 'Below one-slot-per-live-token is outside this prototype'
    per_token = child_scores.permute(3, 0, 1, 2, 4).reshape(root_utility.numel(), -1)
    density = subtree_density(root_utility, per_token, root_cost)
    if protected is not None:
        assert protected.shape == live.shape and protected.dtype == torch.bool
        assert live[protected].all(), 'Recent context must not have been evicted'
        assert int(protected.sum()) * root_cost < budget
        # Protect access only. Extra versions retain their original utilities.
        density = density.masked_fill(protected, torch.inf)
    capped = torch.minimum(child_scores, density[None, None, None, :, None])
    roots, _ = select_global_roots(density[:, None].expand(-1, root_cost), capped, budget)
    retained = roots[:, 0]
    if protected is not None:
        assert retained[protected].all()
    available = child_scores.masked_fill(~retained[None, None, None, :, None], -torch.inf)
    child_count = budget - int(retained.sum()) * root_cost
    # Density determines admission. Once roots are fixed, retain the original
    # highest-scoring available children. This recovers exact version-only
    # choices and tie order whenever no token is pruned.
    selected = threshold_keep(available, child_count)
    assert int(selected.sum()) + int(retained.sum()) * root_cost == budget
    return retained, selected, density
