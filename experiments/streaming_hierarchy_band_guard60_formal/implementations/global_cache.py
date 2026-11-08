"""Global token roots, with the capped-version allocator's canonical ties.

All root blocks precede all children. Children keep their original
layer/KV/head/token/loop order, including when repairing a split root.
"""
import time
import torch
from joint_cache import JointCache
from streaming_cache import threshold_keep


def select_global_roots(root_priorities, child_priorities, count):
    n, root_cost = root_priorities.shape
    assert count % root_cost == 0
    root_size = n * root_cost
    priorities = torch.cat((root_priorities.reshape(-1), child_priorities.reshape(-1)))
    selected = threshold_keep(priorities, count)
    roots = selected[:root_size].reshape(n, root_cost)
    children = selected[root_size:].reshape(child_priorities.shape)
    counts = roots.sum(-1)
    partial = (counts > 0) & (counts < root_cost)
    if partial.any():
        assert int(partial.sum()) == 1
        roots |= partial[:, None]
        child_count = count - int(roots.sum())
        # Keep highest children with original tie order. Dropping the earliest
        # equal-score children would break zero-pruning equivalence.
        children = threshold_keep(child_priorities.masked_fill(~children, -torch.inf), child_count)
    return roots, children


class GlobalContextCache(JointCache):
    def __init__(self, source, mc, ratio):
        super().__init__(source, mc, ratio)
        self.method = 'global_context_stable'

    def allocate(self):
        self.synchronize()
        started = time.perf_counter()
        root_cost = self.L * 2 * self.H
        target = root_cost * 4 * ((self.n + 1) // 2)
        if sum(state.values.shape[0] for state in self.states) == target:
            self.synchronize()
            self.decision_seconds += time.perf_counter() - started
            return
        importance = torch.stack([state.mass / state.n for state in self.states])
        root_priority = importance.amax(0)
        live = self.states[0].ptr[0, 0, :, 3] >= 0
        root_priority = root_priority.masked_fill(~live, -torch.inf)
        children = torch.stack([torch.minimum(state.optional_scores(), importance[layer][None, None, :, None])
                                for layer, state in enumerate(self.states)])
        roots, selected_children = select_global_roots(root_priority[:, None].expand(-1, root_cost), children, target)
        anchors = roots[:, 0][None, None, :, None].expand(2, self.H, -1, 1)
        masks = [torch.cat((selected_children[layer], anchors), dim=-1) for layer in range(self.L)]
        self.synchronize()
        self.decision_seconds += time.perf_counter() - started
        for state, keep in zip(self.states, masks):
            state.compact(keep)
        assert sum(state.values.shape[0] for state in self.states) == target

    def memory(self):
        result = super().memory()
        result['context_root'] = 'One token across all physical layers/KVheads; loop4 K/V anchors bundled, optional versions independent'
        result['root_priority'] = 'Maximum of per-layer head-and-loop-averaged attention importance'
        result['child_tie_order'] = 'Original capped-version layer/KV/head/token/loop order, also during root repair'
        return result
