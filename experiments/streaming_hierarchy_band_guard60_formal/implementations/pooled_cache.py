"""Joint admission using the earlier, unchanged capped-version scalar scorer.

The head/loop accuracy pilot was negative. This adapter intentionally keeps
its predecessor's per-layer head/loop-pooled mass/N and residual statistics.
Only context admission differs from the existing version-only control.
"""
import time
import torch
from joint_cache import JointCache
from density_selection import select_density_contexts


class PooledDensityCache(JointCache):
    def __init__(self, source, mc, ratio, base_method='pooled_version'):
        assert base_method == 'pooled_version'
        super().__init__(source, mc, ratio)
        self.method = 'pooled_density'

    def allocate(self):
        self.synchronize()
        start = time.perf_counter()
        root_cost = self.L * 2 * self.H
        target = root_cost * 4 * ((self.n + 1) // 2)
        if sum(state.values.shape[0] for state in self.states) == target:
            self.synchronize()
            self.decision_seconds += time.perf_counter() - start
            return
        importance = torch.stack([state.mass/state.n for state in self.states])
        # The per-layer importance is already averaged across heads and loops.
        root_utility = 2*self.H*importance.sum(0)
        live = self.states[0].ptr[0, 0, :, 3] >= 0
        root_utility = root_utility.masked_fill(~live, -torch.inf)
        children = torch.stack([torch.minimum(state.optional_scores(), importance[layer][None,None,:,None])
                                for layer,state in enumerate(self.states)])
        roots, selected, _ = select_density_contexts(root_utility, children, root_cost, target)
        anchors = roots[None,None,:,None].expand(2,self.H,-1,1)
        masks = [torch.cat((row,anchors),dim=-1) for row in selected]
        self.synchronize()
        self.decision_seconds += time.perf_counter()-start
        for state, keep in zip(self.states,masks):
            state.compact(keep)
        assert sum(state.values.shape[0] for state in self.states)==target

    def memory(self):
        result=super().memory()
        result['context_root']='Global token last-loop K/V base across all physical layers/heads'
        result['importance_denominator']='N; original head/loop-pooled capped-version scorer unchanged'
        result['context_admission']='Aggregate base usefulness plus optional version benefit density; original-score refill'
        result['base_version_control']='capped_version, identical to global_context_stable when no pruning'
        return result
