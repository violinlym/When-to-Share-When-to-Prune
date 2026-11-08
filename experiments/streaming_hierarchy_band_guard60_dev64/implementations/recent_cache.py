"""Single intervention: recent context roots remain accessible, versions compete."""
import time
import torch
from pooled_cache import PooledDensityCache
from density_selection import select_density_contexts


class RecentDensityCache(PooledDensityCache):
    def __init__(self, source, mc, ratio):
        super().__init__(source, mc, ratio)
        self.method = 'recent_density'

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
        root_utility = 2*self.H*importance.sum(0)
        live = self.states[0].ptr[0, 0, :, 3] >= 0
        root_utility = root_utility.masked_fill(~live, -torch.inf)
        children = torch.stack([torch.minimum(state.optional_scores(), importance[layer][None,None,:,None])
                                for layer,state in enumerate(self.states)])
        recent = (self.n + 4) // 5
        protected = torch.arange(self.n, device=self.device) >= self.n-recent
        roots, selected, _ = select_density_contexts(root_utility, children, root_cost, target, protected)
        anchors = roots[None,None,:,None].expand(2,self.H,-1,1)
        masks = [torch.cat((row,anchors),dim=-1) for row in selected]
        self.synchronize()
        self.decision_seconds += time.perf_counter()-start
        for state, keep in zip(self.states,masks):
            state.compact(keep)
        assert sum(state.values.shape[0] for state in self.states)==target

    def memory(self):
        result = super().memory()
        result['recent_context_protection'] = dict(count=(self.n+4)//5,
            fraction=.2, scope='Base access only; additional native versions still compete globally')
        return result
