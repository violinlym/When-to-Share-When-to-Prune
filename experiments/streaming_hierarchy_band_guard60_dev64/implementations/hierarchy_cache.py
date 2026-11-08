"""Native physical cache adapter for the proposed version merge hierarchy."""
import time
import torch
from joint_cache import JointCache, JointLayer
from streaming_cache import threshold_keep
from hierarchy import build_hierarchy, hierarchy_scores, native_mask


class HierarchyLayer(JointLayer):
    def initialize_statistics(self, kv, scores, peak_scores=None):
        self.drift, self.anchor_order = build_hierarchy(kv)
        self.mass = scores.float().mean(dim=(0,1))

    def optional_scores(self):
        return hierarchy_scores(self.drift, self.anchor_order, self.ptr >= 0, self.mass/self.n)

    def update_statistics(self):
        costs, order = build_hierarchy(self.current)
        self.drift = torch.cat((self.drift, costs), dim=2)
        self.anchor_order = torch.cat((self.anchor_order, order), dim=2)
        assert len(self.attention_weights) == 4
        delta = torch.stack(self.attention_weights).float().mean(dim=(0,1))
        self.mass = torch.cat((self.mass, torch.zeros_like(self.mass[:1]))) + delta
        self.attention_weights = []

    def auxiliary(self):
        result = super().auxiliary()
        result['hierarchy_anchor_order'] = self.anchor_order.numel()*self.anchor_order.element_size()
        return result


class HierarchyCache(JointCache):
    layer_class = HierarchyLayer

    def __init__(self, source, mc, ratio):
        assert ratio in (.4, .5)
        self.target_retention = ratio
        super().__init__(source, mc, ratio)
        self.method = 'hierarchy_version'

    def target_components(self):
        full = self.L*2*4*self.H*self.n
        return (2*full + 4)//5 if self.target_retention == .4 else (full + 1)//2

    def allocate(self):
        self.synchronize()
        start = time.perf_counter()
        target = self.target_components()
        mandatory = self.L*2*self.H*self.n
        assert mandatory <= target <= self.L*2*4*self.H*self.n
        if sum(state.values.shape[0] for state in self.states) == target:
            self.synchronize()
            self.decision_seconds += time.perf_counter()-start
            return
        scores = torch.stack([state.optional_scores() for state in self.states])
        choices = threshold_keep(scores, target-mandatory)
        masks = [native_mask(row,state.anchor_order) for row,state in zip(choices,self.states)]
        self.synchronize()
        self.decision_seconds += time.perf_counter()-start
        for state,keep in zip(self.states,masks):
            state.compact(keep)
        assert sum(state.values.shape[0] for state in self.states) == target

    def memory(self):
        result = super().memory()
        vector_bytes = self.states[0].D*self.states[0].values.element_size()
        assert result['payload_bytes'] == self.target_components()*vector_bytes
        result['target_retention'] = self.target_retention
        result['budget_rule'] = 'ceil(target_retention * current logical Full-KV vector components)'
        result['hierarchy'] = 'Native contiguous complete-linkage tree built once per born token; scalar threshold across all components'
        result['context_root'] = 'All contexts retained in this version-only mechanism control'
        result['residual_statistics'] = 'Three capped-at-selection cluster diameters; no native vectors in metadata'
        return result

    def audit_hierarchy(self):
        # Called only at initial/final publication, not in each decode decision.
        for state in self.states:
            available = (state.ptr[..., :3] >= 0).gather(-1, state.anchor_order.long())
            assert (available[..., :-1].to(torch.int8) >= available[..., 1:].to(torch.int8)).all()
            assert (state.ptr[..., 3] >= 0).all()
            assert (state.drift[..., :-1] >= state.drift[..., 1:]).all()
            expected = torch.arange(3,device=state.ptr.device)
            assert (state.anchor_order.sort(-1).values == expected).all()
        return dict(status='PASS',logical_tokens=self.n,component_count=self.L*2*self.H*self.n,
                    all_context_retained=True,nested_available_anchors=True)
