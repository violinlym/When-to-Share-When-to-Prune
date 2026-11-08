"""Training ablation: attention per causally visible query, not per all queries.

No future importance estimate, window, configuration enumeration, extra forward,
or change to the relative-residual / native routing rule is introduced.
"""
import time
import torch
from joint_cache import JointCache
from global_cache import select_global_roots
from streaming_cache import threshold_keep


def visible_importance(mass):
    n = mass.shape[-1]
    exposure = torch.arange(n, 0, -1, device=mass.device, dtype=torch.float32)
    return mass / exposure


def child_priorities(states, importance):
    children = []
    for layer, state in enumerate(states):
        prefix = torch.cat((torch.zeros_like(state.drift[..., :1]), state.drift.cumsum(-1)), dim=-1)
        gap = prefix.gather(-1, state.loop_routes[..., 1:].clamp_max(3).long()) - prefix[..., :3]
        attention = importance[layer][None, None, :, None]
        score = torch.minimum(gap * attention, attention)
        children.append(score.masked_fill(state.ptr[..., :3] < 0, -torch.inf))
    return torch.stack(children)


class ExposureCache(JointCache):
    def __init__(self, source, mc, ratio, method):
        assert method in ('exposure_version', 'exposure_global')
        self.exposure_method = method
        super().__init__(source, mc, ratio)
        self.method = method

    def allocate(self):
        self.synchronize()
        start = time.perf_counter()
        root_cost = self.L * 2 * self.H
        target = root_cost * 4 * ((self.n + 1) // 2)
        if sum(state.values.shape[0] for state in self.states) == target:
            self.synchronize()
            self.decision_seconds += time.perf_counter() - start
            return
        importance = visible_importance(torch.stack([state.mass for state in self.states]))
        children = child_priorities(self.states, importance)
        if self.exposure_method == 'exposure_version':
            selected = threshold_keep(children, target - root_cost * self.n)
            anchors = torch.ones_like(selected[0, ..., :1])
        else:
            root_priority = importance.amax(0)
            live = self.states[0].ptr[0, 0, :, 3] >= 0
            root_priority = root_priority.masked_fill(~live, -torch.inf)
            roots, selected = select_global_roots(root_priority[:, None].expand(-1, root_cost), children, target)
            anchors = roots[:, 0][None, None, :, None].expand(2, self.H, -1, 1)
        masks = [torch.cat((selected[layer], anchors), dim=-1) for layer in range(self.L)]
        self.synchronize()
        self.decision_seconds += time.perf_counter() - start
        for state, keep in zip(self.states, masks):
            state.compact(keep)
        assert sum(state.values.shape[0] for state in self.states) == target

    def memory(self):
        result = super().memory()
        result['importance_denominator'] = 'N - absolute token position; causal visible queries including self'
        result['context_root'] = ('Mandatory loop4 K/V across all layers and heads' if self.exposure_method == 'exposure_version'
                                  else 'Global token loop4 K/V bundle; independent optional native K/V children')
        result['root_priority'] = 'Maximum across layer visible-query attention means; no fixed layer or K/V quotas'
        return result
