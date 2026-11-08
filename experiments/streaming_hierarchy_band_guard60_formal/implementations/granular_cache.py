"""Cheap head/loop attention statistics; unchanged native version selection.

No alternate attention, output-error evaluation, configuration enumeration,
rolling query window or data-specific coefficient is introduced.
"""
import time
import torch
from joint_cache import JointCache, JointLayer
from streaming_cache import relative_edges, threshold_keep
from exposure_cache import visible_importance


class GranularLayer(JointLayer):
    def initialize_statistics(self, kv, scores):
        self.drift = relative_edges(kv)
        self.mass = scores.float()  # loop, KV head, token
        assert self.mass.shape == (4, self.H, self.n)

    def update_statistics(self):
        self.drift = torch.cat((self.drift, relative_edges(self.current)), dim=2)
        assert len(self.attention_weights) == 4
        delta = torch.stack(self.attention_weights).float()
        assert delta.shape == (4, self.H, self.n + 1)
        self.mass = torch.cat((self.mass, torch.zeros_like(self.mass[..., :1])), dim=-1) + delta
        self.attention_weights = []


def granular_priorities(state, method):
    importance = visible_importance(state.mass)
    if method == 'head_version':
        importance = importance.mean(0, keepdim=True).expand_as(importance)
    else:
        assert method == 'head_loop_version'
    prefix = torch.cat((torch.zeros_like(state.drift[..., :1]), state.drift.cumsum(-1)), dim=-1)
    gap = prefix.gather(-1, state.loop_routes[..., 1:].clamp_max(3).long()) - prefix[..., :3]
    attention = importance[:3].permute(1, 2, 0)[None]
    # Same capped gap score as the existing causal-visibility version control.
    score = torch.minimum(gap * attention, attention)
    return score.masked_fill(state.ptr[..., :3] < 0, -torch.inf)


class GranularCache(JointCache):
    layer_class = GranularLayer

    def __init__(self, source, mc, ratio, method):
        assert method in ('head_version', 'head_loop_version')
        assert mc['query_heads'] == mc['kv_heads'], 'This Ouro pilot is MHA, not an unvalidated GQA mapping'
        self.granular_method = method
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
        priorities = torch.stack([granular_priorities(state, self.granular_method) for state in self.states])
        selected = threshold_keep(priorities, target - root_cost * self.n)
        masks = [torch.cat((row, torch.ones_like(row[..., :1])), dim=-1) for row in selected]
        self.synchronize()
        self.decision_seconds += time.perf_counter() - start
        for state, keep in zip(self.states, masks):
            state.compact(keep)
        assert sum(state.values.shape[0] for state in self.states) == target

    def memory(self):
        result = super().memory()
        result['context_root'] = 'Mandatory loop4 K/V for every token, layer and head; version-scoring ablation only'
        result['importance_granularity'] = self.granular_method
        result['importance_denominator'] = 'N - absolute token position; cumulative causally visible queries including self'
        return result
