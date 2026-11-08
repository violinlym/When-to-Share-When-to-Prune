"""Parent-constrained global threshold: context roots and native KV versions.

Root availability requires one K and one V. Additional native snapshots are
independent children. No sharing configuration or attention output is simulated.
"""
import time
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from streaming_cache import StreamingLayer, StreamingCache, threshold_keep


def select_tree(priorities, count):
    # [...,8]: rootK,rootV,K1,K2,K3,V1,V2,V3. Parents precede tied children.
    keep = threshold_keep(priorities, count)
    partial = keep[..., 0] ^ keep[..., 1]
    if partial.any():
        assert int(partial.sum()) == 1, 'At most one root pair may straddle the cutoff'
        keep[..., 0] |= partial
        keep[..., 1] |= partial
        choices = priorities[..., 2:].masked_fill(~keep[..., 2:], torch.inf).reshape(-1)
        removed = choices.argmin()
        assert torch.isfinite(choices[removed]), 'Pair repair requires a selected optional version'
        optional = keep[..., 2:].contiguous().reshape(-1)
        optional[removed] = False
        keep[..., 2:] = optional.reshape(*keep.shape[:-1], 6)
    return keep


class JointLayer(StreamingLayer):
    def refresh_routes(self):
        ids = torch.arange(4, device=self.ptr.device, dtype=torch.uint8).expand_as(self.ptr)
        available = ids.masked_fill(self.ptr < 0, 4)
        self.loop_routes = available.flip((-1,)).cummin(-1).values.flip((-1,))
        self.routes = self.ptr.gather(-1, self.loop_routes.clamp_max(3).long())

    def history(self, loop):
        if self.values.shape[0] == 0:
            return torch.zeros((2, self.H, self.n, self.D), device=self.values.device, dtype=self.values.dtype)
        routes = self.routes[..., loop]
        out = self.values.index_select(0, routes.clamp_min(0).reshape(-1).long()).reshape(2, self.H, self.n, self.D)
        return out.masked_fill((routes < 0)[..., None], 0)

    def optional_scores(self):
        # Inactive contexts have route sentinel4; their entries remain unavailable.
        prefix = torch.cat((torch.zeros_like(self.drift[..., :1]), self.drift.cumsum(-1)), dim=-1)
        drift = prefix.gather(-1, self.loop_routes[..., 1:].clamp_max(3).long())-prefix[..., :3]
        scores = drift*(self.mass/self.n)[None, None, :, None]
        return scores.masked_fill(self.ptr[..., :3] < 0, -torch.inf)


class CappedVersionCache(StreamingCache):
    """Control: same ancestor-cap scores, but all contexts remain mandatory."""
    def __init__(self, source, mc, ratio):
        super().__init__(source, mc, 'combined', ratio)
        self.method = 'capped_version'

    def allocate(self):
        self.synchronize()
        start = time.perf_counter()
        target = self.L*2*4*self.H*((self.n+1)//2)
        if sum(s.values.shape[0] for s in self.states) == target:
            self.synchronize()
            self.decision_seconds += time.perf_counter()-start
            return
        scores = torch.stack([torch.minimum(s.optional_scores(), (s.mass/s.n)[None, None, :, None]) for s in self.states])
        choices = threshold_keep(scores, target-self.L*2*self.H*self.n)
        masks = [torch.cat((row, torch.ones_like(row[..., :1])), dim=-1) for row in choices]
        self.synchronize()
        self.decision_seconds += time.perf_counter()-start
        for state, mask in zip(self.states, masks):
            state.compact(mask)
        assert sum(s.values.shape[0] for s in self.states) == target


class JointCache(StreamingCache):
    layer_class = JointLayer
    def __init__(self, source, mc, ratio):
        DynamicCache.__init__(self)
        assert ratio in (.25, .4, .5) and mc['loops'] == 4
        self.L, self.R, self.H = mc['layers'], mc['loops'], mc['kv_heads']
        self.groups = mc['query_heads']//self.H
        self.layers = [DynamicLayer() for _ in range(self.L*self.R)]
        self.method, self.ratio = 'joint', ratio
        self.n = self.prompt_tokens = source['prompt_tokens']
        self.calls = self.attention_calls = self.steps = 0
        self.device = source['kv'][0].device
        self.decision_seconds = self.statistics_seconds = 0.
        self.synchronize()
        start = time.perf_counter()
        self.states = [self.layer_class(kv, score, 'combined') for kv, score in zip(source['kv'], source['scores'])]
        self.synchronize()
        stat_start = time.perf_counter()
        peaks = source.get('peak_scores')
        for index, (state, kv, score) in enumerate(zip(self.states, source['kv'], source['scores'])):
            if peaks is None:
                state.initialize_statistics(kv, score)
            else:
                state.initialize_statistics(kv, score, peaks[index])
        self.synchronize()
        self.statistics_seconds += time.perf_counter()-stat_start
        self.allocate()
        self.synchronize()
        self.allocation_seconds = time.perf_counter()-start

    def allocate(self):
        self.synchronize()
        start = time.perf_counter()
        target = self.L*2*4*self.H*((self.n+1)//2)
        if sum(s.values.shape[0] for s in self.states) == target:
            self.synchronize()
            self.decision_seconds += time.perf_counter()-start
            return
        priorities = []
        for state in self.states:
            root = (state.mass/state.n)[None, :].expand(self.H, -1)
            root = root.masked_fill(state.ptr[0, ..., 3] < 0, -torch.inf)
            children = torch.minimum(state.optional_scores(), root[None, :, :, None])
            priorities.append(torch.cat((root[..., None].expand(-1, -1, 2), children[0], children[1]), dim=-1))
        selected = select_tree(torch.stack(priorities), target)
        masks = [torch.stack((torch.cat((row[..., 2:5], row[..., 0:1]), dim=-1),
                              torch.cat((row[..., 5:8], row[..., 1:2]), dim=-1)), dim=0) for row in selected]
        self.synchronize()
        self.decision_seconds += time.perf_counter()-start
        for state, keep in zip(self.states, masks):
            state.compact(keep)
        assert sum(s.values.shape[0] for s in self.states) == target

    def attention(self, original, module, q, k, v, mask, *args, **kwargs):
        loop, layer = divmod(self.attention_calls % (self.L*self.R), self.L)
        assert layer == module.layer_idx and q.shape[-2] == 1
        state = self.states[layer]
        live = torch.cat((state.ptr[0, ..., 3] >= 0, torch.ones((self.H, 1), device=q.device, dtype=torch.bool)), dim=-1)
        live = live.repeat_interleave(self.groups, dim=0)
        extra = torch.zeros((1, live.shape[0], 1, self.n+1), device=q.device, dtype=q.dtype)
        if mask is not None:
            extra = extra+mask[:, :, :, :self.n+1]
        mask = extra.masked_fill(~live[None, :, None, :], torch.finfo(q.dtype).min)
        out, weights = original(module, q, k, v, mask, *args, **kwargs)
        state.observe(weights)
        self.attention_calls += 1
        return out, weights

    def memory(self):
        result = super().memory()
        result['context_retention'] = 'Measured by exact separate coverage audit, not assumed100%'
        result['context_root'] = 'Paired K/V loop4 availability; optional K/V versions independent'
        return result

    def coverage(self):
        live = torch.stack([s.ptr[0, ..., 3] >= 0 for s in self.states])
        counts = torch.stack((live.float().mean(), live[..., :self.prompt_tokens].float().mean(), live.any(dim=(0, 1)).float().mean())).cpu().tolist()
        return dict(local_any_loop_context=counts[0], local_prompt_context=counts[1], global_context_union=counts[2], per_loop_context=counts[0])

    def audit_source(self, source):
        for kv, state in zip(source['kv'], self.states):
            alive = state.ptr[..., 3] >= 0
            assert torch.equal(alive[0], alive[1]), 'K/V context availability differs'
            native = kv.permute(0, 2, 3, 1, 4)
            for loop in range(4):
                index = state.loop_routes[..., loop, None, None].clamp_max(3).long().expand(2, self.H, self.n, 1, state.D)
                expected = native.gather(-2, index).squeeze(-2).masked_fill(~alive[..., None], 0)
                assert torch.equal(expected, state.history(loop))
        self.memory()
