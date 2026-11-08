"""Direct online boundary selection. No enumeration of sharing configurations.

Each K/V component has an ordered chain of native loop representatives. Loop4
is mandatory. Optional representatives compete once globally via scalar scores.
Read routing selects the nearest retained loop at or after the requested loop.
"""
import time
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from cache import H2OCache

ALLOCATION_METHODS = {'fts', 'residual', 'attention', 'combined'}


def relative_edges(kv):
    # [2,4,H,N,D] -> [2,H,N,3]. Only called on newly computed native KV.
    values = kv.float().permute(0, 2, 3, 1, 4)
    return ((values[..., 1:, :]-values[..., :-1, :]).norm(dim=-1)
            / (values[..., :-1, :].norm(dim=-1)+1e-8))


def threshold_keep(scores, count):
    """Exact cardinality with deterministic ties; one scalar order statistic."""
    flat = scores.reshape(-1)
    assert 0 <= count <= flat.numel()
    if count == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    threshold = flat.kthvalue(flat.numel()-count+1).values
    assert torch.isfinite(threshold), 'Insufficient available versions or nonfinite score'
    larger = flat > threshold
    tied = flat == threshold
    left = count-larger.sum()
    keep = larger | (tied & (tied.to(torch.int32).cumsum(0) <= left))
    return keep.reshape(scores.shape)


class StreamingLayer:
    def __init__(self, kv, scores, method):
        self.method = method
        self.H, self.n, self.D = kv.shape[2:]
        self.R = 4
        self.values = kv.permute(0, 2, 3, 1, 4).reshape(-1, self.D).contiguous()
        self.ptr = torch.arange(self.values.shape[0], device=kv.device, dtype=torch.int32).reshape(2, self.H, self.n, 4)
        self.drift = self.mass = None
        self.current = None
        self.attention_weights = []
        self.refresh_routes()

    def initialize_statistics(self, kv, scores):
        self.drift = relative_edges(kv) if self.method in ('residual', 'combined') else None
        self.mass = scores.float().mean(dim=(0, 1)) if self.method in ('attention', 'combined') else None

    def refresh_routes(self):
        loop_ids = torch.arange(4, device=self.ptr.device, dtype=torch.uint8).expand_as(self.ptr)
        available = loop_ids.masked_fill(self.ptr < 0, 4)
        self.loop_routes = available.flip((-1,)).cummin(-1).values.flip((-1,))
        self.routes = self.ptr.gather(-1, self.loop_routes.long())

    def optional_scores(self):
        if self.drift is not None:
            prefix = torch.cat((torch.zeros_like(self.drift[..., :1]), self.drift.cumsum(-1)), dim=-1)
            # Scalar accumulated change over the gap to the next surviving anchor.
            score = prefix.gather(-1, self.loop_routes[..., 1:].long())-prefix[..., :3]
        else:
            score = torch.ones_like(self.ptr[..., :3], dtype=torch.float32)
        if self.mass is not None:
            score = score*(self.mass/self.n)[None, None, :, None]
        return score.masked_fill(self.ptr[..., :3] < 0, -torch.inf)

    def compact(self, keep):
        # Numerical payload is physically compact; no zero-filled hidden copies.
        selected = self.ptr[keep].long()
        self.values = self.values.index_select(0, selected).contiguous()
        self.ptr = torch.full_like(self.ptr, -1)
        self.ptr[keep] = torch.arange(selected.numel(), device=self.ptr.device, dtype=self.ptr.dtype)
        self.refresh_routes()

    def history(self, loop):
        return self.values.index_select(0, self.routes[..., loop].reshape(-1).long()).reshape(2, self.H, self.n, self.D)

    def read_update(self, loop, k, v):
        if loop == 0:
            self.current = torch.empty((2, 4, self.H, 1, self.D), device=k.device, dtype=k.dtype)
            self.attention_weights = []
        self.current[0, loop].copy_(k[0])
        self.current[1, loop].copy_(v[0])
        out = torch.cat((self.history(loop), self.current[:, loop]), dim=2)
        return out[0][None], out[1][None]

    def observe(self, weights):
        if self.mass is not None:
            # Retain only this query's four native attention rows until finish.
            # No alternate attention/output computation is performed.
            self.attention_weights.append(weights[0, :, 0, :])

    def update_statistics(self):
        if self.drift is not None:
            self.drift = torch.cat((self.drift, relative_edges(self.current)), dim=2)
        if self.mass is not None:
            assert len(self.attention_weights) == 4
            delta = torch.stack(self.attention_weights).float().mean(dim=(0, 1))
            self.mass = torch.cat((self.mass, torch.zeros_like(self.mass[:1])))+delta
        self.attention_weights = []

    def append(self):
        added = self.current.permute(0, 2, 3, 1, 4).reshape(-1, self.D)
        indices = torch.arange(added.shape[0], device=self.ptr.device, dtype=torch.int32)+self.values.shape[0]
        self.ptr = torch.cat((self.ptr, indices.reshape(2, self.H, 1, 4)), dim=2)
        self.values = torch.cat((self.values, added), dim=0)
        self.n += 1
        self.current = None
        self.refresh_routes()

    def auxiliary(self):
        arrays = {'anchor_indices': self.ptr, 'read_indices': self.routes, 'loop_routes': self.loop_routes,
                  'scalar_residual_statistics': self.drift, 'scalar_attention_statistics': self.mass}
        return {name: 0 if value is None else value.numel()*value.element_size() for name, value in arrays.items()}


class StreamingCache(H2OCache):
    def __init__(self, source, model_cfg, method, ratio):
        DynamicCache.__init__(self)
        assert ratio in (.25, .4, .5) and method in ALLOCATION_METHODS
        self.L = model_cfg['layers']
        self.R = model_cfg['loops']
        self.H = model_cfg['kv_heads']
        self.groups = model_cfg['query_heads']//self.H
        assert self.R == 4
        self.layers = [DynamicLayer() for _ in range(self.L*self.R)]
        self.method, self.ratio = method, ratio
        self.n = self.prompt_tokens = source['prompt_tokens']
        self.calls = self.attention_calls = self.steps = 0
        self.device = source['kv'][0].device
        self.decision_seconds = 0.
        self.statistics_seconds = 0.
        self.synchronize()
        start = time.perf_counter()
        self.states = [StreamingLayer(kv, scores, method) for kv, scores in zip(source['kv'], source['scores'])]
        self.synchronize()
        statistics_start = time.perf_counter()
        for state, kv, scores in zip(self.states, source['kv'], source['scores']):
            state.initialize_statistics(kv, scores)
        self.synchronize()
        self.statistics_seconds += time.perf_counter()-statistics_start
        self.allocate()
        self.synchronize()
        self.allocation_seconds = time.perf_counter()-start

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def allocate(self):
        self.synchronize()
        decision_start = time.perf_counter()
        target = self.L*2*4*self.H*((self.n+1)//2)
        mandatory = self.L*2*self.H*self.n
        if sum(state.values.shape[0] for state in self.states) == target:
            # The common integer budget sometimes grows by a full native token.
            # Every available version fits, so selection and repacking are redundant.
            self.synchronize()
            self.decision_seconds += time.perf_counter()-decision_start
            return
        selections = []
        if self.method == 'fts':
            for state in self.states:
                keep = torch.zeros_like(state.ptr, dtype=torch.bool)
                keep[..., 0] = keep[..., 3] = True
                if self.n % 2:
                    keep[..., -1, :] = True
                selections.append(keep)
        else:
            scores = torch.stack([state.optional_scores() for state in self.states])
            optional = threshold_keep(scores, target-mandatory)
            for state, choice in zip(self.states, optional):
                keep = torch.cat((choice, torch.ones_like(choice[..., :1])), dim=-1)
                selections.append(keep)
        self.synchronize()
        self.decision_seconds += time.perf_counter()-decision_start
        for state, keep in zip(self.states, selections):
            state.compact(keep)
        assert sum(s.values.shape[0] for s in self.states) == target

    def attention(self, original, module, q, k, v, mask, *args, **kwargs):
        loop, layer = divmod(self.attention_calls % (self.L*self.R), self.L)
        assert layer == module.layer_idx and q.shape[-2] == 1
        out, weights = original(module, q, k, v, mask, *args, **kwargs)
        self.states[layer].observe(weights)
        self.attention_calls += 1
        return out, weights

    def finish(self):
        assert self.calls == self.attention_calls == (self.steps+1)*self.L*self.R
        self.synchronize()
        start = time.perf_counter()
        for state in self.states:
            state.update_statistics()
        self.synchronize()
        self.statistics_seconds += time.perf_counter()-start
        for state in self.states:
            state.append()
        self.n += 1
        self.steps += 1
        self.allocate()
        self.synchronize()
        self.allocation_seconds += time.perf_counter()-start

    def memory(self):
        payload = sum(s.values.numel()*s.values.element_size() for s in self.states)
        breakdown = {name: sum(s.auxiliary()[name] for s in self.states) for name in self.states[0].auxiliary()}
        full = self.L*2*4*self.H*self.n*self.states[0].D*self.states[0].values.element_size()
        if self.method not in ('hierarchy_version', 'hierarchy_price_exchange'):
            assert payload == full*((self.n+1)//2)//self.n
        assert all(s.current is None and s.n == self.n for s in self.states)
        return dict(logical_tokens=self.n, prompt_tokens=self.prompt_tokens, payload_bytes=payload,
            auxiliary_bytes=sum(breakdown.values()), auxiliary_breakdown_bytes=breakdown, full_payload_bytes=full,
            payload_ratio=payload/full, numerical_data_dtype='native_bfloat16', context_retention=1.,
            target_retention=getattr(self, 'target_retention', .5), allocation_scope='global_across_layer_KV_head_token', method=self.method,
            counterfactual_attention_forwards=0, sharing_configurations_enumerated=0)

    def coverage(self):
        return dict(local_any_loop_context=1., local_prompt_context=1., global_context_union=1., per_loop_context=1.)

    def decisions(self):
        return torch.stack([state.loop_routes for state in self.states]).cpu().numpy()

    def audit_source(self, source):
        for kv, state in zip(source['kv'], self.states):
            native = kv.permute(0, 2, 3, 1, 4)
            for loop in range(4):
                indices = state.loop_routes[..., loop, None, None].long().expand(2, self.H, self.n, 1, state.D)
                expected = native.gather(-2, indices).squeeze(-2)
                assert torch.equal(expected, state.history(loop)), 'Native representative route mismatch'
        self.memory()
