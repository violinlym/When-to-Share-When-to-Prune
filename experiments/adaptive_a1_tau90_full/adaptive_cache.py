"""Only A1 changes: per-layer 90% historical attention-mass roots.

The inherited relative six-pair costs, marginal route-gain allocator,
K/V independent version masks, rightward routes and storage are unchanged.
"""

import math

import torch

from sparse_cache import SparseCache as OriginalCache, F, SLOT
from route_gain import allocate


class AdaptiveCache(OriginalCache):
    tau = 0.9

    def __init__(self, native, model, raw, stats, codec, method='adaptive90'):
        self._prefill_n = native.get_seq_length()
        # Full precision *coverage* mass mirrors measure_tau80.py --tau 0.9.
        # Original BF16-reduced self.mass remains untouched for A2.
        self._prefill_coverage_mass = [
            torch.stack([stats[r, l]['coverage_mass'] for r in range(4)]).double().sum(0)
            for l in range(24)
        ]
        # One prefill query and one decode query have equal weight in the
        # historical average; the prefill contributes N query observations.
        self._coverage_cumulative = [x * self._prefill_n for x in self._prefill_coverage_mass]
        self.coverage_steps = 0
        self.coverage_violations = 0
        self.coverage_min = 1.0
        self.coverage_initial = None
        self.root_resurrections = 0
        super().__init__(native, model, raw, stats, codec, method)
        self.audit['policy'] = 'A1 tau=0.9 historical attention-mass coverage; A2 original six-pair marginal route-gain; A3 original rightward routing'
        self.audit['a1_threshold'] = self.tau
        self.audit['initial_attention_mass_coverage_by_layer'] = self.coverage_initial

    def select(self, available):
        device = available.device
        length = available.shape[-1]
        roots = torch.zeros((24, length), dtype=torch.bool, device=device)
        recent = max(1, math.ceil(.2 * length))
        coverages = []
        for layer in range(24):
            eligible = (available[layer, 0] & 8) != 0
            assert bool((eligible == ((available[layer, 1] & 8) != 0)).all())
            roots[layer, -recent:] = eligible[-recent:]
            # Decode uses only observed attention, with prefill weighted by
            # its actual N query positions. Deleted tokens keep past mass but
            # receive zero new mass and can never be selected again.
            mass = self._coverage_cumulative[layer]
            if len(mass) != length:
                raise AssertionError('A1 coverage-history length mismatch')
            total = mass.sum()
            assert bool(total > 0)
            score = self.peak[layer].float().masked_fill(~eligible | roots[layer], -torch.inf)
            order = torch.argsort(score, descending=True, stable=True)
            order = order[torch.isfinite(score[order])]
            current = mass[roots[layer]].sum()
            if bool(current < self.tau * total) and order.numel():
                cumulative = mass[order].cumsum(0)
                needed = int(torch.searchsorted(cumulative, (self.tau * total - current).clamp_min(0), right=False).item()) + 1
                roots[layer, order[:min(needed, order.numel())]] = True
            coverage = float((mass[roots[layer]].sum() / total).item())
            coverages.append(coverage)
            if coverage < self.tau - 1e-6:
                self.coverage_violations += 1
            self.coverage_min = min(self.coverage_min, coverage)
            # Available is the previous mask plus the new token: no resurrection.
            if not bool((roots[layer] <= eligible).all()):
                self.root_resurrections += 1
                raise AssertionError('Root resurrection')
        self.coverage_steps += 1
        if self.coverage_initial is None:
            self.coverage_initial = coverages
            assert min(coverages) >= self.tau - 1e-6

        # The following A2 block is byte-for-byte the allocation logic in the
        # original anchor75/sparse_cache.py, after its fixed-root A1 block.
        free = (F * length // 4 - int(roots.sum()) * 2 * SLOT) // SLOT
        assert free >= 0
        costs = (torch.stack(self.residual) * torch.stack(self.mass)[:, None, :, None]).reshape(-1, 6).cpu().tolist()
        masks, audit = allocate(costs, available.flatten().cpu().tolist(),
                                roots[:, None, :].expand(-1, 2, -1).flatten().cpu().tolist(), free)
        selected = torch.tensor(masks, dtype=torch.uint8, device=device).reshape_as(available)
        assert bool(((selected & available) == selected).all())
        return selected

    def finish(self):
        # Observe current, already-masked decode attention before BaseCache
        # clears it. A2 continues to update its own self.mass unchanged.
        for layer in range(24):
            observed = torch.stack([self.observed[r, layer] for r in range(4)]).double().mean((0,1))
            self._coverage_cumulative[layer] = torch.cat((self._coverage_cumulative[layer],
                                                          observed.new_zeros(1))) + observed
        super().finish()
