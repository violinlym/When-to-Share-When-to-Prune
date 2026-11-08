"""One-pass shadow-price exchange between version refinements and token-root bands.

Version Tree remains the first-stage allocator.  A root can be removed only as
part of a fixed adjacent-layer band in the deep half.  Both layers must agree
that the token is outside the recent suffix and cheap under maximum observed
attention.  Removing a band is compared with the marginal excluded
Version-Tree child price, then the released slots are globally reassigned.
No sharing configuration or counterfactual attention output is enumerated.
"""
import time
import torch

from hierarchy import build_hierarchy, hierarchy_scores, native_mask
from hierarchy_cache import HierarchyCache, HierarchyLayer
from streaming_cache import threshold_keep


class PriceExchangeLayer(HierarchyLayer):
    def initialize_statistics(self, kv, scores, peak_scores=None):
        self.drift, self.anchor_order = build_hierarchy(kv)
        self.loop_head_mass = scores.float()
        self.loop_head_peak = (scores if peak_scores is None else peak_scores).float()
        self.mass = self.loop_head_mass.mean(dim=(0, 1))

    def update_statistics(self):
        costs, order = build_hierarchy(self.current)
        self.drift = torch.cat((self.drift, costs), dim=2)
        self.anchor_order = torch.cat((self.anchor_order, order), dim=2)
        assert len(self.attention_weights) == 4
        delta = torch.stack(self.attention_weights).float()
        self.loop_head_mass = torch.cat((self.loop_head_mass, torch.zeros_like(self.loop_head_mass[..., :1])), dim=-1) + delta
        self.loop_head_peak = torch.maximum(
            torch.cat((self.loop_head_peak, torch.zeros_like(self.loop_head_peak[..., :1])), dim=-1), delta)
        self.mass = self.loop_head_mass.mean(dim=(0, 1))
        self.attention_weights = []

    def root_importance(self):
        return self.loop_head_mass.amax(dim=(0, 1)) / self.n

    def root_peak(self):
        return self.loop_head_peak.amax(dim=(0, 1))

    def auxiliary(self):
        result = super().auxiliary()
        result['loop_head_attention_statistics'] = self.loop_head_mass.numel() * self.loop_head_mass.element_size()
        result['loop_head_peak_statistics'] = self.loop_head_peak.numel() * self.loop_head_peak.element_size()
        return result


class PriceExchangeCache(HierarchyCache):
    layer_class = PriceExchangeLayer

    def __init__(self, source, mc, ratio):
        self.deep_start = mc['layers'] // 2
        self.prune_cap_fraction = .10
        self.last_shadow_price = 0.
        self.last_pruned_roots = 0
        self.last_pruned_bands = 0
        super().__init__(source, mc, ratio)
        self.method = 'hierarchy_price_exchange'

    def allocate(self):
        self.synchronize()
        start = time.perf_counter()
        target = self.target_components()
        alive = torch.stack([state.ptr[0, ..., 3] >= 0 for state in self.states])
        # Root removal is coherent across all heads in a layer.
        assert all(torch.equal(row, row[:1].expand_as(row)) for row in alive)
        root_live = alive[:, 0]
        mandatory = int(root_live.sum()) * 2 * self.H
        scores = torch.stack([state.optional_scores() for state in self.states])
        baseline = threshold_keep(scores, target - mandatory)

        flat = scores.reshape(-1)
        unselected = flat.masked_fill(baseline.reshape(-1), -torch.inf)
        shadow = unselected.max()
        if not torch.isfinite(shadow):
            shadow = torch.tensor(0., device=scores.device)

        selected_value = scores.masked_fill(~baseline, 0).sum(dim=(1, 2, 4))
        selected_count = baseline.sum(dim=(1, 2, 4))
        root_value = torch.stack([state.root_importance() for state in self.states])
        root_peak = torch.stack([state.root_peak() for state in self.states])
        slots = 2 * self.H + selected_count
        loss_density = (2 * self.H * root_value + selected_value) / slots

        recent = (self.n + 4) // 5
        old_end = self.n - recent
        prune = torch.zeros_like(root_live)
        pruned_bands = 0
        # Fixed adjacent-layer bands avoid overlapping actions and scattered
        # single-layer holes. The band score is the exact slot-weighted proxy
        # density of its two roots, not a second attention simulation.
        for first in range(self.deep_start, self.L - 1, 2):
            pair = slice(first, first + 2)
            eligible = root_live[pair].all(dim=0)
            eligible[old_end:] = False
            cap = int(eligible.sum().item() * self.prune_cap_fraction)
            low_peak = torch.zeros_like(eligible)
            if cap:
                band_peak = root_peak[pair].amax(dim=0)
                low_peak = threshold_keep((-band_peak).masked_fill(~eligible, -torch.inf), cap)
            band_slots = slots[pair].sum(dim=0)
            band_loss = (loss_density[pair] * slots[pair]).sum(dim=0) / band_slots
            candidate = eligible & low_peak & (band_loss < shadow)
            count = min(cap, int(candidate.sum().item()))
            if count:
                chosen = threshold_keep((-band_loss).masked_fill(~candidate, -torch.inf), count)
                prune[pair] = chosen[None].expand(2, -1)
                pruned_bands += count

        survivors = root_live & ~prune
        mandatory = int(survivors.sum()) * 2 * self.H
        allowed = survivors[:, None, None, :, None]
        scores = scores.masked_fill(~allowed, -torch.inf)
        optional = threshold_keep(scores, target - mandatory)
        masks = []
        for layer, (choice, state) in enumerate(zip(optional, self.states)):
            keep = native_mask(choice, state.anchor_order)
            keep = keep & survivors[layer][None, None, :, None]
            masks.append(keep)

        self.synchronize()
        self.decision_seconds += time.perf_counter() - start
        for state, keep in zip(self.states, masks):
            state.compact(keep)
        self.last_shadow_price = float(shadow)
        self.last_pruned_roots = int((~survivors).sum())
        self.last_pruned_bands = pruned_bands
        assert sum(state.values.shape[0] for state in self.states) == target

    def memory(self):
        result = super().memory()
        result['method'] = self.method
        result['context_root'] = 'Layer-token root shared across heads; roots pruned only as fixed adjacent deep-layer bands; recent20% protected'
        result['allocation_rule'] = 'One-pass Version-Tree shadow-price exchange with adjacent-layer consensus bands; no configuration enumeration'
        result['pruning_safety'] = 'Both adjacent deep layers must agree on low cumulative-risk and bottom-10% band maximum observed attention'
        result['last_shadow_price'] = self.last_shadow_price
        result['pruned_layer_token_roots'] = self.last_pruned_roots
        result['pruned_adjacent_layer_bands'] = self.last_pruned_bands
        return result

    def audit_hierarchy(self):
        total_pruned = 0
        recent = (self.n + 4) // 5
        for layer, state in enumerate(self.states):
            alive = state.ptr[0, ..., 3] >= 0
            assert torch.equal(alive, alive[:1].expand_as(alive))
            alive_token = alive[0]
            if layer < self.deep_start:
                assert alive_token.all()
            assert alive_token[-recent:].all()
            total_pruned += int((~alive_token).sum())
            available = (state.ptr[..., :3] >= 0).gather(-1, state.anchor_order.long())
            live = alive_token[None, None, :, None]
            nested = available[..., :-1].to(torch.int8) >= available[..., 1:].to(torch.int8)
            assert (nested | ~live.expand_as(nested)).all()
            assert (state.ptr[..., 3] >= 0).eq(live[..., 0].expand_as(state.ptr[..., 3])).all()
            assert (state.drift[..., :-1] >= state.drift[..., 1:]).all()
        for first in range(self.deep_start, self.L - 1, 2):
            left = self.states[first].ptr[0, 0, :, 3] >= 0
            right = self.states[first + 1].ptr[0, 0, :, 3] >= 0
            assert torch.equal(left, right)
        return dict(status='PASS', logical_tokens=self.n, pruned_layer_token_roots=total_pruned,
                    deep_only=True, recent_twenty_percent_protected=True,
                    coherent_across_heads=True, adjacent_layer_band_consensus=True,
                    nested_available_anchors=True)
