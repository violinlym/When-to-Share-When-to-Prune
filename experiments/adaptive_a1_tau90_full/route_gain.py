"""CPU reference allocator; exact local route gains, approximate global solution.

Each group is one (layer, K-or-V, token). Costs already include access weights
and any calibrated sensitivity. This module does not prescribe that proxy.
It never restores versions absent from the available mask. Root bit 8 must
exist in every live group. Six directed replacement costs per group are ordered
(0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
"""
import heapq
import math

PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
INDEX = {p: i for i, p in enumerate(PAIRS)}


def route_loss(cost, mask):
    if mask & 8 == 0:
        raise ValueError('A live group requires its last-loop root')
    loss = 0.0
    for r in range(3):
        s = next(s for s in range(r, 4) if mask & (1 << s))
        if s != r:
            loss += cost[INDEX[r, s]]
    return loss


def allocate(costs, available, live, extra_slots):
    """Greedy positive marginal gain, recomputed after each same-group update.

    Every extra version has the same native BF16 slot size. Caller MUST subtract
    roots and actual metadata bytes before passing extra_slots. This is not an
    exact global optimizer: multi-slot synergies can defeat one-step greedy.
    No gain is assumed monotone or submodular; non-positive additions are skipped.
    """
    if extra_slots < 0 or not (len(costs) == len(available) == len(live)):
        raise ValueError('Invalid budget or group shapes')
    masks = [8 if yes else 0 for yes in live]
    revisions = [0] * len(masks)
    heap = []

    def refresh(g):
        before = route_loss(costs[g], masks[g])
        for r in range(3):
            bit = 1 << r
            if available[g] & bit and not masks[g] & bit:
                gain = before - route_loss(costs[g], masks[g] | bit)
                if gain > 0:
                    heapq.heappush(heap, (-gain, g, r, revisions[g]))

    for g, yes in enumerate(live):
        if len(costs[g]) != 6 or any(not math.isfinite(x) or x < 0 for x in costs[g]):
            raise ValueError('Replacement costs must be six finite nonnegative values')
        if yes:
            if not available[g] & 8:
                raise ValueError('Cannot restore an unavailable root')
            refresh(g)
    used = 0
    while heap and used < extra_slots:
        neg_gain, g, r, revision = heapq.heappop(heap)
        if revision != revisions[g]:
            continue
        masks[g] |= 1 << r
        revisions[g] += 1
        used += 1
        refresh(g)
    assert all(m & a == m for m, a in zip(masks, available))
    return masks, {'extra_slots_used': used, 'extra_slots_budget': extra_slots,
                   'survivor_route_loss': sum(route_loss(c, m) for c, m in zip(costs, masks) if m)}
