"""CPU specification, not a cycle simulator or an FP32-equivalent allocator.

Exact selection for dyadic integer scores M >> s. Children are stored in
coarse-to-fine prerequisite order. Immutable IDs break ties, parent first.
No heap, global sort, or materialized child-score array in the selector.
"""
from dataclasses import dataclass
from collections import Counter


@dataclass(frozen=True)
class Tree:
    tree_id: int
    shifts: tuple                 # None means exactly zero, not tiny positive
    alive: int                    # currently physical coarse-to-fine prefix
    lower: int                    # certified cumulative mass bounds
    upper: int
    current: int                  # hidden behind counted simulated SRAM read


def validate(trees, mass_bits):
    if not 1 <= mass_bits <= 64:
        raise ValueError('unsupported mass width')
    ids = [t.tree_id for t in trees]
    if ids != sorted(set(ids)):
        raise ValueError('tree IDs must be unique and in stable order')
    for t in trees:
        if len(t.shifts) != 3 or not 0 <= t.alive <= 3:
            raise ValueError('four-loop tree required')
        ss = [1000 if s is None else s for s in t.shifts]
        if ss != sorted(ss) or any(s < 0 or s > 15 for s in ss if s != 1000):
            raise ValueError('nonmonotone tree or invalid dyadic shift')
        if not 0 <= t.lower <= t.current <= t.upper < (1 << mass_bits):
            raise ValueError('invalid certificate or accumulator overflow')


def select(trees, budget, mass_bits=48):
    """Return retained (tree_id, child), exact cutoff and logical work counters.

    Certificates must bound CURRENT mass. They cannot be stale scores without
    a valid upper bound. Updating/constructing certificates is caller work.
    All IDs describe one frozen physical snapshot; pruning is not implemented.
    """
    validate(trees, mass_bits)
    total = sum(t.alive for t in trees)
    if not 0 <= budget <= total:
        raise ValueError('budget outside available optional slots')
    work = Counter(live_children=total)
    cache = {}

    def ge(t, child, threshold):
        work['child_tests'] += 1
        if threshold == 0:
            return True
        s = t.shifts[child]
        if s is None:
            return False
        # Widen or detect overflow in hardware; NEVER wrap this shift.
        bound = threshold << s
        work['threshold_shifts'] += 1
        if t.tree_id in cache:
            work['local_mass_reads'] += 1
            work['comparisons'] += 1
            return cache[t.tree_id] >= bound
        work['certificate_scalar_reads'] += 2
        work['comparisons'] += 1
        if t.lower >= bound:
            work['certificate_hits'] += 1
            return True
        work['comparisons'] += 1
        if t.upper < bound:
            work['certificate_hits'] += 1
            return False
        work['current_mass_reads'] += 1
        work['cache_writes'] += 1
        cache[t.tree_id] = t.current
        work['comparisons'] += 1
        return t.current >= bound

    def count_ge(threshold):
        work['count_passes'] += 1
        count = 0
        for t in trees:
            work['tree_metadata_reads'] += 1
            for j in range(t.alive):
                if ge(t, j, threshold):
                    count += 1
                    work['count_increments'] += 1
        return count

    if budget == 0 or budget == total:
        kept = [(t.tree_id, j) for t in trees for j in range(t.alive)] if budget else []
        return kept, None, dict(work)
    lo, hi = 0, 1 << mass_bits
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count_ge(mid) >= budget:
            lo = mid
        else:
            hi = mid
    # Emit all strictly better candidates, then stable ties. Hardware needs a
    # mask/count pass and a stable tie scan; count these separately.
    better, tied = [], []
    for t in trees:
        work['tree_metadata_reads'] += 1
        for j in range(t.alive):
            if ge(t, j, lo + 1):
                better.append((t.tree_id, j))
            elif ge(t, j, lo):
                tied.append((t.tree_id, j))
    kept = better + tied[:budget - len(better)]
    work['output_id_writes'] += len(kept)
    work['boundary_tie_candidates'] += len(tied)
    work['peak_cached_mass_entries'] = len(cache)
    return kept, lo, dict(work)


def oracle(trees, budget):
    """Test-only exhaustive sorting oracle for the SAME integer score."""
    scores = [(0 if t.shifts[j] is None else t.current >> t.shifts[j],
               t.tree_id, j) for t in trees for j in range(t.alive)]
    scores.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [(i, j) for _, i, j in scores[:budget]]


def rightward_mapping(anchors, retained_children):
    """Four native versions, no averaging, no resurrection."""
    if sorted(anchors) != [0, 1, 2] or not 0 <= retained_children <= 3:
        raise ValueError('anchors must permute optional versions 0,1,2')
    live = set(anchors[:retained_children]) | {3}
    return tuple(min(v for v in live if v >= r) for r in range(4))
