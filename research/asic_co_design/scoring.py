"""Integer sampled-L1 alternative; deliberately NOT the frozen L2 metric.

Fixed coordinates and shared integer scale across all four loops are inputs.
No reconstruction from sketches; native BF16 payload remains external.
"""


def dyadic_ceiling(numerator, denominator, max_shift=7):
    """ceil-to-dyadic clipped ratio, returns s for 2**(-s), or None for zero."""
    if numerator < 0 or denominator < 0 or not 0 <= max_shift <= 15:
        raise ValueError('invalid magnitude or precision')
    if numerator == 0:
        return None
    denominator = max(denominator, 1)
    s = 0
    while s < max_shift and (numerator << (s + 1)) <= denominator:
        s += 1
    return s


def sampled_tree(vectors, coordinates, max_shift=7):
    if len(vectors) != 4 or not vectors[0] or any(len(x) != len(vectors[0]) for x in vectors):
        raise ValueError('four equal-sized vectors required')
    if not coordinates or len(set(coordinates)) != len(coordinates):
        raise ValueError('nonempty unique coordinates required')
    if any(d < 0 or d >= len(vectors[0]) for d in coordinates):
        raise ValueError('coordinate out of range')
    if any(not isinstance(x, int) or abs(x) > 32767 for v in vectors for x in v):
        raise ValueError('requires signed integers with a common scale')
    # Compare sampled L1 ratios with dyadic thresholds using shifts.
    distances = {}
    for i in range(4):
        for j in range(i + 1, 4):
            num = sum(abs(vectors[j][d] - vectors[i][d]) for d in coordinates)
            den = max(1, sum(abs(vectors[i][d]) for d in coordinates))
            distances[i, j] = dyadic_ceiling(num, den, max_shift)
    # Dyadic costs are quantized BEFORE tree construction: extra ties can
    # change topology. This is an explicit experimental arm, not exact L2.
    def rank(s):
        return -1000 if s is None else -s
    clusters = [(i,) for i in range(4)]
    costs, anchors = [], []
    for _ in range(3):
        candidates = []
        for c in range(len(clusters) - 1):
            union = clusters[c] + clusters[c + 1]
            g = max((distances[i, j] for i in union for j in union if i < j), key=rank)
            candidates.append((rank(g), c, g))
        _, c, g = min(candidates, key=lambda x: (x[0], x[1]))
        anchors.append(max(clusters[c]))
        costs.append(g)
        clusters[c:c + 2] = [clusters[c] + clusters[c + 1]]
    return tuple(reversed(costs)), tuple(reversed(anchors))
