"""Budget-aware SnapKV-D window adapter with a fixed 128-query horizon."""
import math

import torch
import torch.nn.functional as F


def window_scores(q, k, positions, n, recent, kernel=5):
    """Score surviving non-protected keys from recent causal queries.

    q/k: [loops, heads, query_or_key_positions, head_dim]
    positions: absolute positions of currently retained keys [loops, heads, keys]
    """
    observed = q.shape[-2]
    if observed != min(128, n):
        raise ValueError('Expected the latest min(128, current_length) query rows')
    qp = torch.arange(n - observed, n, device=k.device)
    eligible = positions[..., None, :] <= qp[:, None]
    valid = eligible.any(-1)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    logits = logits.masked_fill(~eligible, float('-inf'))
    # Irreversible pruning can leave an old query without any causal live key.
    # Such rows contribute zero; they never attend to future positions.
    logits = logits.masked_fill(~valid[..., None], 0)
    weights = logits.softmax(-1, dtype=torch.float32).to(q.dtype)
    weights = weights.masked_fill(~valid[..., None], 0)
    candidates = weights if recent == 0 else weights[..., :-recent]
    score = candidates.mean(-2)
    score = F.avg_pool1d(score.flatten(0, 1), kernel, stride=1,
                         padding=kernel // 2).reshape_as(score)
    return score, (~valid).sum(), valid.numel()
