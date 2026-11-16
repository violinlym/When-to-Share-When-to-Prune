"""Pure protocol helpers for layer-wise context loss profiling."""

import math
import numpy as np
import torch


def select_deleted(peak, ratio=0.25, recent_fraction=0.20):
    """Stable lowest-peak selection outside the protected recent window."""
    peak = np.asarray(peak, dtype=np.float64)
    n = len(peak)
    protected = math.ceil(n * recent_fraction)
    eligible = n - protected
    count = min(eligible, math.floor(n * ratio + 0.5))
    order = np.lexsort((np.arange(eligible), peak[:eligible]))
    deleted = np.sort(order[:count]).astype(np.int64)
    assert len(deleted) == count and np.all(deleted < eligible)
    return deleted


def masked_attention(original, target_layer, deleted, prompt_len, calls, audit):
    """Wrap Ouro eager attention. Cache tensors and positions remain full length."""
    deleted = torch.as_tensor(deleted, dtype=torch.long)

    def attention(module, query, key, value, attention_mask, *args, **kwargs):
        call = calls[0]
        calls[0] += 1
        loop, physical = divmod(call, 24)
        if physical != module.layer_idx or loop >= 4:
            raise AssertionError('Unexpected Ouro layer/loop call order')
        if physical != target_layer or deleted.numel() == 0:
            return original(module, query, key, value, attention_mask, *args, **kwargs)
        if key.shape[-2] < prompt_len or deleted.max().item() >= prompt_len:
            raise AssertionError('Historical token or position invariant failed')
        # A separate additive attention mask blocks every head and every query.
        # Keeping the full KV tensor avoids changing RoPE/cache positions.
        if attention_mask is None:
            local_mask = torch.zeros((1, 1, query.shape[-2], key.shape[-2]), device=query.device, dtype=query.dtype)
        else:
            local_mask = attention_mask[..., :key.shape[-2]].clone()
        local_mask[..., deleted.to(query.device)] = torch.finfo(query.dtype).min
        out, weights = original(module, query, key, value, local_mask, *args, **kwargs)
        if torch.any(weights[..., deleted.to(query.device)] != 0):
            raise AssertionError('Pruned token retained nonzero attention probability')
        audit.add(loop)
        return out, weights

    return attention


def output_kl(full_logits, pruned_logits):
    p = torch.log_softmax(full_logits.float(), dim=-1)
    q = torch.log_softmax(pruned_logits.float(), dim=-1)
    return float(torch.sum(torch.exp(p) * (p - q)).clamp_min(0).item())
