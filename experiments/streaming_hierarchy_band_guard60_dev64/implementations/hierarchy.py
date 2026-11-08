"""Contiguous complete-linkage native-version hierarchy, constructed at birth.

This is a representation-distance heuristic, not attention-output error.
Three local adjacent-cluster merges build a nested tree once from four native
versions. No sharing configurations or counterfactual attention are enumerated.
All clusters read their latest native representative, preserving rightward
routing. Only three scalar costs and three native boundary IDs persist.
"""
import torch


def forward_distances(kv):
    """[2,4,H,N,D] -> [2,H,N,4,4], directed relative L2 for r<s."""
    values = kv.float().permute(0, 2, 3, 1, 4)
    earlier, later = torch.triu_indices(4, 4, offset=1, device=kv.device)
    distance = (values[..., later, :] - values[..., earlier, :]).norm(dim=-1)
    distance = distance / (values[..., earlier, :].norm(dim=-1) + 1e-8)
    matrix = torch.zeros((*values.shape[:-2], 4, 4), device=kv.device, dtype=torch.float32)
    matrix[..., earlier, later] = distance
    return matrix


def build_hierarchy(kv):
    """Return descending merge costs and corresponding optional native anchors.

Earlier entries are prerequisites for later refinements. Complete-linkage
diameters are nondecreasing along merges; ties use this reverse merge order,
not floating-point perturbation. When several boundaries are removed together,
the entire surviving cluster's diameter was already considered at construction.
"""
    matrix = forward_distances(kv)
    shape = matrix.shape[:-2]
    labels = torch.arange(4, device=kv.device).expand(*shape, 4).clone()
    boundaries = torch.arange(3, device=kv.device)
    removed, costs = [], []
    for _ in range(3):
        right = labels[..., 1:]
        members = (labels[..., None, :] == boundaries[:, None]) | (labels[..., None, :] == right[..., :, None])
        within = members[..., :, :, None] & members[..., :, None, :]
        candidate = matrix[..., None, :, :].masked_fill(~within, 0).amax(dim=(-2, -1))
        candidate = candidate.masked_fill(labels[..., :3] != boundaries, torch.inf)
        left = candidate.argmin(-1)
        value = candidate.gather(-1, left[..., None]).squeeze(-1)
        next_rep = labels.gather(-1, left[..., None] + 1)
        labels = torch.where(labels == left[..., None], next_rep, labels)
        removed.append(left.to(torch.uint8))
        costs.append(value)
    return torch.stack(costs, -1).flip(-1), torch.stack(removed, -1).flip(-1)


def hierarchy_scores(costs, order, available, importance):
    """Scores are laid out in coarse-to-fine order for deterministic tied cuts."""
    live = available[..., :3].gather(-1, order.long())
    return (costs.clamp_max(1) * importance[None, None, :, None]).masked_fill(~live, -torch.inf)


def native_mask(choices, order):
    optional = torch.zeros_like(choices)
    optional.scatter_(-1, order.long(), choices)
    return torch.cat((optional, torch.ones_like(optional[..., :1])), -1)
