"""Independent scalar agglomeration reference; no model-quality claim."""
import json
from pathlib import Path
import torch
from hierarchy import build_hierarchy, forward_distances, hierarchy_scores, native_mask


def scalar_reference(matrix):
    groups = [[0], [1], [2], [3]]
    costs, anchors = [], []
    while len(groups) > 1:
        proposals = []
        for i in range(len(groups)-1):
            union = groups[i] + groups[i+1]
            diameter = max(float(matrix[a, b]) for a in union for b in union)
            proposals.append((diameter, groups[i][-1], i))
        value, boundary, index = min(proposals)
        costs.append(value); anchors.append(boundary)
        groups[index:index+2] = [groups[index] + groups[index+1]]
    return list(reversed(costs)), list(reversed(anchors))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1108)
    cases = 0
    for dtype in [torch.float32, torch.bfloat16]:
        kv = torch.randn(2,4,2,9,8,dtype=dtype)
        # Degenerate equal versions and a gradual drift with small local gaps.
        kv[:,:,:,0] = kv[:,:1,:,0]
        for loop in range(4):
            kv[:,loop,:,1] = 10 + loop
        cost, order = build_hierarchy(kv)
        matrix = forward_distances(kv)
        assert (cost[..., :-1] >= cost[..., 1:]).all()
        assert (order.sort(-1).values == torch.arange(3)).all()
        for kind in range(2):
            for head in range(2):
                for token in range(9):
                    c,o = scalar_reference(matrix[kind,head,token])
                    assert torch.equal(order[kind,head,token].long(),torch.tensor(o))
                    assert torch.equal(cost[kind,head,token],torch.tensor(c))
                    cases += 1
        scores = hierarchy_scores(cost,order,torch.ones_like(cost[..., :1].expand(-1,-1,-1,4),dtype=torch.bool),torch.ones(9))
        assert (scores[..., :-1] >= scores[..., 1:]).all()
        for saved_optional in range(4):
            selected = torch.zeros_like(scores,dtype=torch.bool)
            selected[..., :saved_optional] = True
            keep = native_mask(selected,order)
            labels = torch.arange(4).expand_as(keep).masked_fill(~keep,4).flip(-1).cummin(-1).values.flip(-1)
            assert (labels>=torch.arange(4)).all() and (labels<4).all()
            for kind in range(2):
                for head in range(2):
                    for token in range(9):
                        actual=max(float(matrix[kind,head,token,loop,labels[kind,head,token,loop]]) for loop in range(4))
                        bound=0. if saved_optional==3 else float(cost[kind,head,token,saved_optional])
                        assert actual<=bound+1e-7
        # A simultaneous adjacent-gap selection can miss accumulated drift.
        # With one optional anchor, hierarchy distributes representatives into
        # two contiguous groups for [10,11,12,13]. This is only a toy proof.
        selected=torch.zeros_like(scores,dtype=torch.bool);selected[...,0]=True
        route=native_mask(selected,order)
        assert route[0,0,1].tolist()==[False,True,False,True]
    record=dict(status='PASS',independent_scalar_reference_cases=cases,
        monotone_merge_cost=True,nested_native_route_distance_bounds=True,
        retained_metadata='3 float32 scalar costs +3 uint8 anchor IDs per K/V component/token',
        scope='CPU prototype only. No GPU experiment, model output or accuracy result; no optimal-allocation claim.')
    (Path(__file__).resolve().parents[1]/'results/cpu_hierarchy_checks.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    main()
