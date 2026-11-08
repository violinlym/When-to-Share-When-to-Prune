"""Independent native-value history and masked-attention pruning fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from pooled_cache import PooledDensityCache as DensityCache
from joint_cache import CappedVersionCache
from test_joint import fake_attention


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1106)
    mc = dict(layers=2,loops=4,kv_heads=2,query_heads=2,head_dim=8)
    steps = drops = 0
    for dtype in [torch.float32,torch.bfloat16]:
        for method in ['pooled_version']:
            signs = torch.tensor([-1,1,-1,1],dtype=dtype)[None,:,None,None,None]
            values = [torch.randn(2,1,2,4,8,dtype=dtype)*signs for _ in range(2)]
            mass = (torch.tensor([40.,10.,.01,.001])*torch.arange(4,0,-1)).expand(4,2,4).clone()
            source = dict(prompt_tokens=4,kv=values,scores=[mass.clone(),mass.clone()])
            cache = DensityCache(source,mc,.25,method)
            assert cache.coverage()['global_context_union'] == .5
            drops += 1
            cache.audit_source(source)
            for step in range(8):
                before = cache.decisions()
                added = [torch.randn(2,4,2,1,8,dtype=dtype) for _ in range(2)]
                for loop in range(4):
                    for layer in range(2):
                        k,v = cache.update(added[layer][0,loop][None],added[layer][1,loop][None],loop*2+layer)
                        expected = torch.zeros(2,2,cache.n,8,dtype=dtype)
                        for kind in range(2):
                            for head in range(2):
                                for token in range(cache.n):
                                    rep = int(before[layer,kind,head,token,loop])
                                    if rep < 4:expected[kind,head,token]=values[layer][kind,rep,head,token]
                        assert torch.equal(k[0,:,:-1],expected[0]) and torch.equal(v[0,:,:-1],expected[1])
                        query = torch.randn(1,2,1,8,dtype=dtype)
                        _,w = cache.attention(fake_attention,SimpleNamespace(layer_idx=layer),query,k,v,None)
                        dead = torch.from_numpy(before[layer,0,:,:,3]==4)
                        assert (w[0,:,0,:-1][dead]==0).all()
                cache.finish()
                values = [torch.cat((old,new),dim=3) for old,new in zip(values,added)]
                after = cache.decisions()
                assert (after[...,:before.shape[-2],:]>=before).all()
                alive = after[...,3]<4
                assert (alive==alive[0,0,0]).all()
                cache.audit_source(dict(kv=values))
                mem = cache.memory()
                assert mem['payload_bytes']==8*4*((cache.n+1)//2)*8*values[0].element_size()
                steps += 1
    # Vanishing representation changes retain every context; exact control ties.
    parity = 0
    for method in ['pooled_version']:
        for n in [2,3,7,12]:
            values=[torch.randn(2,1,2,n,8).expand(-1,4,-1,-1,-1).contiguous() for _ in range(2)]
            masses=[torch.rand(4,2,n)+1 for _ in range(2)]
            source=dict(prompt_tokens=n,kv=values,scores=masses)
            cache=DensityCache(source,mc,.25,method);control=CappedVersionCache(source,mc,.25)
            assert cache.coverage()['global_context_union']==1.
            assert (cache.decisions()==control.decisions()).all()
            parity += 1
    result=dict(status='PASS',native_online_steps=steps,nonzero_pruning_fixtures=drops,
                no_pruning_original_route_equivalence_cases=parity,
                scope='CPU physical state/mask/byte checks only; no model quality or device timing claim')
    root=Path(__file__).resolve().parents[1]
    (root/'results/pooled_dense_cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
