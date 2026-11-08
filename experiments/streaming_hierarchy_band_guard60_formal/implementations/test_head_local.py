"""Independent attention/priority reference and native online-cache checks."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from head_local_recent_cache import HeadLocalRecentCache
from local_recent_cache import LocalRecentCache
from streaming_cache import relative_edges
from global_cache import select_global_roots
from test_joint import fake_attention


def scalar_selection(roots,children,budget):
    # Independent Python stable ordering plus atomic pair completion.
    rs=roots.numel();values=roots.reshape(-1).tolist()+children.reshape(-1).tolist()
    selected=set(sorted(range(len(values)),key=lambda i:(-values[i],i))[:budget])
    for p in range(0,rs,2):
        if (p in selected)!=(p+1 in selected):
            selected|={p,p+1}
            kept=[i for i in selected if i>=rs]
            selected.remove(sorted(kept,key=lambda i:(values[i],-i))[0])
    mask=torch.tensor([i in selected for i in range(len(values))])
    return mask[:rs].reshape_as(roots),mask[rs:].reshape_as(children)


def main():
    torch.set_num_threads(1);torch.manual_seed(1112);cases=0
    for i in range(80):
        root=torch.rand(2,2,10)
        if i%2:root=(root*4).floor()/4
        children=torch.rand(2,2,2,10,3)*root[:,None,:,:,None]
        if i%3==0:children=(children*4).floor()/4
        root[...,-2:]=torch.inf
        rp=root.reshape(-1,1).expand(-1,2)
        for budget in [80,120,160]:
            a,b=select_global_roots(rp,children,budget);ea,eb=scalar_selection(rp,children,budget)
            assert torch.equal(a,ea) and torch.equal(b,eb)
            retained=a[:,0].reshape(2,2,10)
            assert not (b.any(dim=(1,4)) & ~retained).any()
            cases+=1
    steps=0
    mc=dict(layers=2,loops=4,kv_heads=2,query_heads=2,head_dim=8)
    for dtype in [torch.float32,torch.bfloat16]:
        n=10;signs=torch.tensor([-1,1,-1,1],dtype=dtype)[None,:,None,None,None]
        values=[torch.randn(2,1,2,n,8,dtype=dtype)*signs for _ in range(2)]
        # Different heads AND layers prefer different tokens, recent roots low.
        a=torch.tensor([[50.,30,20,10,.001,.001,.001,.001,.001,.001],[.001,.001,.001,.001,50,30,20,10,.001,.001]])
        scores=[a[None].expand(4,-1,-1).to(dtype).clone(),a.flip(0)[None].expand(4,-1,-1).to(dtype).clone()]
        source=dict(prompt_tokens=n,kv=values,scores=scores)
        cache=HeadLocalRecentCache(source,mc,.25)
        expected_mass=[s.float().mean(0) for s in scores]
        alive=cache.decisions()[:,0,:,:,3]<4
        assert alive[...,-2:].all() and not alive.all()
        assert any(not np.array_equal(row[0],row[1]) for row in alive), 'Head averaging would hide this fixture'
        assert any(not np.array_equal(alive[0,h],alive[1,h]) for h in range(2)), 'Layer pooling would hide this fixture'
        # Low-importance protected roots need only K4+V4, not extra snapshots.
        assert (cache.decisions()[...,-2:,:]==3).all()
        for l,state in enumerate(cache.states):
            assert torch.equal(state.mass,expected_mass[l])
            assert torch.equal(state.drift,relative_edges(values[l]))
        cache.audit_source(source)
        for step in range(8):
            before=cache.decisions();added=[torch.randn(2,4,2,1,8,dtype=dtype) for _ in range(2)];weights=[[],[]]
            for loop in range(4):
              for layer in range(2):
                k,v=cache.update(added[layer][0,loop][None],added[layer][1,loop][None],loop*2+layer)
                expected=torch.zeros(2,2,cache.n,8,dtype=dtype)
                for kind in range(2):
                  for head in range(2):
                    for token in range(cache.n):
                      rep=int(before[layer,kind,head,token,loop])
                      if rep<4:expected[kind,head,token]=values[layer][kind,rep,head,token]
                assert torch.equal(k[0,:,:-1],expected[0]) and torch.equal(v[0,:,:-1],expected[1])
                _,w=cache.attention(fake_attention,SimpleNamespace(layer_idx=layer),torch.randn(1,2,1,8,dtype=dtype),k,v,None)
                dead=torch.from_numpy(before[layer,0,:,:,3]==4);assert (w[0,:,0,:-1][dead]==0).all()
                weights[layer].append(w[0,:,0].float())
            cache.finish();values=[torch.cat((old,new),dim=3) for old,new in zip(values,added)]
            for layer,state in enumerate(cache.states):
                expected_mass[layer]=torch.cat((expected_mass[layer],torch.zeros(2,1)),dim=1)+torch.stack(weights[layer]).mean(0)
                assert torch.equal(state.mass,expected_mass[layer]), 'Cumulative statistic mixed heads/layers'
                assert torch.equal(state.drift,relative_edges(values[layer])), 'Residual definition changed'
            after=cache.decisions()
            assert (after[...,:before.shape[-2],:]>=before).all()
            assert (after[...,-((cache.n+4)//5):,3]<4).all()
            assert np.array_equal(after[:,0,:,:,3]<4,after[:,1,:,:,3]<4)
            cache.audit_source(dict(kv=values));assert cache.memory()['payload_bytes']==8*4*((cache.n+1)//2)*8*values[0].element_size()
            steps+=1
    parity=0
    for n in [2,3,7,12]:
        kv=[torch.randn(2,4,2,n,8) for _ in range(2)]
        scores=[torch.arange(1,4*n+1).float().reshape(4,1,n).expand(-1,2,-1).clone() for _ in range(2)]
        source=dict(prompt_tokens=n,kv=kv,scores=scores)
        new=HeadLocalRecentCache(source,mc,.25);old=LocalRecentCache(source,mc,.25)
        assert np.array_equal(new.decisions(),old.decisions());parity+=1
    out=dict(status='PASS',independent_scalar_selection_cases=cases,native_online_steps=steps,
        uniform_head_pooled_equivalence_cases=parity,checks=['independent layer/head attention','same-head root and children',
        'unchanged residual and rightward native reads','zero attention to deleted roots','recent base-only protection',
        'atomic K4V4 root and global exact current-N budget','irreversibility','BF16 and FP32'])
    (Path(__file__).resolve().parents[1]/'results/head_local_cpu_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
