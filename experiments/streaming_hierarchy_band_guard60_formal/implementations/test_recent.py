"""Exercise protected weak recent context, real old deletion and native reads."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from recent_cache import RecentDensityCache
from pooled_cache import PooledDensityCache
from joint_cache import CappedVersionCache
from density_selection import select_density_contexts
from test_joint import fake_attention


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1107)
    mc = dict(layers=2, loops=4, kv_heads=2, query_heads=2, head_dim=8)
    steps = drops = 0
    for dtype in [torch.float32, torch.bfloat16]:
        n = 10
        signs = torch.tensor([-1, 1, -1, 1], dtype=dtype)[None,:,None,None,None]
        values = [torch.randn(2,1,2,n,8,dtype=dtype)*signs for _ in range(2)]
        mass = torch.tensor([40.,10.,8.,6.,.001,.001,.001,.001,.001,.001]).expand(4,2,n).clone()
        source = dict(prompt_tokens=n,kv=values,scores=[mass.clone(),mass.clone()])
        old = PooledDensityCache(source,mc,.25)
        cache = RecentDensityCache(source,mc,.25)
        assert old.decisions()[..., -2:, 3].max() == 4, 'Fixture must expose recent eviction'
        assert (cache.decisions()[..., -2:, 3] < 4).all()
        assert (cache.decisions()[..., :-2, 3] == 4).any(), 'Old token pruning remains possible'
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
                                if rep < 4:
                                    expected[kind,head,token] = values[layer][kind,rep,head,token]
                    assert torch.equal(k[0,:,:-1],expected[0]) and torch.equal(v[0,:,:-1],expected[1])
                    query = torch.randn(1,2,1,8,dtype=dtype)
                    _,w = cache.attention(fake_attention,SimpleNamespace(layer_idx=layer),query,k,v,None)
                    dead = torch.from_numpy(before[layer,0,:,:,3]==4)
                    assert (w[0,:,0,:-1][dead]==0).all()
            cache.finish()
            values = [torch.cat((old,new),dim=3) for old,new in zip(values,added)]
            after = cache.decisions()
            assert (after[...,:before.shape[-2],:]>=before).all()
            assert (after[..., -((cache.n+4)//5):, 3] < 4).all()
            cache.audit_source(dict(kv=values))
            assert cache.memory()['payload_bytes']==8*4*((cache.n+1)//2)*8*values[0].element_size()
            steps += 1
    parity = 0
    for n in [2,3,7,12]:
        values=[torch.randn(2,1,2,n,8).expand(-1,4,-1,-1,-1).contiguous() for _ in range(2)]
        masses=[torch.rand(4,2,n)+1 for _ in range(2)]
        source=dict(prompt_tokens=n,kv=values,scores=masses)
        cache=RecentDensityCache(source,mc,.25);control=CappedVersionCache(source,mc,.25)
        assert np.array_equal(cache.decisions(), control.decisions())
        parity += 1
    # New optional parameter must preserve the old selector exactly when empty.
    root=torch.rand(10)+1
    child=torch.rand(2,2,2,10,3)
    a=select_density_contexts(root,child,8,160)
    b=select_density_contexts(root,child,8,160,torch.zeros(10,dtype=torch.bool))
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    out=dict(status='PASS',native_online_steps=steps,actual_pruning_fixtures=drops,
             no_pruning_capped_version_route_parities=parity,empty_protection_old_selector_parity=True,
             scope='CPU native read/mask/physical budget/recent protection checks, not model accuracy')
    (Path(__file__).resolve().parents[1]/'results/recent_cpu_checks.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':
    main()
