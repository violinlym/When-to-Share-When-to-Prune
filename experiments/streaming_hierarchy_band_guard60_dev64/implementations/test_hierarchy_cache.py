"""Native physical storage and irreversible nested cuts on an online trace."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from hierarchy_cache import HierarchyCache
from test_joint import fake_attention


def main():
    torch.set_num_threads(1)
    torch.manual_seed(1109)
    mc = dict(layers=2,loops=4,kv_heads=2,query_heads=2,head_dim=8)
    steps = 0
    for dtype in [torch.float32,torch.bfloat16]:
        values=[torch.randn(2,4,2,7,8,dtype=dtype) for _ in range(2)]
        source=dict(prompt_tokens=7,kv=values,scores=[torch.rand(4,2,7) for _ in range(2)])
        cache=HierarchyCache(source,mc,.4)
        cache.audit_source(source)
        for step in range(8):
            before=cache.decisions()
            added=[torch.randn(2,4,2,1,8,dtype=dtype) for _ in range(2)]
            for loop in range(4):
                for layer in range(2):
                    k,v=cache.update(added[layer][0,loop][None],added[layer][1,loop][None],loop*2+layer)
                    expected=torch.zeros(2,2,cache.n,8,dtype=dtype)
                    for kind in range(2):
                        for head in range(2):
                            for token in range(cache.n):
                                rep=int(before[layer,kind,head,token,loop])
                                expected[kind,head,token]=values[layer][kind,rep,head,token]
                    assert torch.equal(k[0,:,:-1],expected[0]) and torch.equal(v[0,:,:-1],expected[1])
                    q=torch.randn(1,2,1,8,dtype=dtype)
                    cache.attention(fake_attention,SimpleNamespace(layer_idx=layer),q,k,v,None)
            cache.finish()
            values=[torch.cat((old,new),dim=3) for old,new in zip(values,added)]
            after=cache.decisions()
            assert (after[...,:before.shape[-2],:]>=before).all()
            cache.audit_source(dict(kv=values))
            for state in cache.states:
                available=(state.ptr[...,:3]>=0).gather(-1,state.anchor_order.long())
                assert (available[...,:-1].to(torch.int8)>=available[...,1:].to(torch.int8)).all()
            mem=cache.memory()
            assert cache.audit_hierarchy()['status']=='PASS'
            full_components=mc['layers']*2*mc['loops']*mc['kv_heads']*cache.n
            expected_components=(2*full_components+4)//5
            assert mem['payload_bytes']==expected_components*mc['head_dim']*values[0].element_size()
            assert mem['target_retention']==.4
            assert mem['auxiliary_breakdown_bytes']['hierarchy_anchor_order']==8*cache.n*3
            assert cache.coverage()['global_context_union']==1.
            steps += 1
    result=dict(status='PASS',native_online_steps=steps,growing_physical_budget=True,
                current_length_payload_retention=.4,
                nested_cuts_no_resurrection=True,original_native_history_reference=True,
                metadata_includes_anchor_order=True,scope='CPU adapter only; no model accuracy or online device timing')
    (Path(__file__).resolve().parents[1]/'results/native_cache_cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
