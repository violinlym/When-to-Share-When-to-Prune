"""Independent scalar oracles for trees, executable actions and physical reads."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from hierarchy import forward_distances,build_hierarchy
from heap_allocator import allocate,masks
from joint_hierarchy_cache import JointHierarchyCache
from cache import H2OCache
from joint_cache import CappedVersionCache
from recent_cache import RecentDensityCache
from hierarchy_cache import HierarchyCache


def tree_reference(x):
    dist=np.zeros((4,4),np.float32)
    for i in range(4):
        for j in range(i+1,4):dist[i,j]=np.linalg.norm(x[j]-x[i])/(np.linalg.norm(x[i])+1e-8)
    clusters=[[i] for i in range(4)];costs=[];order=[]
    while len(clusters)>1:
        choices=[]
        for j in range(len(clusters)-1):
            union=clusters[j]+clusters[j+1]
            choices.append((max(float(dist[a,b]) for a in union for b in union if a<b),clusters[j][-1],j))
        cost,boundary,j=min(choices)
        costs.append(cost);order.append(boundary);clusters[j:j+2]=[clusters[j]+clusters[j+1]]
    return dist,np.array(costs),np.array(order)


def scan_reference(I,costs,stage,alive,n,lh,budget):
    stage=stage.copy();alive=alive.copy();actions=[];recent=(n+4)//5
    total=int(((8-stage.sum(1))*alive).sum())
    while total>budget:
        choices=[]
        for r in range(n*lh):
            if not alive[r]:continue
            for k in range(2):
                s=int(stage[r,k])
                if s<3:choices.append((float(I[r])*min(float(costs[r,k,s]),1),k,r))
            if r//lh<n-recent:
                count=8-int(stage[r].sum());remaining=2.
                for k in range(2):
                    for j in range(int(stage[r,k]),3):remaining+=min(float(costs[r,k,j]),1.)
                choices.append((float(I[r])*remaining/count,2,r))
        score,k,r=min(choices)
        if k==2:total-=8-int(stage[r].sum());alive[r]=0
        else:stage[r,k]+=1;total-=1
        actions.append((score,k,r))
    return stage,alive,total,actions


def native_attention(module,q,k,v,mask,*args,**kwargs):
    logits=q@k.transpose(-2,-1)/(q.shape[-1]**.5)
    if mask is not None:logits=logits+mask
    w=logits.float().softmax(-1).to(q.dtype)
    return w@v,w


def fixture(dtype,n=7):
    l,h,d=2,3,8
    kv=[torch.randn(2,4,h,n,d).to(dtype) for _ in range(l)]
    scores=[torch.rand(4,h,n).to(dtype) for _ in range(l)]
    return dict(kv=kv,scores=scores,prompt_tokens=n),dict(layers=l,loops=4,kv_heads=h,query_heads=h,head_dim=d)


def main():
    torch.set_num_threads(2);torch.manual_seed(107);rng=np.random.default_rng(107)
    tree_count=0
    for dtype in (torch.float32,torch.bfloat16):
        for seed in range(8):
            kv=torch.randn(2,4,3,7,8).to(dtype)
            if seed==0:kv.fill_(1)
            actual=forward_distances(kv).numpy();cost,order=build_hierarchy(kv)
            for kind in range(2):
                for head in range(3):
                    for token in range(7):
                        dist,c,o=tree_reference(kv[kind,:,head,token].float().numpy())
                        np.testing.assert_allclose(actual[kind,head,token],dist,rtol=3e-6,atol=2e-7)
                        np.testing.assert_allclose(cost[kind,head,token].flip(-1).numpy(),c,rtol=3e-6,atol=2e-7)
                        assert np.array_equal(order[kind,head,token].flip(-1).numpy(),o)
                        tree_count+=1
    heap_count=0
    for case in range(240):
        n=int(rng.integers(3,15));lh=int(rng.integers(1,5));m=n*lh
        I=rng.choice([0.,.01,.1,.4,1.],m).astype(np.float32)
        costs=np.sort(rng.choice([0.,.1,.2,.5,1.,2.],(m,2,3)).astype(np.float32),axis=-1)
        order=np.array([rng.permutation(3) for _ in range(m*2)],np.uint8).reshape(m,2,3)
        stage=rng.integers(0,4,(m,2),dtype=np.uint8);alive=rng.integers(0,2,m,dtype=np.uint8)
        recent=(n+4)//5;alive[-recent*lh:]=1
        before_stage=stage.copy();before_alive=alive.copy()
        initial=int(((8-stage.sum(1))*alive).sum());budget=int(rng.integers(2*recent*lh,initial+1))
        expected_s,expected_a,total,acts=scan_reference(I,costs,stage,alive,n,lh,budget)
        result=allocate(I,costs,order,stage,alive,n,lh,budget,examples=8*m)
        assert np.array_equal(stage,expected_s) and np.array_equal(alive,expected_a)
        assert result['remaining']==total<=budget and (stage>=before_stage).all() and (alive<=before_alive).all()
        assert [(a[2],int(a[1]),int(a[0])) for a in result['examples']]==acts
        keep=masks(order,stage,alive,n,1,lh)
        assert int(keep.sum())==total and keep[..., -recent:,3].all()
        assert np.array_equal(keep[:,0,...,3],keep[:,1,...,3])
        heap_count+=1
    updates=0
    for dtype in (torch.float32,torch.bfloat16):
        source,mc=fixture(dtype);cache=JointHierarchyCache(source,mc,.25)
        cache.audit_source(source);cache.audit_tree()
        previous=cache.decisions().copy()
        for step in range(8):
            for loop in range(4):
                for layer in range(mc['layers']):
                    k=torch.randn(1,mc['kv_heads'],1,mc['head_dim']).to(dtype);v=torch.randn_like(k);q=torch.randn_like(k)
                    actual_k,actual_v=cache.update(k,v,loop*mc['layers']+layer)
                    assert torch.equal(actual_k[:,:,-1:],k) and torch.equal(actual_v[:,:,-1:],v)
                    module=type('Module',(),{'layer_idx':layer})()
                    out,w=cache.attention(native_attention,module,q,actual_k,actual_v,None)
                    live=cache.states[layer].ptr[0,...,3]>=0
                    assert (w[0,:,0,:-1][~live]==0).all()
            cache.finish();cache.audit_tree();record=cache.memory()
            assert record['payload_bytes']<=record['full_payload_bytes']//2
            current=cache.decisions();assert (current[...,:previous.shape[-2],:]>=previous).all()
            previous=current.copy();updates+=1
        # Every head has its own cumulative statistic; no head/layer averaging.
        assert cache.states[0].mass.shape==(mc['kv_heads'],cache.n)
        assert not torch.equal(cache.states[0].mass[0],cache.states[0].mass[1])
        # No-compression adapter: exact native attention and KV on CPU.
        source,mc=fixture(dtype);full=JointHierarchyCache(source,mc,.25,retention=1.)
        for loop in range(4):
            for layer in range(mc['layers']):
                k=torch.randn(1,mc['kv_heads'],1,mc['head_dim']).to(dtype);v=torch.randn_like(k);q=torch.randn_like(k)
                expected_k=torch.cat((source['kv'][layer][0,loop][None],k),dim=2)
                expected_v=torch.cat((source['kv'][layer][1,loop][None],v),dim=2)
                k1,v1=full.update(k,v,loop*mc['layers']+layer)
                assert torch.equal(k1,expected_k) and torch.equal(v1,expected_v)
                module=type('Module',(),{'layer_idx':layer})()
                expected,w0=native_attention(module,q,expected_k,expected_v,None)
                actual,w1=full.attention(native_attention,module,q,k1,v1,None)
                assert torch.equal(actual,expected) and torch.equal(w0,w1)
        full.finish();assert full.memory()['payload_ratio']==1.
    # Changed integer budget, identical baseline definitions otherwise.
    for n in (6,7,8,9):
        source,mc=fixture(torch.bfloat16,n)
        for cls,args in [(H2OCache,('bound',.25)),(H2OCache,('loop',.25)),(CappedVersionCache,(.25,)),(RecentDensityCache,(.25,)),(HierarchyCache,(.25,))]:
            c=cls(source,mc,*args);assert c.memory()['payload_ratio']<=.5
    report=dict(status='PASS',distance_and_tree_components=tree_count,heap_reference_cases=heap_count,
                online_updates=updates,cpu_uncompressed_attention_bitwise_equal=True,
                native_model_forward='Will be audited for4steps on every authorized GPU worker; CPU attention is not a full-model audit',
                exact_rightward_reads=True,causal_head_statistics=True,physical_budget_le_half=True,
                root_pairing_and_recent_protection=True,irreversible=True)
    path=Path(__file__).resolve().parents[1]/'results/cpu_correctness.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
