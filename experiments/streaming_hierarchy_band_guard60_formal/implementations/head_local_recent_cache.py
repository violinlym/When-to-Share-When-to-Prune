"""Requested local root / head-aware attention intervention only.

Residuals, path-gap accumulation, native tensors, routing and tie repair are
inherited unchanged. Loop averaging and the current-N denominator are retained.
"""
import time
import torch
from joint_cache import JointLayer,JointCache
from local_recent_cache import LocalRecentCache
from streaming_cache import relative_edges
from global_cache import select_global_roots


class HeadLocalLayer(JointLayer):
    def initialize_statistics(self,kv,scores):
        self.drift=relative_edges(kv)
        self.mass=scores.float().mean(dim=0)  # [head,token]; average loops only
        assert self.mass.shape==(self.H,self.n)

    def update_statistics(self):
        self.drift=torch.cat((self.drift,relative_edges(self.current)),dim=2)
        assert len(self.attention_weights)==4
        delta=torch.stack(self.attention_weights).float().mean(dim=0)
        assert delta.shape==(self.H,self.n+1)
        self.mass=torch.cat((self.mass,torch.zeros_like(self.mass[...,:1])),dim=-1)+delta
        self.attention_weights=[]

    def optional_scores(self):
        prefix=torch.cat((torch.zeros_like(self.drift[...,:1]),self.drift.cumsum(-1)),dim=-1)
        gap=prefix.gather(-1,self.loop_routes[...,1:].clamp_max(3).long())-prefix[...,:3]
        scores=gap*(self.mass/self.n)[None,:,:,None]
        return scores.masked_fill(self.ptr[...,:3]<0,-torch.inf)


class HeadLocalRecentCache(LocalRecentCache):
    layer_class=HeadLocalLayer

    def __init__(self,source,mc,ratio):
        assert mc['query_heads']==mc['kv_heads'], 'Ouro MHA; GQA mapping not changed implicitly'
        JointCache.__init__(self,source,mc,ratio)
        self.method='head_local_recent'

    def allocate(self):
        self.synchronize();start=time.perf_counter()
        target=self.L*2*2*self.H*self.n
        if sum(state.values.shape[0] for state in self.states)==target:
            self.synchronize();self.decision_seconds+=time.perf_counter()-start
            return
        importance=torch.stack([state.mass/state.n for state in self.states])
        assert importance.shape==(self.L,self.H,self.n)
        live=torch.stack([state.ptr[0,...,3]>=0 for state in self.states])
        recent=(self.n+4)//5
        assert live[...,-recent:].all(), 'Recent local roots cannot be resurrected'
        root=importance.masked_fill(~live,-torch.inf)
        root[...,-recent:]=torch.inf
        children=torch.stack([torch.minimum(state.optional_scores(),importance[layer][None,:,:,None])
                              for layer,state in enumerate(self.states)])
        # Utility2I / cost2 = I; each child utility<=I. Preserve root-first ties
        # and canonical layer/KV/head/token/loop child order from local_recent.
        roots,selected=select_global_roots(root.reshape(-1,1).expand(-1,2),children,target)
        retained=roots[:,0].reshape(self.L,self.H,self.n)
        assert retained[...,-recent:].all()
        assert not (selected.any(dim=(1,4)) & ~retained).any()
        masks=[torch.cat((row,retained[layer][None,...,None].expand(2,-1,-1,1)),dim=-1)
               for layer,row in enumerate(selected)]
        self.synchronize();self.decision_seconds+=time.perf_counter()-start
        for state,keep in zip(self.states,masks):state.compact(keep)
        assert sum(state.values.shape[0] for state in self.states)==target

    def memory(self):
        result=super().memory()
        result['importance_granularity']='physical_layer,KV_head,token; no averaging over heads'
        result['importance_denominator']='current logical N; unchanged'
        result['importance_loop_reduction']='mean over four native loop attention rows; unchanged'
        return result
