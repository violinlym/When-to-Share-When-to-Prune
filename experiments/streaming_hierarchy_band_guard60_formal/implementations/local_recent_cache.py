"""Layer/head-local context roots under one global current-length budget."""
import time
import torch
from joint_cache import JointCache
from global_cache import select_global_roots


class LocalRecentCache(JointCache):
    def __init__(self,source,mc,ratio):
        super().__init__(source,mc,ratio)
        self.method='local_recent'

    def allocate(self):
        self.synchronize()
        start=time.perf_counter()
        target=self.L*2*2*self.H*self.n
        if sum(state.values.shape[0] for state in self.states)==target:
            self.synchronize()
            self.decision_seconds+=time.perf_counter()-start
            return
        importance=torch.stack([state.mass/state.n for state in self.states])
        root=importance[:,None,:].expand(-1,self.H,-1)
        live=torch.stack([state.ptr[0,...,3]>=0 for state in self.states])
        recent=(self.n+4)//5
        assert live[...,-recent:].all(), 'Recent base access must remain available'
        root=root.masked_fill(~live,-torch.inf)
        root[...,-recent:]=torch.inf
        # Root costs two slots and has utility2I. Since each child utility<=I,
        # maximum root-plus-prefix density is exactlyI, without a prefix sort.
        children=torch.stack([torch.minimum(state.optional_scores(),importance[layer][None,None,:,None])
                              for layer,state in enumerate(self.states)])
        roots,selected=select_global_roots(root.reshape(-1,1).expand(-1,2),children,target)
        retained=roots[:,0].reshape(self.L,self.H,self.n)
        assert retained[...,-recent:].all()
        assert not (selected.any(dim=(1,4)) & ~retained).any(), 'Optional version without its local context root'
        masks=[torch.cat((row,retained[layer][None,...,None].expand(2,-1,-1,1)),dim=-1)
               for layer,row in enumerate(selected)]
        self.synchronize()
        self.decision_seconds+=time.perf_counter()-start
        for state,keep in zip(self.states,masks):
            state.compact(keep)
        assert sum(state.values.shape[0] for state in self.states)==target

    def memory(self):
        result=super().memory()
        result['context_root']='Last-loop nativeK+V at one layer/head/token; independent roots acrosslayer/head'
        result['root_density']='I, exact scalar density for2Ibase utility/cost2 and children<=I'
        result['recent_context_protection']=dict(count=(self.n+4)//5,fraction=.2,scope='Minimum K/V access in every layer/head; extra versions compete')
        result['allocation_scope']='One global payload budget acrosslayers/KV/heads/tokens; no layer quota'
        return result
