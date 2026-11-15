import time
import torch
from transformers.cache_utils import DynamicCache,DynamicLayer
from h2o_reference import H2OCache,Layer,budget_at,select
from policies import snap_scores,rkv_scores,tri_scores


class PruningLayer(Layer):
    def __init__(self,kv,scores,queries,method,algorithm,stats,omega,ratio):
        self.algorithm=algorithm;self.stats=stats;self.omega=omega
        self.ratio=ratio
        self.window={'snapkv_d':128,'rkv':8,'triattention':0}[algorithm]
        self.queries=queries[...,-self.window:,:].clone() if self.window else None
        self.policy_seconds=0.;self.evictions=0
        super().__init__(kv,scores,method,0,0,self.ratio)

    def compress(self):
        capacity=sum(budget_at(self.n,self.ratio));size=self.payload.shape[3]
        self.recent=min(self.window,capacity);self.hh=capacity-self.recent
        assert size>=capacity, 'Capacity cannot require resurrection'
        if size==capacity:return
        t=time.perf_counter()
        k=self.payload[0]
        if self.algorithm=='triattention':
            score=tri_scores(k,self.stats,self.omega,self.n)
        elif self.hh==0:
            score=torch.zeros_like(self.scores[...,:size-self.recent])
        elif self.algorithm=='snapkv_d':
            score=snap_scores(self.queries,k,self.positions,self.n,self.recent)
        elif self.algorithm=='rkv':
            score=rkv_scores(self.queries,k,self.recent)
        else:raise ValueError(self.algorithm)
        if self.recent:
            score=torch.cat((score,torch.zeros_like(self.scores[...,:self.recent])),dim=-1)
        if self.method=='bound':score=score.sum(0,keepdim=True)
        idx=select(score,self.hh,self.recent)
        if self.method=='bound':idx=idx.expand(self.R,-1,-1)
        self.payload=self.payload.gather(3,idx[None,:,:,:,None].expand(2,-1,-1,-1,self.D)).contiguous()
        self.positions=self.positions.gather(2,idx)
        self.scores=self.scores.gather(2,idx)
        self.evictions+=size-capacity;self.policy_seconds+=time.perf_counter()-t

    def query(self,r,q):
        if self.window:
            if r==0:self.new_q=torch.empty((self.R,self.H,1,self.D),device=q.device,dtype=q.dtype)
            self.new_q[r].copy_(q[0])

    def finish(self):
        if self.window:
            self.queries=torch.cat((self.queries,self.new_q),dim=-2)[...,-self.window:,:].contiguous()
            self.new_q=None
        super().finish()


class PruningCache(H2OCache):
    def __init__(self,source,mc,condition,calibration):
        DynamicCache.__init__(self)
        self.L=mc['layers'];self.R=mc['loops'];self.H=mc['kv_heads'];self.groups=1
        assert mc['query_heads']==self.H
        self.layers=[DynamicLayer() for _ in range(self.L*self.R)]
        self.n=self.prompt_tokens=source['prompt_tokens']
        self.method=condition['method'];self.ratio=condition.get('ratio',.3)
        self.algorithm=condition['algorithm'];self.hh,self.recent=budget_at(self.n,self.ratio)
        self.states=[]
        for l in range(self.L):
            stats={name:calibration[name][l] for name in ('q_real','q_imag','q_abs')} if self.algorithm=='triattention' else None
            omega=calibration['omega'] if self.algorithm=='triattention' else None
            self.states.append(PruningLayer(source['kv'][l],source['scores'][l],source['queries'][l],self.method,self.algorithm,stats,omega,self.ratio))
        self.calls=self.attention_calls=self.steps=0

    def attention(self,original,module,q,k,v,mask,*args,**kwargs):
        r,l=divmod(self.attention_calls%(self.L*self.R),self.L)
        self.states[l].query(r,q)
        return super().attention(original,module,q,k,v,mask,*args,**kwargs)

    def memory(self):
        out=super().memory()
        query_bytes=sum(s.queries.numel()*s.queries.element_size() for s in self.states if s.queries is not None)
        out['query_buffer_bytes']=query_bytes;out['auxiliary_bytes']+=query_bytes
        out['recent_slots']=self.states[0].recent;out['hh_slots']=self.states[0].hh
        out['policy_host_enqueue_seconds']=sum(s.policy_seconds for s in self.states)
        out['algorithm']=self.algorithm
        return out
