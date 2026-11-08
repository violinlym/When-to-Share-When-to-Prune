"""Native KV payload compaction; dense masked reads preserve absolute positions.

Official real-drop selection: top HH outside the recent suffix, then chronological
order; accumulate attention to existing keys. Ouro bound selection shares the
decision across loops. Qwen GQA shares it within each native KV head.
"""
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


def select(scores, hh, recent):
    size = scores.shape[-1]
    if size <= hh + recent:
        return torch.arange(size, device=scores.device).expand(*scores.shape[:-1], size)
    eligible = scores[..., :size-recent] if recent else scores
    top = eligible.topk(hh, dim=-1).indices.sort(dim=-1).values
    tail = torch.arange(size-recent, size, device=scores.device).expand(*scores.shape[:-1], recent)
    return torch.cat((top, tail), dim=-1)


def budget_at(n, ratio):
    if ratio is None: return n, 0
    from fractions import Fraction
    keep = 2*Fraction(str(ratio))
    capacity = (n*keep.numerator+keep.denominator-1)//keep.denominator
    recent = capacity//2
    return capacity-recent, recent


class Layer:
    def __init__(self, kv, scores, method, hh, recent, ratio=None):
        # kv [2,R,H,N,D], scores [R,H,N]. Original prefill tensors never mutated.
        self.method = method; self.hh = hh; self.recent = recent; self.ratio = ratio
        _, self.R, self.H, self.n, self.D = kv.shape
        self.payload = kv
        self.scores = scores.clone()
        self.positions = torch.arange(self.n, device=kv.device).expand(self.R,self.H,self.n).clone()
        self.current = None; self.delta = None
        self.compress()

    def compress(self):
        if self.method == 'full': return
        if self.ratio is not None: self.hh,self.recent = budget_at(self.n,self.ratio)
        assert self.payload.shape[3] >= self.hh+self.recent, 'Cannot resurrect evicted tokens'
        rank = self.scores.sum(dim=0, keepdim=True) if self.method == 'bound' else self.scores
        idx = select(rank, self.hh, self.recent)
        if self.method == 'bound': idx = idx.expand(self.R,-1,-1)
        self.payload = self.payload.gather(3,idx[None,:,:,:,None].expand(2,-1,-1,-1,self.D)).contiguous()
        self.scores = self.scores.gather(2,idx)
        self.positions = self.positions.gather(2,idx)

    def read_update(self, r, k, v):
        if r == 0:
            self.current = torch.empty((2,self.R,self.H,1,self.D), device=k.device, dtype=k.dtype)
            self.delta = torch.empty((self.R,self.H,self.positions.shape[-1]+1),device=k.device,dtype=k.dtype)
        self.current[0,r].copy_(k[0]); self.current[1,r].copy_(v[0])
        # Full-size temporary operands preserve native absolute positions and
        # the same eager numerical shape. Deleted payload does not persist.
        out = torch.zeros((2,self.H,self.n+1,self.D),device=k.device,dtype=k.dtype)
        idx = self.positions[r][None,:,:,None].expand(2,-1,-1,self.D)
        out.scatter_(2,idx,self.payload[:,r])
        out[:,:,-1:].copy_(self.current[:,r])
        return out[0][None],out[1][None]

    def observe(self, r, weights, groups):
        # Native dtype retained, as in upstream sum(0).sum(1).
        a = weights.sum(0).sum(1).reshape(self.H,groups,self.n+1).mean(dim=1)
        self.delta[r,:,:-1] = a.gather(1,self.positions[r])
        self.delta[r,:,-1] = a[:,-1]

    def finish(self):
        self.payload = torch.cat((self.payload,self.current),dim=3)
        self.scores = torch.cat((self.scores,torch.zeros_like(self.scores[...,:1])),dim=2) + self.delta
        self.positions = torch.cat((self.positions,torch.full_like(self.positions[...,:1],self.n)),dim=2)
        self.current = self.delta = None
        self.n += 1; self.compress()
        assert self.payload.shape[3] == (self.n if self.method=='full' else min(self.n,self.hh+self.recent))


class H2OCache(DynamicCache):
    def __init__(self, source, model_cfg, method, ratio):
        super().__init__()
        self.L = model_cfg['layers']; self.R = model_cfg['loops']
        self.H = model_cfg['kv_heads']; self.groups = model_cfg['query_heads']//self.H
        self.layers = [DynamicLayer() for _ in range(self.L*self.R)]
        self.n = source['prompt_tokens']; self.prompt_tokens = self.n
        self.method = method; self.ratio = ratio
        self.hh,self.recent = budget_at(self.n,ratio)
        assert method=='full' or self.hh+self.recent > 0, 'Upstream zero-slot edge case requires explicit handling'
        self.states = [Layer(kv,score,method,self.hh,self.recent,ratio) for kv,score in zip(source['kv'],source['scores'])]
        self.calls = self.attention_calls = self.steps = 0

    def get_seq_length(self, layer_idx=0): return self.n
    def get_mask_sizes(self, cache_position, layer_idx=0): return self.n+cache_position.shape[0],0
    def get_max_cache_shape(self, layer_idx=0): return -1

    def update(self,k,v,layer_idx,cache_kwargs=None):
        assert layer_idx == self.calls % (self.L*self.R)
        assert k.shape[-2] == v.shape[-2] == 1
        self.calls += 1
        r,l = divmod(layer_idx,self.L)
        return self.states[l].read_update(r,k,v)

    def attention(self,original,module,q,k,v,mask,*args,**kwargs):
        r,l = divmod(self.attention_calls%(self.L*self.R),self.L)
        assert l==module.layer_idx and q.shape[-2]==1
        s = self.states[l]
        if self.method != 'full':
            live = torch.zeros((self.H,self.n+1),device=q.device,dtype=torch.bool)
            live.scatter_(1,s.positions[r],True);live[:,-1]=True
            live = live.repeat_interleave(self.groups,dim=0)
            extra = torch.zeros((1,live.shape[0],1,self.n+1),device=q.device,dtype=q.dtype)
            if mask is not None: extra = extra + mask[:,:,:,:self.n+1]
            mask = extra.masked_fill(~live[None,:,None,:],torch.finfo(q.dtype).min)
        out,w = original(module,q,k,v,mask,*args,**kwargs)
        s.observe(r,w,self.groups);self.attention_calls+=1
        return out,w

    def finish(self):
        assert self.calls==self.attention_calls==(self.steps+1)*self.L*self.R
        for s in self.states:s.finish()
        self.n+=1;self.steps+=1
        self.hh,self.recent = budget_at(self.n,self.ratio)

    def memory(self):
        payload = sum(s.payload.numel()*s.payload.element_size() for s in self.states)
        aux = sum(s.scores.numel()*s.scores.element_size()+s.positions.numel()*s.positions.element_size() for s in self.states)
        slots = self.states[0].positions.shape[-1]
        full = self.L*2*self.R*self.H*self.n*self.states[0].D*self.states[0].payload.element_size()
        assert payload == full*slots//self.n
        expected = self.n if self.method=='full' else sum(budget_at(self.n,self.ratio))
        assert slots == expected, (self.n,slots,expected)
        assert all(s.n==self.n and s.payload.shape[3]==expected for s in self.states)
        return dict(logical_tokens=self.n,prompt_tokens=self.prompt_tokens,slots_per_loop_head=slots,
                    hh_slots=self.hh,recent_slots=self.recent,payload_bytes=payload,auxiliary_bytes=aux,
                    full_payload_bytes=full,payload_ratio=payload/full)

    def decisions(self):
        return torch.stack([s.positions for s in self.states]).cpu().numpy().astype('int32')

    def coverage(self):
        # Any-loop union within each physical layer/KVhead, plus global union.
        live = torch.zeros((self.L,self.R,self.H,self.n),device=self.states[0].positions.device,dtype=torch.bool)
        for l,s in enumerate(self.states):live[l].scatter_(2,s.positions,True)
        union=live.any(dim=1)
        a=torch.stack((union.float().mean(),union[...,:self.prompt_tokens].float().mean(),
                       union.any(dim=0).any(dim=0).float().mean(),live.float().mean())).cpu().tolist()
        return dict(local_any_loop_context=a[0],local_prompt_context=a[1],global_context_union=a[2],per_loop_context=a[3])
