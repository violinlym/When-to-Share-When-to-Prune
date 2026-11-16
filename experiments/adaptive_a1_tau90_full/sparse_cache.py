"""Route-aware A2 ablation; native values, peak roots, charged pair costs."""
import math
import torch
from base_cache import SparseCache as BaseCache,F,SLOT
from route_gain import PAIRS,allocate
class SparseCache(BaseCache):
 @staticmethod
 def relative(x):
  xf=x.float();norm=xf[:,:3].square().sum((2,4));floor=norm.mean((1,2),keepdim=True)*1e-6+1e-12
  return torch.stack([(xf[:,r]-xf[:,s]).square().sum((1,3))/norm[:,r].clamp_min(floor[:,0]) for r,s in PAIRS],-1).contiguous()
 def planned_metadata(self):return 24*self.n*(4+4+2*6*4+2+2*4)
 def select(self,available):
  device=available.device;roots=torch.zeros((24,self.n),dtype=torch.bool,device=device);recent=max(1,math.ceil(.2*self.n));target=max(recent,math.floor(self.fraction*self.n))
  for l in range(24):
   eligible=(available[l,0]&8)!=0;assert bool((eligible==((available[l,1]&8)!=0)).all())
   roots[l,-recent:]=eligible[-recent:];scores=self.peak[l].masked_fill(~eligible|roots[l],-torch.inf);n=min(target-int(roots[l].sum()),int((eligible&~roots[l]).sum()))
   if n:roots[l,scores.topk(n).indices]=True
  free=(F*self.n//4-int(roots.sum())*2*SLOT)//SLOT
  assert free>=0
  costs=(torch.stack(self.residual)*torch.stack(self.mass)[:,None,:,None]).reshape(-1,6).cpu().tolist()
  masks,audit=allocate(costs,available.flatten().cpu().tolist(),roots[:,None,:].expand(-1,2,-1).flatten().cpu().tolist(),free)
  return torch.tensor(masks,dtype=torch.uint8,device=device).reshape_as(available)
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.audit['policy']='A1 unchanged historical peak; 25% KV-payload budget; A2 greedy actual-route marginal reduction in relative pair residual; A3 unchanged rightward aliases. Not a global optimum or end-to-end error estimate.'
  self.audit['metadata_bytes_per_layer_token']=66
  self.audit['route_encoding']='two int32 token offsets plus existing uint8 masks; logical slot=offset+popcount(mask below requested loop)'

 def make_routes(self):
  # Base class keeps the name routes; tensors now hold only token base offsets.
  return [(self.bits(m).sum(-1).cumsum(-1)-self.bits(m).sum(-1)).to(torch.int32) for m in self.mask]
 def history(self,l,r):
  values=[]
  for k in range(2):
   m=self.mask[l][k];alive=(m&8)!=0;rank=torch.zeros_like(m,dtype=torch.int32)
   for s in range(r):rank+=((m>>s)&1).to(torch.int32)
   address=torch.where(alive,self.routes[l][k]+rank,0).long()
   values.append(torch.where(alive[None,:,None],self.pool[l][k][:,address],0))
  return torch.stack(values)
