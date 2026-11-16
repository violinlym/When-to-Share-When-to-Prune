"""Native BF16 roots, independent residual-weighted versions and explicit aliases.
Deleted tokens are masked out, never advertised as retained context.
Allocation is greedy, irreversible and not globally optimal.
"""
import math
import torch
from transformers.cache_utils import DynamicCache,DynamicLayer
F=24*4*2*16*128*2
SLOT=16*128*2
class SparseCache(DynamicCache):
 def __init__(self,native,model,raw,stats,codec,method):
  super().__init__();self.layers=[DynamicLayer() for _ in range(96)];self.n=native.get_seq_length();self.raw,self.stats=raw,stats;self.calls=0;self.current={};self.observed={};self.fraction=.75 if method=='anchor75' else .90
  self.peak=[];self.mass=[];self.residual=[];source=[]
  for l in range(24):
   x=torch.stack([torch.stack([getattr(native.layers[r*24+l],name)[0] for r in range(4)]) for name in ['keys','values']]);source.append(x)
   self.peak.append(torch.stack([stats[r,l]['peak'] for r in range(4)]).amax(0).float())
   self.mass.append(torch.stack([stats[r,l]['mass'] for r in range(4)]).float().mean((0,1))/self.n)
   self.residual.append(self.relative(x))
  available=torch.full((24,2,self.n),15,dtype=torch.uint8,device=source[0].device);chosen=self.select(available)
  self.mask=[m.clone() for m in chosen];self.pool=[]
  for l,x in enumerate(source):self.pool.append([x[k].permute(1,2,0,3)[:,self.bits(chosen[l,k])].contiguous() for k in range(2)])
  self.routes=self.make_routes();self.audit={'policy':'A1 historical peak root selection; A2 accumulated mean attention times relative cross-loop residual; A3 nearest right same-token alias; deleted keys masked','root_fraction':self.fraction,'recent_root_fraction':.2,'prefill':self.distribution()}
  self.full_initial=F*self.n;self.initial_bytes=self.nbytes();self.initial_metadata=self.metadata_bytes();raw.clear();stats.clear();self.check()
 @staticmethod
 def bits(m):return (m[...,None] & (1<<torch.arange(4,device=m.device)))!=0
 @staticmethod
 def relative(x):
  xf=x.float();diff=(xf[:,:3]-xf[:,3:4]).square().sum((2,4));norm=xf[:,:3].square().sum((2,4));floor=norm.mean((1,2),keepdim=True)*1e-6+1e-12
  return (diff/norm.clamp_min(floor)).permute(0,2,1).contiguous()
 def planned_metadata(self):return 24*self.n*(4+4+2*3*4+2+2*4*4) # peaks, mass, residuals, masks, routes
 def select(self,available):
  device=available.device;roots=torch.zeros((24,self.n),dtype=torch.bool,device=device);recent=max(1,math.ceil(.2*self.n));target=max(recent,math.floor(self.fraction*self.n))
  for l in range(24):
   eligible=(available[l,0]&8)!=0;assert bool((eligible==((available[l,1]&8)!=0)).all())
   roots[l,-recent:]=eligible[-recent:];scores=self.peak[l].masked_fill(~eligible|roots[l],-torch.inf);n=min(target-int(roots[l].sum()),int((eligible&~roots[l]).sum()))
   if n:roots[l,scores.topk(n).indices]=True
  out=torch.zeros_like(available);out|=roots[:,None,:].to(torch.uint8)*8
  free=(F*self.n//4-int(roots.sum())*2*SLOT)//SLOT
  assert free>=0,(self.n,free)
  gain=torch.stack(self.residual)*torch.stack(self.mass)[:,None,:,None]
  elig=self.bits(available)[...,:3]&roots[:,None,:,None]
  count=min(free,int(elig.sum()))
  if count:
   chosen=gain.masked_fill(~elig,-torch.inf).flatten().topk(count).indices
   flags=torch.zeros_like(elig);flags.flatten()[chosen]=True
   out|=(flags.to(torch.uint8)*(1<<torch.arange(3,device=device))).sum(-1).to(torch.uint8)
  assert bool(((out&available)==out).all());return out
 def make_routes(self):
  result=[]
  for l in range(24):
   rr=[]
   for k in range(2):
    bits=self.bits(self.mask[l][k]);cnt=bits.sum(-1);offset=cnt.cumsum(0)-cnt;arr=[]
    for r in range(4):
     alive=bits[:,r:].any(-1);next_r=bits[:,r:].long().argmax(-1)+r
     before=(bits&(torch.arange(4,device=bits.device)[None,:]<next_r[:,None])).sum(-1)
     arr.append(torch.where(alive,offset+before,-1).to(torch.int32))
    rr.append(torch.stack(arr))
   result.append(torch.stack(rr))
  return result
 def history(self,l,r):
  return torch.stack([torch.where((self.routes[l][k,r]>=0)[None,:,None],self.pool[l][k][:,self.routes[l][k,r].clamp_min(0).long()],0) for k in range(2)])
 def update(self,k,v,layer_idx,cache_kwargs=None):
  r,l=divmod(layer_idx,24);self.current[r,l]=torch.stack((k[0],v[0]));x=self.history(l,r);return torch.cat((x[0][None],k),-2),torch.cat((x[1][None],v),-2)
 def attention(self,original,module,q,k,v,mask,*args,**kwargs):
  r,l=divmod(self.calls%96,24);self.calls+=1;alive=torch.cat(((self.mask[l][0]&8)!=0,torch.ones(1,device=q.device,dtype=torch.bool)))
  penalty=torch.zeros((1,1,1,len(alive)),device=q.device,dtype=q.dtype).masked_fill(~alive[None,None,None,:],-torch.inf)
  mask=penalty if mask is None else mask+penalty
  out,w=original(module,q,k,v,mask,*args,**kwargs);self.observed[r,l]=w[0,:,0].float().detach();return out,w
 def finish(self):
  new=[]
  for l in range(24):
   x=torch.stack([self.current[r,l] for r in range(4)],1);new.append(x);a=torch.stack([self.observed[r,l] for r in range(4)])
   self.peak[l]=torch.maximum(torch.cat((self.peak[l],self.peak[l].new_zeros(1))),a.amax((0,1)))
   self.mass[l]=torch.cat((self.mass[l],self.mass[l].new_zeros(1)))+a.mean((0,1))
   self.residual[l]=torch.cat((self.residual[l],self.relative(x)),1)
  self.n+=1;available=torch.stack([torch.cat((m,torch.full((2,1),15,dtype=torch.uint8,device=m.device)),1) for m in self.mask]);chosen=self.select(available)
  for l,x in enumerate(new):
   for k in range(2):
    values=torch.cat((self.pool[l][k],x[k,:, :,0].permute(1,0,2)),1);self.pool[l][k]=values[:,self.bits(chosen[l,k])[self.bits(available[l,k])]].contiguous()
   self.mask[l]=chosen[l].clone()
  self.routes=self.make_routes();self.audit['final']=self.distribution();self.current.clear();self.observed.clear();self.raw.clear();self.stats.clear();self.check()
 def distribution(self):
  m=torch.stack(self.mask);counts=self.bits(m).sum(-1);return {'coverage':self.coverage(),'roots_by_layer':((m[:,0]&8)!=0).sum(-1).tolist(),'versions_by_layer_kind':counts.sum(2).tolist(),'version_histogram_by_layer_kind':{str(i):(counts==i).sum(2).tolist() for i in range(5)},'kv_different_masks':int((m[:,0]!=m[:,1]).sum())}
 def coverage(self):return float(torch.stack([(m[0]&8)!=0 for m in self.mask]).float().mean())
 def nbytes(self):return sum(t.numel()*t.element_size() for pair in self.pool for t in pair)
 def metadata_bytes(self):return sum(t.numel()*t.element_size() for t in self.peak+self.mass+self.residual+self.mask+self.routes)
 def check(self):
  assert self.metadata_bytes()==self.planned_metadata()
  assert self.nbytes()<=F*self.n//4
 def get_seq_length(self,layer_idx=0):return self.n
 def get_mask_sizes(self,cache_position,layer_idx=0):return self.n+cache_position.shape[0],0
 def get_max_cache_shape(self,layer_idx=0):return -1
