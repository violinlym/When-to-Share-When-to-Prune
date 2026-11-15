import torch
from transformers.cache_utils import DynamicCache,DynamicLayer
from sparse_cache import SparseCache,F
# Compare every routed value to independent uncompressed history; test deletions,
# non-adjacent version aliases, masks and exact KV-payload budget after updates.
torch.set_num_threads(2);torch.manual_seed(123)
for method in ['anchor75','anchor90']:
 truth=[torch.randn(2,4,16,31,128,dtype=torch.bfloat16) for _ in range(24)]
 native=DynamicCache();native.layers=[DynamicLayer() for _ in range(96)];stats={}
 for r in range(4):
  for l in range(24):
   native.update(truth[l][0,r][None],truth[l][1,r][None],r*24+l);a=torch.rand(16,3,31).softmax(-1);stats[r,l]={'peak':a.amax((0,1)),'mass':a.sum(1)*31/3}
 c=SparseCache(native,None,{},stats,{},method)
 for step in range(6):
  for l in range(24):
   for r in range(4):
    h=c.history(l,r)
    for k in range(2):
     for i in range(c.n):
      options=[s for s in range(r,4) if int(c.mask[l][k,i])&(1<<s)]
      expected=truth[l][k,options[0],:,i] if options else torch.zeros_like(h[k,:,i]);assert torch.equal(h[k,:,i],expected)
   # Production attention must not put probability mass on deleted tokens.
   c.calls=l
   def softmax_attention(module,q,k,v,mask,*a,**kw):
    w=(q.float()@k.float().transpose(-1,-2)/128**.5+mask).softmax(-1);return (w@v.float()).transpose(1,2),w
   q=torch.randn(1,16,1,128);kv=torch.randn(1,16,c.n+1,128);out,w=c.attention(softmax_attention,None,q,kv,kv,None)
   dead=(c.mask[l][0]&8)==0;assert (w[...,:c.n][...,dead]==0).all();assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)))
  counts=c.distribution();hist=counts['version_histogram_by_layer_kind']
  for layer in range(24):
   for kind in range(2):
    assert sum(hist[str(i)][layer][kind] for i in range(5))==c.n
    assert sum(i*hist[str(i)][layer][kind] for i in range(5))==counts['versions_by_layer_kind'][layer][kind]
  c.check()
  if step==5:break
  for l in range(24):
   x=torch.randn(2,4,16,1,128,dtype=torch.bfloat16);truth[l]=torch.cat((truth[l],x),3)
   for r in range(4):c.current[r,l]=x[:,r];c.observed[r,l]=torch.rand(16,c.n+1).softmax(-1)
  c.finish()
 print(method,'PASS routes, zero masking, budget, no resurrection, 5 decode updates')
