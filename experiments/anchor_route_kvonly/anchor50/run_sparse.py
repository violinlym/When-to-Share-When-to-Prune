import argparse,json,time,sys,hashlib,os,fcntl,random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache,DynamicLayer
from sparse_cache import SparseCache
P=Path(__file__).resolve().parent
def load_codec(*args):return {}
def SlotCache(native,model,raw,stats,codec,method):return SparseCache(native,model,raw,stats,codec,method)
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',default='anchor50');ap.add_argument('--smoke',action='store_true');ap.add_argument('--trained',action='store_true');ap.add_argument('--axis',action='store_true');ap.add_argument('--queue',action='store_true');ap.add_argument('--screen',action='store_true');ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(20261119);torch.backends.cuda.matmul.allow_tf32=False
 model=AutoModelForCausalLM.from_pretrained(P/'model',trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval();tok=AutoTokenizer.from_pretrained(P/'model',trust_remote_code=True,local_files_only=True);codec=load_codec(a.trained,a.axis);label=a.method
 checkpoint_hash='none_native_bf16';raw={};handles=[]
 for l in range(24):
  for kind,name in enumerate(['k_proj','v_proj']):
   def capture(mod,args,out,l=l,kind=kind):raw.setdefault((l,kind),[]).append(out.detach())
   handles.append(getattr(model.model.layers[l].self_attn,name).register_forward_hook(capture))
 def forward(ids,c):return model(torch.tensor([ids],device='cuda'),past_key_values=c,use_cache=True,exit_at_step=3,use_weighted_exit=False,logits_to_keep=1).logits
 docs=json.loads((P/'manifest64.json').read_text());indices=[0] if a.smoke else list(range(4)) if a.screen else list(range(len(docs)));random.Random(20261119).shuffle(indices)
 module=sys.modules[type(model).__module__];original=module.eager_attention_forward
 for unit,i in enumerate(indices):
  if not a.queue and unit%a.workers!=a.worker:continue
  dest=P/'results'/('smoke' if a.smoke else 'slots_pilot')/f'{i:03d}_{label}.json';dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():continue
  lease=dest.with_suffix('.lock').open('a')
  try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:continue
  if dest.exists():continue
  d=docs[i];ids=d['context_ids'];assert ids==tok.encode(d['context'],add_special_tokens=False)
  stats={};calls=[0];start=time.time();torch.cuda.reset_peak_memory_stats()
  def observe(m,q,k,v,mask,*aa,**kw):
   out,w=original(m,q,k,v,mask,*aa,**kw);r,l=divmod(calls[0],24);calls[0]+=1
   idx=torch.linspace(max(0,q.shape[-2]-128),q.shape[-2]-1,min(8,q.shape[-2]),device='cuda').long()
   stats[r,l]={'mass':w.sum(0).sum(1).detach().clone(),'peak':w.amax((0,1,2)).detach().clone(),'q':q[0,:,idx].detach().clone(),'a':w[0,:,idx].detach().clone(),'o':out[0,idx].transpose(0,1).detach().clone()}
   return out,w
  module.eager_attention_forward=observe
  native=DynamicCache();native.layers=[DynamicLayer() for _ in range(96)];logits=forward(ids,native)
  module.eager_attention_forward=original
  c=SlotCache(native,model,raw,stats,codec,a.method);del native
  module.eager_attention_forward=lambda *aa,**kw:c.attention(original,*aa,**kw)
  generated=[];limit=6 if a.smoke else d['generation_kwargs']['max_gen_toks'];reason='length';eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
  for step in range(limit):
   t=int(logits[0,-1].argmax());generated.append(t)
   if t in eos:reason='eos';break
   text=tok.decode(generated,skip_special_tokens=False)
   if any(s in text for s in d['generation_kwargs']['until']):reason='stop';break
   if step+1<limit:logits=forward([t],c);c.finish()
  fullbytes=c.full_initial*c.get_seq_length()//len(ids);assert c.nbytes()<=fullbytes//2
  result=dict(method=label,task=d['task'],doc_id=d['doc_id'],manifest_index=i,gold=d.get('gold'),generated_ids=generated,text=tok.decode(generated,skip_special_tokens=True),stop_reason=reason,seconds=time.time()-start,prompt_tokens=len(ids),initial_cache_bytes=c.initial_bytes,initial_full_bytes=c.full_initial,final_cache_bytes=c.nbytes(),final_full_bytes=fullbytes,initial_metadata=c.initial_metadata,final_metadata=c.metadata_bytes(),budget_policy='KV payload only; metadata measured separately',context_coverage=c.coverage(),allocation_audit=c.audit,peak_allocated_bytes=torch.cuda.max_memory_allocated(),shared_codec_bytes=0,source_hashes={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['sparse_cache.py','base_cache.py','route_gain.py','run_sparse.py','manifest64.json']},codec_checkpoint_sha256=checkpoint_hash)
  tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False));tmp.replace(dest);print(i,label,len(generated),result['seconds'],flush=True)
  module.eager_attention_forward=original;raw.clear();stats.clear();del c;torch.cuda.empty_cache()
if __name__=='__main__':main()
