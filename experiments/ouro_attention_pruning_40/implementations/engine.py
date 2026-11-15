import contextlib
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer
from cache import H2OCache,PruningCache


class PrefillObserver:
    def __init__(self, cfg):
        self.cfg=cfg;self.calls=0;self.scores={};self.queries={}
    def attention(self,original,module,q,k,v,mask,*args,**kwargs):
        out,w=original(module,q,k,v,mask,*args,**kwargs)
        r,l=divmod(self.calls,self.cfg['layers'])
        assert l==module.layer_idx and q.shape[-2]==k.shape[-2]
        H=self.cfg['kv_heads'];G=self.cfg['query_heads']//H
        a=w.sum(0).sum(1).reshape(H,G,w.shape[-1]).mean(dim=1)
        self.scores[r,l]=a;self.queries[r,l]=q[0,:,-128:].clone();self.calls+=1
        return out,w


class Engine:
    def __init__(self,cfg,model):
        self.cfg=cfg;self.name=model;self.mc=cfg['models'][model];self.calibration=None
        torch.set_num_threads(4);torch.manual_seed(cfg['sample_seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        self.tok=AutoTokenizer.from_pretrained(self.mc['path'],trust_remote_code=True,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(self.mc['path'],trust_remote_code=True,local_files_only=True,
                  torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
        eos=self.model.generation_config.eos_token_id
        self.eos=set(eos if isinstance(eos,list) else [eos if eos is not None else self.tok.eos_token_id])
        c=self.model.config
        assert c.num_hidden_layers==self.mc['layers'] and c.num_key_value_heads==self.mc['kv_heads']
        assert not getattr(c,'use_sliding_window',False)

    @contextlib.contextmanager
    def attach(self,observer):
        module=sys.modules[type(self.model).__module__]
        original=module.eager_attention_forward
        module.eager_attention_forward=lambda *a,**kw:observer.attention(original,*a,**kw)
        try:yield
        finally:module.eager_attention_forward=original

    def native_cache(self):
        cache=DynamicCache();cache.layers=[DynamicLayer() for _ in range(self.mc['layers']*self.mc['loops'])]
        return cache

    def forward(self,ids,cache):
        kw=dict(exit_at_step=3,use_weighted_exit=False) if self.name=='ouro' else {}
        logits=self.model(torch.tensor([ids],device='cuda'),use_cache=True,past_key_values=cache,logits_to_keep=1,**kw).logits
        if isinstance(cache,H2OCache):cache.finish()
        assert torch.isfinite(logits).all(), 'Nonfinite output is technical failure'
        return logits

    @torch.inference_mode()
    def prepare(self,doc):
        ids=self.tok.encode(doc['context'],add_special_tokens=False)
        if self.name=='ouro':assert ids==doc['context_ids']
        torch.cuda.synchronize();started=time.perf_counter()
        native=self.native_cache();obs=PrefillObserver(self.mc)
        with self.attach(obs):logits=self.forward(ids,native)
        L=self.mc['layers'];R=self.mc['loops'];assert obs.calls==L*R
        source=dict(prompt_tokens=len(ids),input_ids=ids,initial_logits=logits,kv=[],scores=[],queries=[])
        for l in range(L):
            source['kv'].append(torch.stack([torch.stack([getattr(native.layers[r*L+l],name)[0] for r in range(R)]) for name in ('keys','values')]))
            source['scores'].append(torch.stack([obs.scores[r,l] for r in range(R)]))
            source['queries'].append(torch.stack([obs.queries[r,l] for r in range(R)]))
        torch.cuda.synchronize();source['prefill_seconds']=time.perf_counter()-started
        return source

    @torch.inference_mode()
    def qualify(self,source):
        native=self.native_cache();L=self.mc['layers'];R=self.mc['loops']
        for l in range(L):
            for r in range(R):native.update(source['kv'][l][0,r][None].clone(),source['kv'][l][1,r][None].clone(),r*L+l)
        adapted=H2OCache(source,self.mc,'full',None);logits=source['initial_logits']
        for _ in range(4):
            token=int(logits[0,-1].argmax());expected=self.forward([token],native)
            with self.attach(adapted):actual=self.forward([token],adapted)
            assert torch.equal(actual,expected),('Native Full logits changed',self.name,(actual-expected).abs().max().item())
            logits=expected
        return dict(status='PASS',model=self.name,native_full_logit_identity_steps=4,prompt_tokens=source['prompt_tokens'])

    @torch.inference_mode()
    def generate(self,doc,source,condition):
        method=condition['method'];ratio=condition['ratio']
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        if condition['algorithm']=='h2o':
            cache=H2OCache(source,self.mc,condition['method'],condition['ratio'])
        else:
            cache=PruningCache(source,self.mc,condition,self.calibration)
        pre=cache.memory();initial=cache.decisions();precover=cache.coverage()
        current=source['initial_logits'];ids=[];reason='length';history=[]
        limit=doc['generation_kwargs']['max_gen_toks'];stops=doc['generation_kwargs']['until']
        assert doc['generation_kwargs']['do_sample'] is False
        with self.attach(cache):
            for i in range(limit):
                token=int(current[0,-1].argmax());ids.append(token)
                if token in self.eos:reason='eos';break
                text=self.tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                if any(s and s in text for s in stops):reason='stop_string';break
                if i+1<limit:
                    current=self.forward([token],cache)
                    mem=cache.memory();history.append([cache.n,mem['payload_bytes'],mem['full_payload_bytes']])
        text=self.tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        ends=[text.find(s) for s in stops if s and s in text]
        if ends:text=text[:min(ends)]
        final=cache.decisions();post=cache.memory();postcover=cache.coverage()
        assert cache.n==source['prompt_tokens']+len(ids)-1
        torch.cuda.synchronize()
        result=dict(generated_ids=ids,generated_text=text,generated_tokens=len(ids),stop_reason=reason,
                    model=self.name,input_ids=source['input_ids'],eos_token_ids=sorted(self.eos),
                    prefill_cache=pre,final_cache=post,prefill_coverage=precover,final_coverage=postcover,
                    byte_history=history,generation_seconds=time.perf_counter()-started,prefill_seconds=source['prefill_seconds'],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    source_staging='Immutable full-prefill tensors reused only for independent condition initialization, never read by decode. Not included in KV payload; peak memory includes them.',
                    cache_policy='physically compact persistent payload; transient dense masked eager reads; no speedup claim')
        return result,initial,final
