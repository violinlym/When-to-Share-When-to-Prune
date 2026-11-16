"""Paired full GSM8K/MATH500 generation with only A1 changed."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from adaptive_cache import AdaptiveCache
from sparse_cache import SparseCache as FixedCache, F, SLOT


HERE = Path(__file__).resolve().parent
MODEL = Path('/home/manager/A_liyanmin/Project/Loop transformer/independent-prototypes/model')
SEED = 20261119
METHODS = {'fixed75': FixedCache, 'adaptive90': AdaptiveCache}


def root_counts(cache):
    return [int(((m[0] & 8) != 0).sum().item()) for m in cache.mask]


def attention_coverage(cache, mass):
    return [float((mass[l][(cache.mask[l][0] & 8) != 0].sum() / mass[l].sum()).item())
            for l in range(24)]


def allocation(cache):
    roots = root_counts(cache)
    payload = cache.nbytes()
    total_slots = payload // SLOT
    extra = total_slots - 2 * sum(roots)
    if extra < 0 or total_slots * SLOT != payload:
        raise AssertionError('Invalid K/V slot accounting')
    by_kind = cache.distribution()['versions_by_layer_kind']
    extra_by_layer_kind = [[int(by_kind[l][k]-roots[l]) for k in range(2)] for l in range(24)]
    if sum(sum(x) for x in extra_by_layer_kind) != extra:
        raise AssertionError('A2 slot accounting mismatch')
    return dict(roots_by_layer=roots, root_slots=sum(roots), extra_kv_version_slots=extra,
                extra_by_layer_kind=extra_by_layer_kind, payload_bytes=payload,
                payload_fraction=payload/(F*cache.n), metadata_bytes=cache.metadata_bytes())


@torch.inference_mode()
def run_method(model, tokenizer, module, original, native, prefill_logits,
               stats, doc, index, method, smoke=False):
    klass = METHODS[method]
    cache = klass(native, model, {}, dict(stats), {}, 'anchor75' if method=='fixed75' else 'adaptive90')
    n = len(doc['context_ids'])
    mass = [sum(stats[r,l]['coverage_mass'].double() for r in range(4)) for l in range(24)]
    before = allocation(cache)
    before['historical_attention_mass_coverage_by_layer'] = attention_coverage(cache,mass)
    if method=='adaptive90' and min(before['historical_attention_mass_coverage_by_layer']) < .9-1e-6:
        raise AssertionError('Prefill 90% coverage violated')
    generated = []
    logits = prefill_logits
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos])
    limit = 6 if smoke else doc['generation_kwargs']['max_gen_toks']
    reason = 'length'
    start = time.monotonic()
    try:
        module.eager_attention_forward = lambda *args,**kw: cache.attention(original,*args,**kw)
        for step in range(limit):
            token = int(logits[0,-1].argmax())
            generated.append(token)
            if token in eos:
                reason = 'eos'
                break
            text = tokenizer.decode(generated,skip_special_tokens=False)
            if any(stop in text for stop in doc['generation_kwargs']['until']):
                reason = 'stop'
                break
            if step+1 < limit:
                logits = model(torch.tensor([[token]],device='cuda'),past_key_values=cache,
                               use_cache=True,exit_at_step=3,use_weighted_exit=False,
                               logits_to_keep=1).logits
                if smoke:
                    for layer in range(24):
                        dead = (cache.mask[layer][0]&8)==0
                        for loop in range(4):
                            if bool((cache.observed[loop,layer][:,:cache.n][:,dead]!=0).any()):
                                raise AssertionError('Deleted token received attention')
                cache.finish()
                if smoke:
                    recent = __import__('math').ceil(.2*cache.n)
                    for layer in range(24):
                        if not bool(((cache.mask[layer][0,-recent:]&8)!=0).all()):
                            raise AssertionError('Recent token not protected')
    finally:
        module.eager_attention_forward = original
    after = allocation(cache)
    target = F*cache.n//4
    if after['payload_bytes'] > target:
        raise AssertionError('25% Full KV payload cap violated')
    record = dict(method=method,task=doc['task'],doc_id=doc['doc_id'],manifest_index=index,
                  generated_ids=generated,text=tokenizer.decode(generated,skip_special_tokens=True),
                  stop_reason=reason,generated_tokens=len(generated),decode_seconds=time.monotonic()-start,
                  prompt_tokens=n,initial=before,final=after,
                  initial_full_bytes=F*n,final_full_bytes=F*cache.n,
                  budget_violation_count=0,root_resurrection_count=getattr(cache,'root_resurrections',0),
                  coverage_violation_layer_steps=getattr(cache,'coverage_violations',None),
                  minimum_observed_coverage=getattr(cache,'coverage_min',None),
                  coverage_selection_steps=getattr(cache,'coverage_steps',None))
    del cache
    return record


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--method',choices=('both','fixed75','adaptive90'),default='both')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--task',choices=('both','gsm8k','math500'),default='both')
    args=parser.parse_args()
    if not (0<=args.worker<args.workers):parser.error('Invalid worker partition')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,local_files_only=True,
                                               torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    docs=json.loads((HERE/'manifest1819.json').read_text())
    if len(docs)!=1819:
        raise AssertionError('Full benchmark manifest must contain 1819 prompts')
    module=sys.modules[type(model).__module__]
    original=module.eager_attention_forward
    methods=list(METHODS) if args.method=='both' else [args.method]
    out=HERE/'results'/('smoke' if args.smoke else 'formal')
    out.mkdir(parents=True,exist_ok=True)
    sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
             for name in ('base_cache.py','sparse_cache.py','route_gain.py','adaptive_cache.py','run_full.py','manifest1819.json')}
    indices=[0] if args.smoke else list(range(len(docs)))
    for index in indices:
        doc=docs[index]
        if index%args.workers!=args.worker or (args.task!='both' and doc['task']!=args.task):continue
        needed=[method for method in methods if not (out/f'{index:04d}_{method}.json').exists()]
        if not needed:continue
        ids=doc['context_ids']
        if ids!=tokenizer.encode(doc['context'],add_special_tokens=False):
            raise AssertionError(f'Tokenizer mismatch at {index}')
        stats={}
        calls=[0]
        def observe(attn_module,query,key,value,mask,*aa,**kw):
            result,weights=original(attn_module,query,key,value,mask,*aa,**kw)
            loop,layer=divmod(calls[0],24)
            calls[0]+=1
            if loop>=4 or layer!=attn_module.layer_idx:raise AssertionError('Prefill call order')
            stats[loop,layer]={'peak':weights.amax((0,1,2)).detach().clone(),
                               'mass':weights.sum(0).sum(1).detach().clone(),
                               'coverage_mass':(weights.float().sum((0,1,2)) /
                                                (weights.shape[1]*weights.shape[2]*4)).detach().clone()}
            return result,weights
        native=DynamicCache()
        native.layers=[DynamicLayer() for _ in range(96)]
        try:
            module.eager_attention_forward=observe
            prefill_logits=model(torch.tensor([ids],device='cuda'),past_key_values=native,
                                 use_cache=True,exit_at_step=3,use_weighted_exit=False,
                                 logits_to_keep=1).logits
        finally:
            module.eager_attention_forward=original
        if calls[0]!=96 or any(layer.get_seq_length()!=len(ids) for layer in native.layers):
            raise AssertionError('Incomplete Full KV prefill')
        for method in needed:
            torch.cuda.reset_peak_memory_stats()
            record=run_method(model,tokenizer,module,original,native,prefill_logits,stats,doc,index,method,args.smoke)
            record['source_hashes']=sources
            record['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            path=out/f'{index:04d}_{method}.json'
            temporary=path.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(record,ensure_ascii=False))
            temporary.replace(path)
            print(index,method,record['generated_tokens'],round(record['decode_seconds'],1),flush=True)
        del native,stats,prefill_logits
        torch.cuda.empty_cache()


if __name__=='__main__':main()
