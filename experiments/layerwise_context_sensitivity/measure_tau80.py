"""Measure fixed 80% attention-mass anchor occupancy, before any A2 allocation."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from run_profile import DEFAULT_MODEL, HERE, SEED


TAU = .8
RECENT = .2
FIXED = .75
LAYERS = 24
LOOPS = 4
KV_ELEMENTS = 2 * 16 * 128  # one K and one V root for one physical layer/token
BYTES_PER_ROOT = KV_ELEMENTS * 2  # native BF16


def root_masks(peak, mass, tau=TAU):
    """Same peak order for both policies; only the stopping rule differs."""
    n = len(peak)
    recent = math.ceil(RECENT*n)
    eligible = n-recent
    order = np.lexsort((np.arange(eligible), -np.asarray(peak[:eligible],dtype=np.float64)))
    fixed = np.zeros(n,dtype=bool)
    adaptive = np.zeros(n,dtype=bool)
    fixed[eligible:] = True
    adaptive[eligible:] = True
    fixed_target=max(recent, math.floor(FIXED*n))  # matches anchor75 A1
    fixed[order[:fixed_target-recent]] = True
    normalized=np.asarray(mass,dtype=np.float64)
    normalized=normalized/normalized.sum()
    running=float(normalized[adaptive].sum())
    for token in order:
        if running >= tau-1e-12:
            break
        adaptive[token]=True
        running += float(normalized[token])
    assert running >= tau-1e-9
    assert fixed[eligible:].all() and adaptive[eligible:].all()
    return fixed,adaptive,order


@torch.inference_mode()
def measure_one(model,tokenizer,doc,index):
    ids=doc['context_ids']
    if ids!=tokenizer.encode(doc['context'],add_special_tokens=False):
        raise AssertionError('Tokenizer mismatch')
    n=len(ids)
    peak=np.zeros((LAYERS,n),dtype=np.float32)
    mass=np.zeros((LAYERS,n),dtype=np.float64)
    tail=np.zeros((LAYERS,n),dtype=np.float64)
    module=sys.modules[type(model).__module__]
    original=module.eager_attention_forward
    calls=[0]
    def observe(attn_module,query,key,value,mask,*args,**kwargs):
        output,weights=original(attn_module,query,key,value,mask,*args,**kwargs)
        loop,layer=divmod(calls[0],LAYERS)
        calls[0]+=1
        if loop>=LOOPS or layer!=attn_module.layer_idx:
            raise AssertionError('Loop/layer call order mismatch')
        tail_q=min(32,weights.shape[-2])
        peak[layer]=np.maximum(peak[layer],weights.amax((0,1,2)).float().cpu().numpy())
        mass[layer]+=(weights.float().sum((0,1,2))/(weights.shape[1]*weights.shape[2]*LOOPS)).cpu().numpy()
        tail[layer]+=(weights[:,:,-tail_q:,:].float().sum((0,1,2))/(weights.shape[1]*tail_q*LOOPS)).cpu().numpy()
        return output,weights
    cache=DynamicCache()
    cache.layers=[DynamicLayer() for _ in range(LAYERS*LOOPS)]
    try:
        module.eager_attention_forward=observe
        model(torch.tensor([ids],device=next(model.parameters()).device),past_key_values=cache,
              use_cache=True,exit_at_step=3,use_weighted_exit=False,logits_to_keep=1)
    finally:
        module.eager_attention_forward=original
    if calls[0]!=LAYERS*LOOPS or any(x.get_seq_length()!=n for x in cache.layers):
        raise AssertionError('Full KV prefill failed')
    rows=[]
    for layer in range(LAYERS):
        fixed,adaptive,_=root_masks(peak[layer],mass[layer])
        pmass=mass[layer]/mass[layer].sum()
        ptail=tail[layer]/tail[layer].sum()
        rows.append(dict(sample_index=index,dataset=doc['task'],doc_id=doc['doc_id'],
                         layer=layer+1,prompt_tokens=n,recent_tokens=math.ceil(RECENT*n),
                         fixed_roots=int(fixed.sum()),adaptive_roots=int(adaptive.sum()),
                         fixed_ratio=float(fixed.mean()),adaptive_ratio=float(adaptive.mean()),
                         fixed_mass_coverage=float(pmass[fixed].sum()),
                         adaptive_mass_coverage=float(pmass[adaptive].sum()),
                         fixed_tail32_coverage=float(ptail[fixed].sum()),
                         adaptive_tail32_coverage=float(ptail[adaptive].sum()),
                         fixed_root_kv_bytes=int(fixed.sum()*BYTES_PER_ROOT),
                         adaptive_root_kv_bytes=int(adaptive.sum()*BYTES_PER_ROOT),
                         full_kv_bytes=int(LOOPS*n*BYTES_PER_ROOT)))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    ap.add_argument('--worker',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,local_files_only=True,
           torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    docs=json.loads((HERE/'manifest48.json').read_text())
    output=HERE/'results'/'tau80_anchors'
    output.mkdir(parents=True,exist_ok=True)
    for index,doc in enumerate(docs):
        if (a.smoke and index!=0) or index%a.workers!=a.worker:
            continue
        path=output/f'{index:03d}.json'
        if path.exists():
            continue
        rows=measure_one(model,tokenizer,doc,index)
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(rows))
        temp.replace(path)
        print(index,doc['task'],round(np.mean([x['adaptive_ratio'] for x in rows]),3),flush=True)


if __name__=='__main__':
    main()
