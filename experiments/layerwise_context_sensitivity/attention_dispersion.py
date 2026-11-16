"""Profile full-prefill attention dispersion on the same 48 paired prompts."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from core import select_deleted
from run_profile import DEFAULT_MODEL, HERE, SEED


def metrics(probability):
    p = np.asarray(probability, dtype=np.float64)
    p = p / p.sum()
    positive = p[p > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    n = len(p)
    return entropy / math.log(n), math.exp(entropy) / n, float(np.sort(p)[-math.ceil(n*.25):].sum())


@torch.inference_mode()
def profile_one(model, tokenizer, doc, index, reference):
    ids = doc['context_ids']
    if ids != tokenizer.encode(doc['context'], add_special_tokens=False):
        raise AssertionError('Tokenizer mismatch')
    n = len(ids)
    peak = np.zeros((24,n), dtype=np.float32)
    global_mass = np.zeros((24,n), dtype=np.float64)
    tail_mass = np.zeros((24,n), dtype=np.float64)
    module = sys.modules[type(model).__module__]
    original = module.eager_attention_forward
    calls = [0]

    def observe(attn_module, query, key, value, mask, *args, **kwargs):
        out, weights = original(attn_module, query, key, value, mask, *args, **kwargs)
        loop, physical = divmod(calls[0],24)
        calls[0] += 1
        if loop >= 4 or physical != attn_module.layer_idx:
            raise AssertionError('Ouro loop/layer order mismatch')
        tail = min(32,weights.shape[-2])
        peak[physical] = np.maximum(peak[physical],weights.amax((0,1,2)).float().cpu().numpy())
        global_mass[physical] += (weights.float().sum((0,1,2)) / (weights.shape[1]*weights.shape[2]*4)).cpu().numpy()
        tail_mass[physical] += (weights[:,:,-tail:,:].float().sum((0,1,2)) / (weights.shape[1]*tail*4)).cpu().numpy()
        return out, weights

    cache = DynamicCache()
    cache.layers = [DynamicLayer() for _ in range(96)]
    try:
        module.eager_attention_forward = observe
        model(torch.tensor([ids],device=next(model.parameters()).device),past_key_values=cache,
              use_cache=True,exit_at_step=3,use_weighted_exit=False,logits_to_keep=1)
    finally:
        module.eager_attention_forward = original
    if calls[0] != 96:
        raise AssertionError('Incomplete prefill')

    rows=[]
    for physical in range(24):
        deleted=select_deleted(peak[physical],.25,.20)
        old=reference[(index,physical+1)]
        checksum=hashlib.sha256(deleted.tobytes()).hexdigest()
        if checksum != old['selected_tokens_sha256']:
            raise AssertionError(f'Deletion set differs from original KL run: sample {index}, layer {physical+1}')
        global_removed=float(global_mass[physical,deleted].sum()/global_mass[physical].sum())
        if abs(global_removed-float(old['deleted_attention_mass'])) > 1e-5:
            raise AssertionError('Historical attention mass differs from original KL run')
        ge,gs,gt=metrics(global_mass[physical])
        te,ts,tt=metrics(tail_mass[physical])
        rows.append(dict(sample_index=index,dataset=doc['task'],layer=physical+1,
                         prompt_tokens=n,mean_kl=old['mean_kl'],deleted_tokens=len(deleted),
                         global_normalized_entropy=ge,global_effective_token_fraction=gs,
                         global_top25_mass=gt,global_deleted_mass=global_removed,
                         tail32_normalized_entropy=te,tail32_effective_token_fraction=ts,
                         tail32_top25_mass=tt,
                         tail32_deleted_mass=float(tail_mass[physical,deleted].sum()/tail_mass[physical].sum())))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    ap.add_argument('--worker',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,
          local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    docs=json.loads((HERE/'manifest48.json').read_text())
    with (HERE/'per_sample.csv').open(newline='') as f:
        reference={(int(r['sample_index']),int(r['layer'])):r for r in csv.DictReader(f)}
    out=HERE/'results'/'attention_dispersion'
    out.mkdir(parents=True,exist_ok=True)
    for index,doc in enumerate(docs):
        if index%a.workers!=a.worker:
            continue
        path=out/f'{index:03d}.json'
        if path.exists():
            continue
        rows=profile_one(model,tokenizer,doc,index,reference)
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(rows))
        temp.replace(path)
        print(index,doc['task'],len(rows),flush=True)


if __name__=='__main__':
    main()
