"""Regenerate only Full KV items whose historical manifest hash differs."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from run_full import HERE, MODEL, SEED


SOURCE=Path('/home/manager/A_liyanmin/Project/Loop transformer/independent-prototypes/full_baselines_v2')


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args()
    docs=json.loads((HERE/'manifest1819.json').read_text())
    source=json.loads((SOURCE/'pilot_manifest.json').read_text())
    current_hash=hashlib.sha256((SOURCE/'pilot_manifest.json').read_bytes()).hexdigest()
    lookup={(x['task'],str(x['doc_id'])):(i,x) for i,x in enumerate(source)}
    required=[]
    for index,doc in enumerate(docs):
        old_index,old=lookup[(doc['task'],str(doc['doc_id']))]
        for field in ('context','context_ids','generation_kwargs','gold','source_input_sha256'):
            if doc.get(field)!=old.get(field):raise ValueError(f'Manifest mismatch {index} {field}')
        old_result=json.loads((SOURCE/'results'/'native_pilot'/f'{old_index:03d}_full.json').read_text())
        if old_result['source_hashes']['pilot_manifest.json']!=current_hash:
            required.append(index)
    print('full references requiring exact replay',len(required),flush=True)
    if not required:return
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,
                 torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True)
    out=HERE/'results'/'full_ref'
    out.mkdir(parents=True,exist_ok=True)
    for index in required:
        if index%a.workers!=a.worker:continue
        path=out/f'{index:04d}_full.json'
        if path.exists():continue
        doc=docs[index]
        ids=doc['context_ids']
        if ids!=tok.encode(doc['context'],add_special_tokens=False):raise ValueError('Tokenizer mismatch')
        cache=DynamicCache()
        cache.layers=[DynamicLayer() for _ in range(96)]
        logits=model(torch.tensor([ids],device='cuda'),past_key_values=cache,use_cache=True,
                     exit_at_step=3,use_weighted_exit=False,logits_to_keep=1).logits
        eos=model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        generated=[]
        reason='length'
        for step in range(doc['generation_kwargs']['max_gen_toks']):
            token=int(logits[0,-1].argmax())
            generated.append(token)
            if token in eos:reason='eos';break
            text=tok.decode(generated,skip_special_tokens=False)
            if any(stop in text for stop in doc['generation_kwargs']['until']):reason='stop';break
            if step+1<doc['generation_kwargs']['max_gen_toks']:
                logits=model(torch.tensor([[token]],device='cuda'),past_key_values=cache,
                             use_cache=True,exit_at_step=3,use_weighted_exit=False,
                             logits_to_keep=1).logits
        record=dict(method='full',task=doc['task'],doc_id=doc['doc_id'],manifest_index=index,
                    generated_ids=generated,text=tok.decode(generated,skip_special_tokens=True),
                    stop_reason=reason,prompt_tokens=len(ids),source_input_sha256=doc['source_input_sha256'],
                    manifest_sha256=hashlib.sha256((HERE/'manifest1819.json').read_bytes()).hexdigest())
        temp=path.with_suffix(f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(record,ensure_ascii=False))
        temp.replace(path)
        print(index,doc['task'],len(generated),flush=True)


if __name__=='__main__':main()
