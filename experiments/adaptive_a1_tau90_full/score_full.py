"""Frozen paired accuracy scoring for the 1819 full benchmark prompts."""

import argparse
import csv
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from scorer_v2 import extract_answer, equivalent


HERE=Path(__file__).resolve().parent
FULL_ROOT=Path('/home/manager/A_liyanmin/Project/Loop transformer/independent-prototypes/full_baselines_v2')
SEED=20261119


def gsm_answer(text):
    # Final explicit answer, as prompted by the existing GSM8K few-shot template.
    text=text.split('\nQ:',1)[0]
    matches=list(re.finditer(r'(?:the answer is|####)\s*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?)',text,re.I))
    return matches[-1].group(1).replace(',','') if matches else None


def correct(task,text,gold):
    if task=='gsm8k':
        answer=gsm_answer(text)
        try:return answer, answer is not None and Decimal(answer)==Decimal(str(gold).replace(',',''))
        except InvalidOperation:return answer,False
    answer=extract_answer(text)
    return answer,equivalent(answer,str(gold))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(fixed,adaptive,rng,draws=10000):
    fixed=np.asarray(fixed,dtype=float)
    adaptive=np.asarray(adaptive,dtype=float)
    delta=adaptive-fixed
    draws_indices=rng.integers(0,len(delta),size=(draws,len(delta)))
    distribution=delta[draws_indices].mean(1)
    ci=np.quantile(distribution,[.025,.975])
    improved=int(np.count_nonzero(delta>0))
    worsened=int(np.count_nonzero(delta<0))
    p=float(binomtest(min(improved,worsened),improved+worsened,.5).pvalue) if improved+worsened else 1.0
    return float(delta.mean()),float(ci[0]),float(ci[1]),improved,worsened,p


def load_full_reference(docs):
    source=json.loads((FULL_ROOT/'pilot_manifest.json').read_text())
    source_hash=sha(FULL_ROOT/'pilot_manifest.json')
    lookup={(x['task'],str(x['doc_id'])):(i,x) for i,x in enumerate(source)}
    if len(lookup)!=len(source):raise ValueError('Full reference has duplicate task/doc IDs')
    records={}
    for i,doc in enumerate(docs):
        old_index,old=lookup[(doc['task'],str(doc['doc_id']))]
        for key in ('context','context_ids','generation_kwargs','gold','source_input_sha256'):
            if doc.get(key)!=old.get(key):raise ValueError(f'Full reference prompt mismatch: {i} {key}')
        path=FULL_ROOT/'results'/'native_pilot'/f'{old_index:03d}_full.json'
        raw=json.loads(path.read_text())
        if raw['method']!='full' or raw['task']!=doc['task'] or str(raw['doc_id'])!=str(doc['doc_id']):
            raise ValueError(f'Full reference identity mismatch: {path}')
        if raw['source_hashes']['pilot_manifest.json']!=source_hash:
            path=HERE/'results'/'full_ref'/f'{i:04d}_full.json'
            raw=json.loads(path.read_text())
            if raw['manifest_index']!=i or raw['source_input_sha256']!=doc['source_input_sha256'] or raw['manifest_sha256']!=sha(HERE/'manifest1819.json'):
                raise ValueError(f'Replayed Full KV provenance mismatch: {path}')
        records[i]=(raw,path,old_index)
    return records


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--partial',action='store_true')
    a=ap.parse_args()
    docs=json.loads((HERE/'manifest1819.json').read_text())
    if len(docs)!=1819 or sum(x['task']=='gsm8k' for x in docs)!=1319 or sum(x['task']=='math500' for x in docs)!=500:
        raise ValueError('Full manifest count mismatch')
    if len(set((x['task'],str(x['doc_id'])) for x in docs))!=1819:
        raise ValueError('Duplicate benchmark item')
    manifest_hash=sha(HERE/'manifest1819.json')
    ref=load_full_reference(docs)
    rows=[]
    for i,doc in enumerate(docs):
        for method in ('full','fixed75','adaptive90'):
            if method=='full':raw,path,_=ref[i]
            else:
                path=HERE/'results'/'formal'/f'{i:04d}_{method}.json'
                if not path.exists():
                    if a.partial:continue
                    raise FileNotFoundError(path)
                raw=json.loads(path.read_text())
                if raw['manifest_index']!=i or raw['task']!=doc['task'] or str(raw['doc_id'])!=str(doc['doc_id']):
                    raise ValueError(f'Generated record mismatch: {path}')
                if raw['source_hashes']['manifest1819.json']!=manifest_hash:
                    raise ValueError('Generated manifest hash mismatch')
                for phase in ('initial','final'):
                    if raw[phase]['payload_fraction']>.25+1e-12:
                        raise ValueError(f'KV budget violated: {path}')
                if method=='adaptive90' and min(raw['initial']['historical_attention_mass_coverage_by_layer'])<.9-1e-6:
                    raise ValueError(f'Prefill attention coverage violated: {path}')
            prediction,passed=correct(doc['task'],raw['text'],doc['gold'])
            rows.append(dict(task=doc['task'],index=i,doc_id=doc['doc_id'],method=method,
                             correct=int(passed),prediction=prediction or '',gold=doc['gold'],
                             generated_tokens=len(raw['generated_ids']),stop_reason=raw['stop_reason'],
                             initial_payload_fraction=1.0 if method=='full' else raw['initial']['payload_fraction'],
                             final_payload_fraction=1.0 if method=='full' else raw['final']['payload_fraction'],
                             initial_root_slots='' if method=='full' else raw['initial']['root_slots'],
                             initial_a2_slots='' if method=='full' else raw['initial']['extra_kv_version_slots'],
                             final_root_slots='' if method=='full' else raw['final']['root_slots'],
                             final_a2_slots='' if method=='full' else raw['final']['extra_kv_version_slots'],
                             source_path=str(path),source_sha256=sha(path)))
    for task,outname in (('gsm8k','gsm8k_full_results.csv'),('math500','math500_full_results.csv')):
        subset=[x for x in rows if x['task']==task]
        with (HERE/outname).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(subset)
    rng=np.random.default_rng(SEED)
    summaries=[]
    paired=[]
    for task,count in (('gsm8k',1319),('math500',500)):
        table={method:{x['index']:x for x in rows if x['task']==task and x['method']==method}
               for method in ('full','fixed75','adaptive90')}
        for method in table:
            if not a.partial and len(table[method])!=count:raise ValueError(f'Incomplete {task} {method}')
        common=sorted(set(table['fixed75'])&set(table['adaptive90']))
        if common:
            delta,low,high,improved,worsened,p=paired_bootstrap(
                [table['fixed75'][i]['correct'] for i in common],
                [table['adaptive90'][i]['correct'] for i in common],rng)
            paired.append(dict(task=task,paired_items=len(common),delta_accuracy=delta,
                               ci95_low=low,ci95_high=high,adaptive_only_correct=improved,
                               fixed_only_correct=worsened,mcnemar_exact_p=p))
        for method,group in table.items():
            xs=list(group.values())
            summaries.append(dict(task=task,method=method,items=len(xs),correct=sum(x['correct'] for x in xs),
                                  accuracy=sum(x['correct'] for x in xs)/len(xs) if xs else '',
                                  mean_initial_payload_fraction=np.mean([x['initial_payload_fraction'] for x in xs]) if xs else '',
                                  mean_final_payload_fraction=np.mean([x['final_payload_fraction'] for x in xs]) if xs else ''))
    for name,data in (('accuracy_summary.csv',summaries),('paired_stats.csv',paired)):
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0].keys(),lineterminator='\n');w.writeheader();w.writerows(data)
    print(json.dumps({'summary':summaries,'paired':paired},indent=2))


if __name__=='__main__':main()
