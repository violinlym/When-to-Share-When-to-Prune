"""Condition claims; release native staging before each decode."""
import argparse,fcntl,json,os,time,traceback
from pathlib import Path
rank=int(os.environ.get('SLURM_PROCID','0'))
if 'SLURM_JOB_ID' in os.environ:
 os.environ['ROCR_VISIBLE_DEVICES']=os.environ.get('SLURM_LOCALID','0')
 for key in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):os.environ.pop(key,None)
from common import ROOT,check,sha,write

def save_result(path,result,initial,final):
 import numpy as np
 path.parent.mkdir(parents=True,exist_ok=True);state=path.with_suffix('.decisions.npz');tmp=state.with_name(state.name+'.tmp-'+str(os.getpid()))
 with tmp.open('wb') as stream:np.savez_compressed(stream,initial=initial,final=final)
 tmp.replace(state);result.update(decisions_path=str(state),decisions_sha256=sha(state));write(path,result)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
 cfg,docs,identity=check()
 from engine import Engine
 if args.check:print(json.dumps(dict(status='PASS',identity=identity,new_responses=cfg['planned_new_responses'])));return
 import numpy as np
 import torch
 assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
 store=Path(cfg['store'])
 for folder in ('raw','claims','failures','qualification'):(store/folder).mkdir(parents=True,exist_ok=True)
 ordered=[docs[i+64*domain] for i in range(64) for domain in range(3)]
 units=[(doc,method) for doc in ordered for method in cfg['new_methods']]
 engine=None;qualified=False
 for offset in range(len(units)):
  doc,method=units[(rank+offset)%len(units)];stem='%04d-%s'%(doc['index'],method);path=store/'raw'/method/('%04d.json'%doc['index'])
  with (store/'claims'/(stem+'.lock')).open('a') as claim:
   try:fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if path.exists():
    old=json.loads(path.read_text());assert old['identity']==identity and old['source_input_sha256']==doc['source_input_sha256'];assert sha(old['decisions_path'])==old['decisions_sha256'];continue
   try:
    if engine is None:
     start=time.perf_counter();engine=Engine(cfg,'ouro');print(json.dumps(dict(event='loaded',rank=rank,seconds=time.perf_counter()-start)),flush=True)
    print(json.dumps(dict(event='start',index=doc['index'],method=method,domain=doc['domain'],rank=rank)),flush=True)
    source=engine.prepare(doc)
    if not qualified:
     record=engine.qualify(source);record.update(identity=identity,job=os.environ['SLURM_JOB_ID'],rank=rank,index=doc['index']);write(store/'qualification'/('%s-%d.json'%(os.environ['SLURM_JOB_ID'],rank)),record);qualified=True
    assert source['input_ids']==doc['context_ids']
    ratio=.4 if method in ('hierarchy_price_exchange', 'hierarchy_version') else .25
    result,initial,final=engine.generate(doc,source,dict(method=method,ratio=ratio))
    result.update(identity=identity,method=method,task=doc['task'],domain=doc['domain'],index=doc['index'],doc_id=doc['doc_id'],source_input_sha256=doc['source_input_sha256'],manifest_sha256=cfg['manifests']['questions.jsonl'],job=os.environ['SLURM_JOB_ID'],rank=rank)
    save_result(path,result,initial,final);print(json.dumps(dict(event='done',method=method,index=doc['index'],rank=rank,tokens=result['generated_tokens'],seconds=result['generation_seconds'])),flush=True);del source
   except Exception:
    write(store/'failures'/(stem+'-'+os.environ['SLURM_JOB_ID']+'.json'),dict(identity=identity,error=traceback.format_exc()));raise
if __name__=='__main__':main()
