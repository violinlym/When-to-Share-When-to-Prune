import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+str(os.getpid()));tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def check():
 c=json.loads((ROOT/'configs/protocol.json').read_text());assert c['status']=='AUTHORIZED'
 for p,h in json.loads((ROOT/'configs/source_lock.json').read_text()).items():assert sha(ROOT/p)==h,p
 for p,h in c['dependencies'].items():assert sha(p)==h,p
 for p,h in c['manifests'].items():assert sha(ROOT/'prompts'/p)==h,p
 all_docs=[json.loads(x) for x in (ROOT/'prompts/questions.jsonl').read_text().splitlines()]
 docs=[]
 for task in ('gsm8k','math500','mbpp_plus'):
  docs.extend([d for d in all_docs if d['task']==task][:64])
 docs=[dict(d,source_manifest_index=d['index'],index=i) for i,d in enumerate(docs)]
 assert len(docs)==192 and c['planned_new_responses']==len(docs)*len(c['new_methods'])
 for i,d in enumerate(docs):
  assert d['index']==i
  for k in ['max_gen_toks','until','do_sample']:assert d['generation_kwargs'][k]==c['generation'][d['task']][k]
 return c,docs,sha(ROOT/'configs/source_lock.json')
