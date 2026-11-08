"""Promote valid, already-authorized work without selecting on development accuracy."""
import json
from pathlib import Path
from common import ROOT,check,sha,write


def main():
    cfg,_,identity=check();pilot=Path(cfg['pilot_root']);expected=cfg['pilot_source_identity']
    assert sha(pilot/'configs/source_lock.json')==expected
    for rel,digest in json.loads((pilot/'configs/source_lock.json').read_text()).items():assert sha(pilot/rel)==digest,rel
    assert json.loads((pilot/'results/cpu_correctness.json').read_text())['status']=='PASS'
    for name in ['generation_completion.json','raw_audit.json','summary.json']:
        r=json.loads((pilot/'results'/name).read_text())
        assert r['identity']==expected and r['status'] in ('PASS','COMPLETE'),name
    result=json.loads((pilot/'results/summary.json').read_text())
    for domain,methods in result['summary'].items():
        for method in cfg['new_methods']:assert methods[method]['count']==32
    docs=[json.loads(x) for x in (pilot/'prompts/questions.jsonl').read_text().splitlines()]
    rows=json.loads((pilot/'results/per_document.json').read_text());refs={}
    for method in cfg['new_methods']:
        row=next(r for r in rows if r['method']==method and r['index']==0)
        assert sha(row['raw_path'])==row['raw_sha256']
        raw=json.loads(Path(row['raw_path']).read_text());assert sha(raw['decisions_path'])==raw['decisions_sha256']
        refs[method]=row
    for name in ['joint_hierarchy_cache.py','heap_allocator.py','libexp7_heap.so','hierarchy.py','hierarchy_cache.py','engine.py','cache.py','streaming_cache.py','joint_cache.py','recent_cache.py']:
        assert sha(ROOT/'implementations'/name)==sha(pilot/'implementations'/name),name
    write(ROOT/'configs/engineering_doc.json',docs[0]);write(ROOT/'configs/engineering_references.json',refs)
    write(ROOT/'configs/promotion_gate.json',dict(status='PASS',pilot_identity=expected,formal_identity=identity,
        pilot_summary_sha256=sha(pilot/'results/summary.json'),pilot_audit_sha256=sha(pilot/'results/raw_audit.json'),
        basis='Explicit user authorization for full comparison after valid preliminary tests; no accuracy threshold or formula tuning',
        descriptive_development_correct={d:{m:r['correct'] for m,r in methods.items()} for d,methods in result['summary'].items()}))
    print('EXPERIMENT7_DEVELOPMENT_VALID_NO_ACCURACY_GATE',expected)


if __name__=='__main__':main()
