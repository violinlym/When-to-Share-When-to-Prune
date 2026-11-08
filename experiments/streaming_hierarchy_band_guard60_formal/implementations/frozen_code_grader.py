"""Contained original EvalPlus functions on three MBPP training assertions."""
import argparse
import concurrent.futures
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket


def grade(sample):
    from evalplus.sanitize import sanitize
    from evalplus.eval import untrusted_check
    code=sanitize(sample['prompt']+sample['completion'], entrypoint=sample['target']['entry_point'])
    target=sample['target']
    status,details=untrusted_check(dataset='mbpp',code=code,inputs=target['inputs'],
        entry_point=target['entry_point'],expected=target['expected'],atol=target['atol'],
        ref_time=target['reference_time'],fast_check=False,min_time_limit=target['min_time_limit'],
        gt_time_limit_factor=target['gt_time_limit_factor'])
    return dict(index=sample['index'],method=sample['method'],status=status,correct=status=='pass',
        details=[bool(x) for x in details],solution_sha256=hashlib.sha256(code.encode()).hexdigest())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixtures',action='store_true');args=parser.parse_args()
    for root in ['/pfs/lustrep3/users/yiminliu','/users/yiminliu/Project','/flash/project_462001687']:
        assert not Path(root).exists(),root
    assert {name for _,name in socket.if_nameindex()}=={'lo'}
    os.chdir('/work');Path('/tmp/home').mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path('/scorer/code_contract.json').read_text())
    for name,digest in contract['modules'].items():
        module=importlib.import_module(name)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==digest,name
    if args.fixtures:
        target=dict(entry_point='add_one',inputs=[[1],[3]],expected=[2,4],atol=0.,reference_time=[0.,0.],min_time_limit=4,gt_time_limit_factor=4)
        result=[grade(dict(index=i,method=name,prompt='"""Add one.\nassert add_one(1)==2\n"""\n',completion=code,target=target)) for i,(name,code) in enumerate([
            ('correct','def add_one(x):\n    return x+1\n'),('incorrect','def add_one(x):\n    return x-1\n')])]
        assert result[0]['correct'] and not result[1]['correct']
        Path('code_fixtures.json').write_text(json.dumps(dict(status='PASS',results=result,contract=contract),indent=2)+'\n')
        print('ISOLATED_TRAINING_CODE_FIXTURES_PASS');return
    path=Path('code_samples.json');data=path.read_bytes();samples=json.loads(data)
    assert len(samples)==24
    output=Path('code_results.json')
    assert not output.exists(),'Preserve scored evidence'
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(64,len(samples)),mp_context=mp.get_context('fork')) as pool:
        rows=list(pool.map(grade,samples,chunksize=1))
    temp=output.with_suffix('.tmp');temp.write_text(json.dumps(dict(status='COMPLETE',samples_sha256=hashlib.sha256(data).hexdigest(),rows=rows),indent=2)+'\n');temp.replace(output)
    print('TRAINING_CODE_COMPLETE',len(rows))


if __name__=='__main__':main()
