"""Preserve the qualified EvalPlus grader; parse original training literals."""
import argparse
import ast
import concurrent.futures
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
from frozen_code_grader import grade as original_grade


def parse_target(target):
    assert not target['setup']
    inputs, expected = [], []
    for statement in target['tests']:
        node = ast.parse(statement).body[0]
        assert isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
        comparison = node.test
        assert len(comparison.ops) == len(comparison.comparators) == 1 and isinstance(comparison.ops[0], ast.Eq)
        call = comparison.left
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        assert call.func.id == target['entry_point'] and not call.keywords
        # Parsing inside the grader preserves tuple/set/dict input types.
        inputs.append([ast.literal_eval(arg) for arg in call.args])
        expected.append(ast.literal_eval(comparison.comparators[0]))
    assert len(inputs) == 3
    return dict(entry_point=target['entry_point'], inputs=inputs, expected=expected,
        atol=0., reference_time=[0., 0., 0.], min_time_limit=target['min_time_limit'],
        gt_time_limit_factor=target['gt_time_limit_factor'])


def grade(sample):
    return original_grade(dict(sample, target=parse_target(sample['target'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', action='store_true')
    args = parser.parse_args()
    for root in ('/pfs/lustrep3/users/yiminliu', '/users/yiminliu/Project', '/flash/project_462001687'):
        assert not Path(root).exists(), root
    assert {name for _, name in socket.if_nameindex()} == {'lo'}
    os.chdir('/work')
    Path('/tmp/home').mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path('/scorer/code_contract.json').read_text())
    for name, digest in contract['modules'].items():
        module = importlib.import_module(name)
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == digest, name
    if args.fixtures:
        target = dict(entry_point='add_one', tests=['assert add_one(1)==2', 'assert add_one(3)==4', 'assert add_one(-1)==0'],
            setup='', min_time_limit=4, gt_time_limit_factor=4)
        outcomes = [grade(dict(index=i, method=name, target=target,
            prompt='"""Add one.\nassert add_one(1)==2\n"""\n', completion=code))
            for i, (name, code) in enumerate([('correct', 'def add_one(x):\n    return x+1\n'),
                                            ('incorrect', 'def add_one(x):\n    return x-1\n')])]
        assert outcomes[0]['correct'] and not outcomes[1]['correct']
        Path('code_fixtures.json').write_text(json.dumps(dict(status='PASS', rows=outcomes), indent=2)+'\n')
        print('PILOT_CODE_FIXTURES_PASS')
        return
    path = Path('code_samples.json')
    content = path.read_bytes()
    samples = json.loads(content)
    assert len(samples) == 192
    output = Path('code_results.json')
    assert not output.exists(), 'Preserve scored evidence'
    with concurrent.futures.ProcessPoolExecutor(max_workers=64, mp_context=mp.get_context('fork')) as pool:
        results = list(pool.map(grade, samples, chunksize=1))
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(status='COMPLETE', samples_sha256=hashlib.sha256(content).hexdigest(),
        rows=results), indent=2)+'\n')
    temporary.replace(output)
    print('PILOT_CODE_COMPLETE', len(results))


if __name__ == '__main__':
    main()
