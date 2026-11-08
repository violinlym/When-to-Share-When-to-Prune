import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('gsm8k', 'math500', 'mbpp_plus', 'humaneval')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def check():
    cfg = json.loads((ROOT/'configs/protocol.json').read_text())
    lock = ROOT/'configs/source_lock.json'
    for path, digest in json.loads(lock.read_text()).items():
        assert sha(ROOT/path) == digest, path
    assert cfg['status'] in ('PREPARED','AUTHORIZED')
    for path, digest in json.loads((ROOT/'configs/dependencies.json').read_text()).items():
        assert sha(path) == digest, path
    docs = {}
    for task in TASKS:
        path = ROOT/'prompts'/(task+'.jsonl')
        assert sha(path) == cfg['manifests'][task]['sha256']
        docs[task] = [json.loads(x) for x in path.read_text().splitlines()]
        assert len(docs[task]) == cfg['tasks'][task]
        for i, row in enumerate(docs[task]):
            row.setdefault('index', i)
            assert row['index'] == i and row['task'] == task
            for key in ('max_gen_toks','until','do_sample'):
                assert row['generation_kwargs'][key] == cfg['generation'][task][key]
    return cfg, docs, sha(lock)
