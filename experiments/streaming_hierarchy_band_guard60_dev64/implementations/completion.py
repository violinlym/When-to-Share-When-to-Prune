"""JSON/hash-only completion check, no scoring on the GPU allocation."""
import json
from pathlib import Path
from common import ROOT, check, sha, write


def main():
    cfg, docs, identity = check()
    missing = []
    count = 0
    for method in cfg['new_methods']:
        for doc in docs:
            path = Path(cfg['store'])/'raw'/method/('%04d.json' % doc['index'])
            if not path.exists():
                missing.append([method, doc['index']])
                continue
            raw = json.loads(path.read_text())
            assert raw['identity'] == identity and raw['method'] == method
            assert raw['source_input_sha256'] == doc['source_input_sha256']
            assert sha(raw['decisions_path']) == raw['decisions_sha256']
            count += 1
    write(ROOT/'results/generation_completion.json', dict(status='INCOMPLETE' if missing else 'COMPLETE',
        identity=identity, completed_new=count, planned_new=cfg['planned_new_responses'], reused_controls=0, missing=missing))
    assert not missing, 'Missing outputs remain NA'
    print('GENERATION_COMPLETE', count)


if __name__ == '__main__':
    main()
