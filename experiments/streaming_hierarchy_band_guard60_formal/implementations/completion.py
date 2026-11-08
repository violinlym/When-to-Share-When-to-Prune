"""Require every frozen benchmark response and native-forward qualification."""
import json
from pathlib import Path
from common import TASKS, check, sha, write, ROOT


def main():
    cfg, docs, identity = check()
    method = cfg['new_methods'][0]
    store = Path(cfg['store'])
    missing = []
    count = 0
    for task in TASKS:
        for doc in docs[task]:
            path = store / 'raw' / method / task / ('%04d.json' % doc['index'])
            if not path.exists():
                missing.append([task, doc['index']])
                continue
            raw = json.loads(path.read_text())
            assert raw['identity'] == identity and raw['method'] == method
            assert raw['source_input_sha256'] == doc['source_input_sha256']
            assert raw['manifest_sha256'] == cfg['manifests'][task]['sha256']
            assert sha(raw['decisions_path']) == raw['decisions_sha256']
            count += 1
    qualifications = []
    for path in sorted((store / 'qualification').glob('*.json')):
        row = json.loads(path.read_text())
        if row.get('status') == 'PASS' and row.get('identity') == identity:
            qualifications.append(path.name)
    complete = not missing and count == cfg['planned_new_responses'] and qualifications
    record = dict(status='COMPLETE' if complete else 'INCOMPLETE', identity=identity,
                  planned=cfg['planned_new_responses'], completed=count,
                  reused_full=cfg['reused_responses'], missing=missing,
                  native_full_forward_qualification=qualifications)
    write(ROOT / 'results/generation_completion.json', record)
    print(json.dumps(record), flush=True)
    assert complete, 'Missing generation remains incomplete, not an incorrect answer'


if __name__ == '__main__':
    main()
