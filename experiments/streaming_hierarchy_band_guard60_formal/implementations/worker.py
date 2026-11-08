"""Four-benchmark rollout of the frozen Band-Guard 60%-compression candidate."""
import argparse
import fcntl
import json
import os
import time
import traceback
from pathlib import Path

rank = int(os.environ.get('SLURM_PROCID', '0'))
if 'SLURM_JOB_ID' in os.environ:
    os.environ['ROCR_VISIBLE_DEVICES'] = os.environ.get('SLURM_LOCALID', '0')
    for key in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        os.environ.pop(key, None)

from common import ROOT, TASKS, check, sha, write


def save_result(path, result, initial, final):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    state = path.with_suffix('.decisions.npz')
    temporary = state.with_name(state.name + '.tmp-' + str(os.getpid()))
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, initial=initial, final=final)
    temporary.replace(state)
    result.update(decisions_path=str(state), decisions_sha256=sha(state))
    write(path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    cfg, docs, identity = check()
    method = 'hierarchy_price_exchange'
    assert cfg['new_methods'] == [method]
    assert cfg['methods'] == ['full', 'fts', method]
    from engine import Engine
    if args.check:
        print(json.dumps(dict(status='PASS', identity=identity, method=method,
                              tasks={task: len(docs[task]) for task in TASKS},
                              payload_retention=0.40)))
        return

    import torch
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count() == 1
    store = Path(cfg['store'])
    refs = json.loads((ROOT / 'configs/full_reuse.json').read_text())
    for folder in ('raw', 'claims', 'failures', 'qualification'):
        (store / folder).mkdir(parents=True, exist_ok=True)
    units = [(task, row) for task in TASKS for row in docs[task]]
    engine = None
    qualified = False
    for offset in range(len(units)):
        task, doc = units[(rank + offset) % len(units)]
        index = doc['index']
        stem = '%s-%04d-%s' % (task, index, method)
        path = store / 'raw' / method / task / ('%04d.json' % index)
        with (store / 'claims' / (stem + '.lock')).open('a') as claim:
            try:
                fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if path.exists():
                old = json.loads(path.read_text())
                assert old['identity'] == identity
                assert old['source_input_sha256'] == doc['source_input_sha256']
                assert sha(old['decisions_path']) == old['decisions_sha256']
                continue
            try:
                if engine is None:
                    start = time.perf_counter()
                    engine = Engine(cfg, 'ouro')
                    print(json.dumps(dict(event='loaded', rank=rank,
                                          seconds=time.perf_counter() - start)), flush=True)
                reference = refs[task][str(index)]
                assert sha(reference['raw_path']) == reference['raw_sha256']
                full = json.loads(Path(reference['raw_path']).read_text())
                assert full['source_input_sha256'] == doc['source_input_sha256']
                print(json.dumps(dict(event='start', task=task, index=index, rank=rank)), flush=True)
                unit_start = time.perf_counter()
                source = engine.prepare(doc)
                assert source['input_ids'] == doc['context_ids']
                assert int(source['initial_logits'][0, -1].argmax()) == full['generated_ids'][0]
                if not qualified:
                    record = engine.qualify(source)
                    record.update(identity=identity, method=method, job=os.environ['SLURM_JOB_ID'],
                                  rank=rank, task=task, index=index)
                    write(store / 'qualification' / ('%s-%d.json' % (os.environ['SLURM_JOB_ID'], rank)), record)
                    qualified = True
                result, initial, final = engine.generate(
                    doc, source, dict(method=method, ratio=0.40))
                result.update(identity=identity, method=method, task=task, index=index,
                              doc_id=doc['doc_id'], source_input_sha256=doc['source_input_sha256'],
                              manifest_sha256=cfg['manifests'][task]['sha256'],
                              job=os.environ['SLURM_JOB_ID'], rank=rank,
                              worker_unit_seconds=time.perf_counter() - unit_start)
                save_result(path, result, initial, final)
                print(json.dumps(dict(event='done', task=task, index=index, rank=rank,
                                      generated_tokens=result['generated_tokens'],
                                      stop_reason=result['stop_reason'],
                                      seconds=result['generation_seconds'])), flush=True)
                del source
            except Exception:
                write(store / 'failures' / (stem + '-' + os.environ['SLURM_JOB_ID'] + '.json'),
                      dict(identity=identity, task=task, index=index, error=traceback.format_exc()))
                raise


if __name__ == '__main__':
    main()
