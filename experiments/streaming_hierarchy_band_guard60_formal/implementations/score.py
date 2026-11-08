"""Frozen formal scorers with current-N payload and paired outcome audits."""
import argparse
import collections
import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import types
import numpy as np
from common import ROOT, TASKS, check, sha, write
from experiment7_audit import audit

CFG = DOCS = IDENTITY = REFS = TOK = CONTROLS = None


def setup():
    global CFG, DOCS, IDENTITY, REFS, TOK, CONTROLS
    CFG, DOCS, IDENTITY = check()
    REFS = json.loads((ROOT/'configs/full_reuse.json').read_text())
    CONTROLS = json.loads((ROOT/'configs/control_reuse.json').read_text())
    sys.path.append(str(Path(CFG['scoring_root'])/'implementations'))
    site = Path('/scratch/project_462001687/yiminliu/ouro_motivation_accuracy/venv/lib/python3.12/site-packages')
    for name in ('lm_eval', 'lm_eval.tasks', 'lm_eval.tasks.hendrycks_math'):
        module = types.ModuleType(name)
        module.__path__ = [str(site.joinpath(*name.split('.')))]
        sys.modules[name] = module
    name = 'lm_eval.tasks.hendrycks_math.utils'
    spec = importlib.util.spec_from_file_location(name, site/'lm_eval/tasks/hendrycks_math/utils.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    import gsm_scoring, math_scoring
    gsm_scoring.test()
    math_scoring.test()
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(CFG['models']['ouro']['path'], trust_remote_code=True, local_files_only=True)


def inspect(unit):
    task, index, method = unit
    doc = DOCS[task][index]
    ref = REFS[task][str(index)]
    if method == 'full':
        assert sha(ref['raw_path']) == ref['raw_sha256']
        raw = json.loads(Path(ref['raw_path']).read_text())
        assert raw['doc_id'] == doc['doc_id'] and raw['source_input_sha256'] == doc['source_input_sha256']
        assert raw['method'] == 'full' and raw['generated_tokens'] == ref['generated_tokens']
        assert TOK.encode(doc['context'], add_special_tokens=False) == doc['context_ids']
        assert raw['stop_reason'] == ref['stop_reason'] and isinstance(ref['correct'], bool)
        row = dict(ref, method='full', reused=True)
        row.update(initial_payload_fraction=1., final_payload_fraction=1., final_local_any_loop_context=1.,
                   final_global_context_union=1., final_local_prompt_context=1.)
        return dict(row=row, sample=None)
    if method == 'fts':
        control = CONTROLS[method][task][str(index)]
        assert sha(control['raw_path']) == control['raw_sha256']
        raw = json.loads(Path(control['raw_path']).read_text())
        assert raw['source_input_sha256'] == doc['source_input_sha256'] and raw['method'] == 'fts'
        assert TOK.encode(doc['context'], add_special_tokens=False) == doc['context_ids'] == raw['input_ids']
        row = dict(control, reused=True)
        return dict(row=row, sample=None)
    path = Path(CFG['store'])/'raw'/method/task/('%04d.json' % index)
    if not path.exists():
        return dict(missing=[method, task, index])
    raw = json.loads(path.read_text())
    assert raw['identity'] == IDENTITY and raw['method'] == method and raw['task'] == task
    assert raw['doc_id'] == doc['doc_id'] and raw['index'] == index
    assert raw['source_input_sha256'] == doc['source_input_sha256']
    assert raw['manifest_sha256'] == CFG['manifests'][task]['sha256']
    assert TOK.encode(doc['context'], add_special_tokens=False) == doc['context_ids'] == raw['input_ids']
    ids = raw['generated_ids']
    assert 1 <= len(ids) == raw['generated_tokens'] <= doc['generation_kwargs']['max_gen_toks']
    full = json.loads(Path(ref['raw_path']).read_text())
    assert ids[0] == full['generated_ids'][0]
    assert raw['stop_reason'] in ('eos', 'length', 'stop_string')
    if raw['stop_reason'] == 'length':
        assert len(ids) == doc['generation_kwargs']['max_gen_toks']
    if raw['stop_reason'] == 'eos':
        assert ids[-1] in raw['eos_token_ids']
    text = TOK.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ends = [text.find(s) for s in doc['generation_kwargs']['until'] if s and s in text]
    if ends:
        text = text[:min(ends)]
    assert text == raw['generated_text']
    assert raw['full_prefill_numeric_staging_retained_during_decode'] is False
    assert raw['full_prefill_tensor_weakrefs_dead'] is True
    assert raw['source_release_freed_allocated_bytes'] >= raw['source_full_kv_tensor_bytes']
    row = dict(task=task, index=index, doc_id=doc['doc_id'], method=method, model='ouro', reused=False,
               raw_path=str(path), raw_sha256=sha(path), generated_tokens=len(ids), stop_reason=raw['stop_reason'],
               prompt_tokens=len(doc['context_ids']), final_local_prompt_context=raw['final_coverage']['local_prompt_context'])
    for key in ['prefill_seconds', 'generation_seconds', 'allocation_seconds', 'statistics_seconds',
                'decision_seconds', 'allocator_policy_seconds', 'cache_update_seconds',
                'prefill_attention_observer_device_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes']:
        row[key] = raw.get(key)
    row.update(audit(raw, doc, method, CFG['models']['ouro']))
    sample = None
    if task == 'gsm8k':
        import gsm_scoring
        result = gsm_scoring.evaluate(gsm_scoring.predictions(text), doc['gold'])[gsm_scoring.PRIMARY]
        row.update(correct=result['correct'], prediction=result['prediction'])
    elif task == 'math500':
        import math_scoring
        pred = math_scoring.math_prediction(text)
        row.update(correct=math_scoring.math_correct(pred, doc['gold']), prediction=pred)
    else:
        sample = dict(task_id=doc['doc_id'], method=method, prompt=doc['context'], entry_point=doc['entry_point'], completion=text)
    return dict(row=row, sample=sample)


def prepare():
    setup()
    assert CFG['status'] == 'AUTHORIZED'
    for method in CFG['new_methods']:
        qualifications = [json.loads(path.read_text()) for path in
                          (Path(CFG['store'])/'qualification').glob('*.json')]
        assert any(q.get('status') == 'PASS' and q.get('identity') == IDENTITY
                   and q.get('method') == method for q in qualifications)
    units = [(t, i, m) for t in TASKS for i in range(len(DOCS[t])) for m in CFG['methods']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=CFG['resources']['cpu_workers'], mp_context=mp.get_context('fork')) as pool:
        data = list(pool.map(inspect, units, chunksize=4))
    rows = [x['row'] for x in data if 'row' in x]
    missing = [x['missing'] for x in data if 'missing' in x]
    write(ROOT/'results/collected.json', rows)
    write(ROOT/'results/raw_audit.json', dict(status='INCOMPLETE' if missing else 'PASS', identity=IDENTITY,
          planned=4326, completed=len(rows), new_responses=1442, reused_controls=2884, missing=missing,
          every_step_payload_le_half_current_length=True, source_release=True, initial_final_routes_and_irreversibility=True))
    assert not missing and len(rows) == 4326, 'Keep missing generations NA; fixed denominator required'
    for task in ('mbpp_plus', 'humaneval'):
        work = Path(CFG['store'])/'scoring'/task
        work.mkdir(parents=True, exist_ok=True)
        assert not (work/'eval_results.json').exists(), 'Never overwrite scored evidence'
        samples = [x['sample'] for x in data if x.get('sample') and x['row']['task'] == task]
        samples.sort(key=lambda x: (int(x['task_id'].split('/')[-1]), CFG['new_methods'].index(x['method'])))
        path = work/'samples.raw.jsonl'
        path.write_text(''.join(json.dumps(x)+'\n' for x in samples))
        (work/'cache/evalplus').mkdir(parents=True, exist_ok=True)
        write(work/'identity.json', dict(task=task, source_identity=IDENTITY,
              evalplus_version=CFG['cpu_evaluator_identity']['version'], evaluator_sources=CFG['cpu_evaluator_identity'],
              official_data_sha256=CFG['datasets'][task]['provenance']['official_data_sha256'],
              selected_task_ids=sorted(d['doc_id'] for d in DOCS[task]), method_order=CFG['new_methods'], samples_raw_sha256=sha(path)))
    print('FORMAL_RAW_AUDIT_PASS', len(rows), flush=True)


def paired(rows, candidate, control):
    left = {r['index']: r for r in rows if r['method'] == candidate}
    right = {r['index']: r for r in rows if r['method'] == control}
    assert left.keys() == right.keys()
    indices = sorted(left)
    delta = np.array([int(left[i]['correct'])-int(right[i]['correct']) for i in indices])
    rng = np.random.default_rng(0)
    boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1)*100
    return dict(delta_pp=float(delta.mean()*100), bootstrap95_pp=np.quantile(boot, [.025, .975]).tolist(),
                gained=[i for i, d in zip(indices, delta) if d > 0], lost=[i for i, d in zip(indices, delta) if d < 0],
                scope='Paired task-level bootstrap; descriptive intervals, no multiplicity correction')


def finalize():
    cfg, docs, identity = check()
    rows = json.loads((ROOT/'results/collected.json').read_text())
    audit_record = json.loads((ROOT/'results/raw_audit.json').read_text())
    assert audit_record['status'] == 'PASS' and audit_record['identity'] == identity and len(rows) == 4326
    for task in ('mbpp_plus', 'humaneval'):
        work = Path(cfg['store'])/'scoring'/task
        results = json.loads((work/'eval_results.json').read_text())['eval']
        mapping = json.loads((work/'mapping.json').read_text())
        assembly = json.loads((work/'parallel_assembly.json').read_text())
        assert assembly['status'] == 'COMPLETE' and assembly['source_identity'] == identity
        assert assembly['samples_raw_sha256'] == sha(work/'samples.raw.jsonl')
        seen, scored = collections.Counter(), {}
        for m in mapping:
            tid = m['task_id']
            result = results[tid][seen[tid]]
            seen[tid] += 1
            assert hashlib.sha256(result['solution'].encode()).hexdigest() == m['solution_sha256']
            assert (tid, m['method']) not in scored
            scored[tid, m['method']] = result
        assert len(scored) == len(docs[task])*len(cfg['new_methods'])
        for row in rows:
            if row['task'] != task or row['reused']:
                continue
            r = scored[row['doc_id'], row['method']]
            row.update(correct=r['base_status'] == 'pass' and (task == 'humaneval' or r['plus_status'] == 'pass'),
                       base_status=r['base_status'], plus_status=r.get('plus_status'))
    summary, pairs = {}, {}
    for task in TASKS:
        group = [r for r in rows if r['task'] == task]
        summary[task] = {}
        for method in cfg['methods']:
            selected = [r for r in group if r['method'] == method]
            assert len(selected) == len(docs[task]) and all(isinstance(r['correct'], bool) for r in selected)
            entry = dict(correct=sum(r['correct'] for r in selected), count=len(selected),
                         accuracy_pct=100*np.mean([r['correct'] for r in selected]),
                         capped=sum(r['stop_reason'] == 'length' for r in selected))
            for key in ['initial_payload_fraction', 'final_payload_fraction', 'final_local_any_loop_context',
                        'final_local_prompt_context', 'final_global_context_union']:
                entry['mean_'+key] = float(np.mean([r[key] for r in selected]))
            if method != 'full':
                updates = sum(r['generated_tokens'] for r in selected)
                for key in ['statistics_seconds', 'decision_seconds', 'allocator_policy_seconds', 'cache_update_seconds']:
                    entry[key.replace('_seconds', '_ms_per_update')] = (None if selected[0][key] is None else
                        1000*sum(r[key] for r in selected)/updates)
                entry['tokens_per_second_prefill_plus_generation'] = updates/sum(r['prefill_seconds']+r['generation_seconds'] for r in selected)
                entry['prefill_observer_device_ms_mean'] = 1000*np.mean([r['prefill_attention_observer_device_seconds'] for r in selected])
                for key in ['final_mean_K_versions', 'final_mean_V_versions', 'final_auxiliary_to_full_fraction', 'final_per_layer_mean_versions', 'final_per_layer_head_context_fraction', 'final_KV_slots_by_layer_head']:
                    if key in selected[0]:
                        value = np.mean([r[key] for r in selected], axis=0)
                        entry['mean_'+key] = value.tolist()
            summary[task][method] = entry
        pairs[task] = {'hierarchy_price_exchange_vs_'+control:paired(group,'hierarchy_price_exchange',control) for control in ['full','fts']}
    wins = {t:pairs[t]['hierarchy_price_exchange_vs_fts']['delta_pp']>0 for t in TASKS}
    record = dict(status='COMPLETE', identity=identity, summary=summary, paired=pairs,
                  strictly_beats_fts_by_task=wins, nominal_four_task_superiority=all(wins.values()),
                  scope='Candidate retains 40% current-Full-KV payload (60% compression); FTS retains 50% (50% compression). This tests the compression-accuracy frontier, not matched-budget superiority. Paired intervals accompany nominal wins. Not75%goal completion. All testcaps remain scored; missing/technicalfailures excluded from success, not silently wrong.')
    write(ROOT/'results/per_document.json', rows)
    write(ROOT/'results/summary.json', record)
    lines = ['# CoVerKV Band-Guard: formal comparison at 60% compression', '',
             'Frozen Band-Guard Joint (Version Tree plus adjacent-layer Root pruning) retains 40% of current Full-KV payload. Full and fixed FTS references are reused by hash; FTS retains 50% payload, so it is a more generous memory reference. This is a compression-accuracy frontier comparison, not a matched-budget win. Token limits count as scored responses; truncation counts are reported separately. No posthoc per-domain winner selection.', '',
             '| Method | GSM8K400 | MATH500 | MBPP+378 | HumanEval164 |', '|---|---:|---:|---:|---:|']
    for method in cfg['methods']:
        cells = ['%d/%d (%.2f%%)' % (summary[t][method]['correct'], summary[t][method]['count'], summary[t][method]['accuracy_pct']) for t in TASKS]
        lines.append('| '+method+' | '+' | '.join(cells)+' |')
    lines += ['', 'Candidate native-KV payload is audited at 40% of the Full-KV payload for the current logical length at every update; FTS controls retain 50%. Non-reconstructive metadata, native prefill, temporary read operands and peak memory are reported separately. Policy statistics/selection, cache packing, observer device timing and overall generation are reported separately. Detailed paired gains/losses and95%intervals: `results/summary.json`.', '']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    print('FORMAL_SCORING_COMPLETE', json.dumps(wins), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'prepare', 'finalize'])
    args = parser.parse_args()
    if args.mode == 'check':
        setup()
        for task in TASKS:
            for i in range(len(DOCS[task])):
                inspect((task, i, 'full'))
                inspect((task, i, 'fts'))
        print('FORMAL_SCORER_FULL_AND_FTS_2884_INPUTS_PASS')
    elif args.mode == 'prepare':
        prepare()
    else:
        finalize()
