"""Fixed training-cohort paired scores and independent capacity/route audits."""
import argparse
import json
from pathlib import Path
import sys
from common import ROOT, check, sha, write
from frozen_math_helpers import math_setup, repeat


def audit(raw, doc, method, mc):
    import numpy as np
    prompt = len(doc['context_ids'])
    n = prompt + raw['generated_tokens'] - 1
    per_token = mc['layers']*2*mc['loops']*mc['kv_heads']*mc['head_dim']*2
    assert raw['input_ids'] == doc['context_ids']
    records = [[prompt, raw['prefill_cache']['payload_bytes'], raw['prefill_cache']['full_payload_bytes']]] + raw['byte_history']
    assert [x[0] for x in records] == list(range(prompt, n+1))
    for length, payload, full in records:
        assert full == per_token*length
        assert payload == per_token*((length+1)//2), (method, length, payload, full)
    assert sha(raw['decisions_path']) == raw['decisions_sha256']
    row = {}
    with np.load(raw['decisions_path'], allow_pickle=False) as state:
        initial, final = state['initial'], state['final']
        for phase, name in [('initial', 'prefill_cache'), ('final', 'final_cache')]:
            data, memory = state[phase], raw[name]
            length = prompt if phase == 'initial' else n
            capacity = (length+1)//2
            assert memory['logical_tokens'] == length
            assert memory['payload_bytes'] == per_token*capacity
            assert memory['full_payload_bytes'] == per_token*length
            assert memory['payload_ratio'] == capacity/length
            row[phase+'_payload_fraction'] = capacity/length
            row[phase+'_auxiliary_to_full_fraction'] = memory['auxiliary_bytes']/(per_token*length)
            if method in ('bound', 'loop'):
                assert raw['decision_format'] == 'token_positions'
                assert data.shape == (mc['layers'], 4, mc['kv_heads'], capacity)
                assert (data >= 0).all() and (data < length).all() and (np.diff(data, axis=-1) > 0).all()
                recent = capacity//2
                if recent:
                    assert (data[..., -recent:] == np.arange(length-recent, length)).all()
                if method == 'bound':
                    assert (data == data[:, :1]).all()
                live = np.zeros((mc['layers'], 4, mc['kv_heads'], length), dtype=bool)
                np.put_along_axis(live, data, True, axis=-1)
                row[phase+'_local_any_loop_context'] = float(live.any(axis=1).mean())
                row[phase+'_global_context_union'] = float(live.any(axis=(0, 1, 2)).mean())
            else:
                assert raw['decision_format'] == 'loop_source_versions'
                assert data.shape == (mc['layers'], 2, mc['kv_heads'], length, 4)
                assert (data >= np.arange(4)).all() and (data <= 4).all()
                alive=data[..., 3] < 4
                assert (data[~alive] == 4).all()
                assert np.array_equal(alive[:,0],alive[:,1])
                if method not in ('joint','global_context','global_context_stable','exposure_global','pooled_density','recent_density'):assert alive.all()
                if method in ('global_context','global_context_stable','exposure_global','pooled_density','recent_density'):assert (alive==alive[0,0,0]).all()
                if method == 'hierarchy_version':
                    h=raw['hierarchy_'+('initial' if phase=='initial' else 'final')]
                    assert h['status']=='PASS' and h['logical_tokens']==length
                    assert h['component_count']==mc['layers']*2*mc['kv_heads']*length
                    assert h['all_context_retained'] and h['nested_available_anchors']
                if method == 'recent_density':
                    assert alive[..., -((length+4)//5):].all(), 'Recent context was evicted'
                assert (np.diff(data.astype('int16'), axis=-1) >= 0).all()
                counts = (1+(np.diff(data, axis=-1) != 0).sum(-1))*alive
                assert int(counts.sum())*mc['head_dim']*2 == memory['payload_bytes']
                assert memory['counterfactual_attention_forwards'] == 0
                assert memory['sharing_configurations_enumerated'] == 0
                if method == 'fts':
                    ordinary = data[..., :-1, :] if length % 2 else data
                    assert (ordinary == np.array([0, 3, 3, 3])).all()
                    if length % 2:
                        assert (data[..., -1, :] == np.arange(4)).all()
                row[phase+'_local_any_loop_context'] = float(alive[:,0].mean())
                row[phase+'_global_context_union'] = float(alive[:,0].any(axis=(0,1)).mean())
                row[phase+'_per_layer_context_fraction'] = alive[:,0].mean(axis=(1,2)).tolist()
                row[phase+'_K_versions_per_retained_context'] = float(counts[:,0].sum()/alive[:,0].sum())
                row[phase+'_V_versions_per_retained_context'] = float(counts[:,1].sum()/alive[:,1].sum())
                row[phase+'_component_route_disagreement'] = float((data[:, 0] != data[:, 1]).any(axis=-1).mean())
                row[phase+'_mean_K_versions'] = float(counts[:, 0].mean())
                row[phase+'_mean_V_versions'] = float(counts[:, 1].mean())
                row[phase+'_per_layer_mean_versions'] = counts.mean(axis=(1, 2, 3)).tolist()
                row[phase+'_version_count_histogram'] = {str(k): int((counts == k).sum()) for k in range(5)}
        if method in ('bound', 'loop'):
            # Every retained original prompt token was already retained after prefill.
            for before, after in zip(initial.reshape(-1, initial.shape[-1]), final.reshape(-1, final.shape[-1])):
                assert np.isin(after[after < prompt], before).all()
        else:
            # Removed native representatives cannot return; routing only moves later.
            assert (final[..., :prompt, :] >= initial).all()
            if n == prompt:
                assert np.array_equal(initial, final)
    for phase, raw_phase in [('initial', 'prefill_coverage'), ('final', 'final_coverage')]:
        assert abs(row[phase+'_local_any_loop_context']-raw[raw_phase]['local_any_loop_context']) < 1e-6
        assert abs(row[phase+'_global_context_union']-raw[raw_phase]['global_context_union']) < 1e-6
    return row


def paired(group, candidate, control):
    import numpy as np
    rows = {(r['index'], r['method']): r for r in group}
    indices = sorted(r['index'] for r in group if r['method'] == candidate)
    delta = np.array([int(rows[i, candidate]['correct'])-int(rows[i, control]['correct']) for i in indices])
    rng = np.random.default_rng(0)
    boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)*100
    return dict(delta_pp=float(delta.mean()*100), bootstrap95_pp=np.quantile(boot, [.025, .975]).tolist(),
        recovered=[i for i, d in zip(indices, delta) if d > 0], regressed=[i for i, d in zip(indices, delta) if d < 0],
        interpretation='32 paired training items; descriptive bootstrap, no multiple-comparison correction')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'prepare', 'finalize'])
    args = parser.parse_args()
    cfg, docs, identity = check()
    sys.path.insert(0, str(Path(cfg['scoring_root'])/'implementations'))
    math_setup()
    import gsm_scoring, math_scoring
    from code_only import parse_target
    gsm_scoring.test()
    math_scoring.test()
    targets = json.loads((ROOT/'prompts/targets.json').read_text())
    refs = json.loads((ROOT/'configs/full_reuse.json').read_text())
    controls=json.loads((ROOT/'configs/control_reuse.json').read_text())
    assert len(controls)==1632
    for row in controls:assert sha(row['raw_path'])==row['raw_sha256']
    for doc in docs:
        target = targets[str(doc['index'])]
        if target['kind'] == 'boxed_math':
            assert math_scoring.math_prediction(target['reference_solution']) is not None
        elif target['kind'] == 'mbpp_training_original_tests':
            parse_target(target)
        ref = refs[str(doc['index'])]
        assert sha(ref['raw_path']) == ref['raw_sha256']
        assert ref['index'] == doc['index'] and ref['doc_id'] == doc['doc_id']
        full = json.loads(Path(ref['raw_path']).read_text())
        assert full['source_input_sha256'] == doc['source_input_sha256']
    if args.mode == 'check':
        print('SCORING_INPUTS_PASS 96 same-input Full references and targets; no generated code executed')
        return
    store = Path(cfg['store'])
    work = store/'scoring'
    if args.mode == 'prepare':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg['models']['ouro']['path'], trust_remote_code=True, local_files_only=True)
        missing = [(m, d['index']) for m in cfg['new_methods'] for d in docs
                   if not (store/'raw'/m/('%04d.json' % d['index'])).exists()]
        if missing:
            write(ROOT/'results/raw_audit.json', dict(status='INCOMPLETE', missing=missing, identity=identity))
            raise RuntimeError('Missing outputs remain NA; full fixed denominator required')
        replay=json.loads((store/'qualification/engineering-replay-pass.json').read_text())
        assert replay['status']=='PASS' and replay['identity']==identity
        rows, samples = [], []
        for doc in docs:
            index = doc['index']
            target = targets[str(index)]
            ref = refs[str(index)]
            full = json.loads(Path(ref['raw_path']).read_text())
            rows.extend(dict(control,reused=True) for control in controls if control['index']==index)
            for method in cfg['new_methods']:
                path = store/'raw'/method/('%04d.json' % index)
                raw = json.loads(path.read_text())
                assert raw['identity'] == identity and raw['method'] == method
                assert raw['source_input_sha256'] == doc['source_input_sha256']
                assert raw['manifest_sha256'] == cfg['manifests']['questions.jsonl']
                assert raw['index'] == index and raw['doc_id'] == doc['doc_id']
                ids = raw['generated_ids']
                assert ids[0] == full['generated_ids'][0]
                assert 1 <= len(ids) == raw['generated_tokens'] <= doc['generation_kwargs']['max_gen_toks']
                assert raw['stop_reason'] in ('length', 'eos', 'stop_string')
                if raw['stop_reason'] == 'length':
                    assert len(ids) == doc['generation_kwargs']['max_gen_toks']
                if raw['stop_reason'] == 'eos':
                    assert ids[-1] in raw['eos_token_ids']
                text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                ends = [text.find(s) for s in doc['generation_kwargs']['until'] if s and s in text]
                if ends:
                    text = text[:min(ends)]
                assert text == raw['generated_text']
                row = dict(index=index, domain=doc['domain'], task=doc['task'], method=method, doc_id=doc['doc_id'],
                    raw_path=str(path), raw_sha256=sha(path), generated_tokens=len(ids), stop_reason=raw['stop_reason'],
                    tail_repeat=repeat(ids), reused=False, generation_seconds=raw['generation_seconds'],
                    allocation_seconds=raw.get('allocation_seconds'), decision_seconds=raw.get('decision_seconds'),
                    statistics_seconds=raw.get('statistics_seconds'), allocator_policy_seconds=raw.get('allocator_policy_seconds'),
                    cache_update_seconds=raw.get('cache_update_seconds'), peak_allocated_bytes=raw['peak_allocated_bytes'], peak_reserved_bytes=raw['peak_reserved_bytes'])
                assert raw['full_prefill_numeric_staging_retained_during_decode'] is False
                assert raw['full_prefill_tensor_weakrefs_dead'] is True
                assert raw['source_release_freed_allocated_bytes']>=raw['source_full_kv_tensor_bytes']
                row['prefill_attention_observer_device_seconds']=raw['prefill_attention_observer_device_seconds']
                row.update(audit(raw, doc, method, cfg['models']['ouro']))
                if target['kind'] == 'numeric':
                    result = gsm_scoring.evaluate(gsm_scoring.predictions(text), target['gold'])[gsm_scoring.PRIMARY]
                    row.update(correct=result['correct'], prediction=result['prediction'])
                elif target['kind'] == 'boxed_math':
                    prediction = math_scoring.math_prediction(text)
                    gold = math_scoring.math_prediction(target['reference_solution'])
                    row.update(correct=math_scoring.math_correct(prediction, gold), prediction=prediction)
                else:
                    samples.append(dict(index=index, method=method, prompt=doc['context'], completion=text, target=target))
                rows.append(row)
        assert len(rows) == 1728 and len(samples) == 32
        write(ROOT/'results/hierarchy_audit.json', dict(status='PASS',identity=identity,responses=96,scope='All contexts retained, native hierarchical cuts checked at initial/final state; every-step payload and no resurrection audited. Changed version choices do not promise capped_version output equivalence.'))
        write(ROOT/'results/collected.json', rows)
        write(work/'code_samples.json', samples)
        write(ROOT/'results/raw_audit.json', dict(status='PASS', identity=identity, new_responses=96, reused_controls=1632,
            all_decode_steps_exact_current_length_payload=True, initial_final_routes_and_irreversibility=True,
            importance_denominator='N; original pooled capped-version score; no future queries used'))
        print('RAW_AUDIT_PASS 96 new + 1632 reused controls')
        return
    import numpy as np
    rows = json.loads((ROOT/'results/collected.json').read_text())
    code = json.loads((work/'code_results.json').read_text())
    assert code['status'] == 'COMPLETE' and code['samples_sha256'] == sha(work/'code_samples.json')
    code_rows = {(r['index'], r['method']): r for r in code['rows']}
    assert len(code_rows) == 32
    for row in rows:
        if not row['reused'] and row['domain'] == 'mbpp_training_pilot':
            row.update({k: v for k, v in code_rows[row['index'], row['method']].items() if k not in ('index', 'method')})
        assert isinstance(row['correct'], bool)
    summary, pairs = {}, {}
    for domain in sorted({r['domain'] for r in rows}):
        group = [r for r in rows if r['domain'] == domain]
        summary[domain], pairs[domain] = {}, {}
        for method in cfg['methods']:
            selected = [r for r in group if r['method'] == method]
            assert len(selected) == 32
            entry = dict(correct=sum(r['correct'] for r in selected), count=32,
                accuracy_pct=100*sum(r['correct'] for r in selected)/32,
                capped=sum(r['stop_reason'] == 'length' for r in selected),
                high_repeat=sum(r['tail_repeat'] >= .8 for r in selected))
            if method != 'full':
                for key in ('initial_payload_fraction', 'final_payload_fraction', 'final_auxiliary_to_full_fraction',
                            'final_local_any_loop_context', 'final_global_context_union', 'generation_seconds'):
                    entry['mean_'+key] = float(np.mean([r[key] for r in selected]))
                entry['end_to_end_generated_tokens_per_second'] = sum(r['generated_tokens'] for r in selected)/sum(r['generation_seconds'] for r in selected)
                if method in ('fts', 'residual', 'attention', 'combined', 'capped_version', 'joint', 'global_context', 'global_context_stable', 'exposure_version', 'exposure_global', 'head_version', 'head_loop_version', 'pooled_density', 'recent_density', 'hierarchy_version'):
                    entry['allocator_policy_ms_per_cache_update'] = 1000*sum(r['allocator_policy_seconds'] for r in selected)/sum(r['generated_tokens'] for r in selected)
                    entry['allocator_policy_fraction_of_generation_wall'] = sum(r['allocator_policy_seconds'] for r in selected)/sum(r['generation_seconds'] for r in selected)
                    entry['statistics_ms_per_cache_update'] = 1000*sum(r['statistics_seconds'] for r in selected)/sum(r['generated_tokens'] for r in selected)
                    entry['decision_ms_per_cache_update'] = 1000*sum(r['decision_seconds'] for r in selected)/sum(r['generated_tokens'] for r in selected)
                    entry['allocation_including_packing_ms_per_cache_update'] = 1000*sum(r['allocation_seconds'] for r in selected)/sum(r['generated_tokens'] for r in selected)
                    entry['decision_fraction_of_generation_wall'] = sum(r['decision_seconds'] for r in selected)/sum(r['generation_seconds'] for r in selected)
                    entry['allocation_fraction_of_generation_wall'] = sum(r['allocation_seconds'] for r in selected)/sum(r['generation_seconds'] for r in selected)
                for key in ('final_component_route_disagreement', 'final_mean_K_versions', 'final_mean_V_versions'):
                    if key in selected[0]:
                        entry['mean_'+key] = float(np.mean([r[key] for r in selected]))
            summary[domain][method] = entry
        for candidate in cfg['new_methods']:
            for control in ('capped_version','recent_density','global_context_stable','full','fts','bound','loop'):
                if candidate != control:
                    pairs[domain][candidate+'_vs_'+control] = paired(group, candidate, control)
    interpretation = ('Fixed 96 training inputs, 32/domain; MBPP uses original training tests, not official MBPP+. '
        'All compared compressed methods have identical payload at each same logical length: 4*ceil(N/2) pair-slots/layer/head. '
        'Routing and scalar selection statistics reported separately. Native full prefill retained only as independent-condition staging; '
        'the new condition and prior joint-pilot conditions release staging before decode, verified by weakrefs and allocated bytes; some older reused controls retain unused staging. Fixed engineering replay verifies the harness preserves the prior capped_version response and routes. Peak allocation includes prefill and dense transient reads. No end-to-end GPU-memory or hardware-speedup claim. '
        'Pairwise intervals are descriptive and exploratory. Formal four-benchmark superiority is not established by this pilot.')
    result = dict(status='COMPLETE', identity=identity, planned_total=1728, summary=summary, paired=pairs, interpretation=interpretation)
    write(ROOT/'results/per_document.json', rows)
    write(ROOT/'results/summary.json', result)
    lines = ['# Joint context/version density: fixed training pilot', '', interpretation, '', '| Domain | Method | Correct / 32 | Accuracy |', '|---|---|---:|---:|']
    for domain, methods in summary.items():
        for method, entry in methods.items():
            lines.append('| %s | %s | %d | %.2f%% |' % (domain, method, entry['correct'], entry['accuracy_pct']))
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
