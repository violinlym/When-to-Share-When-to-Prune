"""Score the fixed 64/domain Version-Tree-60 mechanism test."""
import argparse
import json
from pathlib import Path
import sys

from common import ROOT, check, sha, write
from frozen_math_helpers import math_setup, repeat


def audit_payload(raw, doc, method, mc):
    import numpy as np
    prompt = len(doc['context_ids'])
    final_n = prompt + raw['generated_tokens'] - 1
    vector_bytes = mc['head_dim'] * 2
    full_components_per_token = mc['layers'] * 2 * mc['loops'] * mc['kv_heads']
    records = [[prompt, raw['prefill_cache']['payload_bytes'], raw['prefill_cache']['full_payload_bytes']]] + raw['byte_history']
    assert [x[0] for x in records] == list(range(prompt, final_n + 1))
    for n, payload, full in records:
        assert full == full_components_per_token * n * vector_bytes
        if method == 'full':
            assert payload == full
        elif method == 'fts':
            assert payload == full_components_per_token * ((n + 1) // 2) * vector_bytes
        else:
            target_components = (2 * full_components_per_token * n + 4) // 5
            assert payload == target_components * vector_bytes
    row = {}
    if method in ('hierarchy_version', 'hierarchy_price_exchange'):
        assert raw['decision_format'] == 'loop_source_versions'
        assert sha(raw['decisions_path']) == raw['decisions_sha256']
        with np.load(raw['decisions_path'], allow_pickle=False) as state:
            initial, final = state['initial'], state['final']
            assert initial.shape == (mc['layers'], 2, mc['kv_heads'], prompt, 4)
            assert final.shape == (mc['layers'], 2, mc['kv_heads'], final_n, 4)
            assert (initial <= 4).all() and (final <= 4).all()
            assert (np.diff(initial.astype('int16'), axis=-1) >= 0).all()
            assert (np.diff(final.astype('int16'), axis=-1) >= 0).all()
            assert (final[..., :prompt, :] >= initial).all()
            alive = final[..., 3] < 4
            if method == 'hierarchy_version':
                assert alive.all()
            else:
                assert np.array_equal(alive[:, 0], alive[:, 1])
                assert (final[~alive] == 4).all()
                # A root is shared by K/V and all heads for one layer-token.
                root_alive = alive[:, 0, 0]
                assert np.array_equal(np.broadcast_to(root_alive[:, None, :], alive[:, 0].shape), alive[:, 0])
                pruned_roots = int((~root_alive).sum())
                # The runtime field counts bands removed in the last allocate
                # call; roots are cumulative. Reconstruct cumulative bands
                # from the persisted route trace and verify pair coherence.
                deep_start = mc['layers'] // 2
                pruned_bands = 0
                for first in range(deep_start, mc['layers'] - 1, 2):
                    left = ~root_alive[first]
                    right = ~root_alive[first + 1]
                    assert np.array_equal(left, right)
                    pruned_bands += int(left.sum())
                assert pruned_roots == 2 * pruned_bands
                assert raw['final_cache']['pruned_layer_token_roots'] == pruned_roots
            counts = (1 + (np.diff(final, axis=-1) != 0).sum(-1)) * alive
            row.update(final_mean_K_versions=float(counts[:, 0].mean()),
                       final_mean_V_versions=float(counts[:, 1].mean()),
                       final_per_layer_mean_versions=counts.mean(axis=(1, 2, 3)).tolist(),
                       final_component_route_disagreement=float((final[:, 0] != final[:, 1]).any(axis=-1).mean()))
            if method == 'hierarchy_price_exchange':
                # Each removed layer-token root releases one K and one V slot
                # for every KV head. Under a fixed total budget these slots are
                # exactly the additional Version-Tree refinements it purchases.
                row.update(final_local_context_coverage=float(root_alive.mean()),
                           final_pruned_layer_token_roots=pruned_roots,
                           final_pruned_adjacent_layer_bands=pruned_bands,
                           refinements_purchased_by_pruning=pruned_roots * 2 * mc['kv_heads'],
                           final_selected_K_extra_versions=int((counts[:, 0] - alive[:, 0]).sum()),
                           final_selected_V_extra_versions=int((counts[:, 1] - alive[:, 1]).sum()))
        if method == 'hierarchy_version':
            assert abs(raw['final_coverage']['local_any_loop_context'] - 1.) < 1e-6
        else:
            assert 0. < raw['final_coverage']['local_any_loop_context'] <= 1.
        assert abs(raw['final_coverage']['global_context_union'] - 1.) < 1e-6
        assert raw['hierarchy_initial']['status'] == raw['hierarchy_final']['status'] == 'PASS'
    row.update(final_payload_fraction=raw['final_cache']['payload_ratio'],
               final_auxiliary_to_full_fraction=raw['final_cache'].get('auxiliary_bytes', 0) / raw['final_cache']['full_payload_bytes'])
    return row


def paired(rows, candidate, control):
    import numpy as np
    by = {(r['index'], r['method']): r for r in rows}
    ids = sorted(i for i, m in by if m == candidate)
    delta = np.array([int(by[i, candidate]['correct']) - int(by[i, control]['correct']) for i in ids])
    rng = np.random.default_rng(20261105)
    boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(1) * 100
    return dict(delta_pp=float(delta.mean() * 100), bootstrap95_pp=np.quantile(boot, [.025, .975]).tolist(),
                recovered=[i for i, d in zip(ids, delta) if d > 0], regressed=[i for i, d in zip(ids, delta) if d < 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'finalize'))
    args = parser.parse_args()
    cfg, docs, identity = check()
    sys.path.insert(0, str(Path(cfg['scoring_root']) / 'implementations'))
    math_setup()
    import gsm_scoring, math_scoring
    gsm_scoring.test(); math_scoring.test()
    targets = json.loads((ROOT / 'prompts/targets.json').read_text())
    store = Path(cfg['store'])
    work = store / 'scoring'
    if args.mode == 'prepare':
        rows, samples = [], []
        for doc in docs:
            target = targets[str(doc['source_manifest_index'])]
            for method in cfg['methods']:
                source_store = Path(cfg['control_store']) if method in ('full', 'fts') else store
                path = source_store / 'raw' / method / ('%04d.json' % doc['index'])
                raw = json.loads(path.read_text())
                assert raw['method'] == method
                if method == 'hierarchy_price_exchange': assert raw['identity'] == identity
                assert raw['source_input_sha256'] == doc['source_input_sha256']
                text = raw['generated_text']
                row = dict(index=doc['index'], task=doc['task'], domain=doc['domain'], doc_id=doc['doc_id'], method=method,
                           raw_path=str(path), raw_sha256=sha(path), generated_tokens=raw['generated_tokens'],
                           stop_reason=raw['stop_reason'], tail_repeat=repeat(raw['generated_ids']),
                           generation_seconds=raw['generation_seconds'], allocation_seconds=raw.get('allocation_seconds'),
                           statistics_seconds=raw.get('statistics_seconds'), decision_seconds=raw.get('decision_seconds'),
                           allocator_policy_seconds=raw.get('allocator_policy_seconds'), cache_update_seconds=raw.get('cache_update_seconds'))
                row.update(audit_payload(raw, doc, method, cfg['models']['ouro']))
                if target['kind'] == 'numeric':
                    result = gsm_scoring.evaluate(gsm_scoring.predictions(text), target['gold'])[gsm_scoring.PRIMARY]
                    row.update(correct=result['correct'], prediction=result['prediction'])
                elif target['kind'] == 'boxed_math':
                    prediction = math_scoring.math_prediction(text)
                    gold = math_scoring.math_prediction(target['reference_solution'])
                    row.update(correct=math_scoring.math_correct(prediction, gold), prediction=prediction)
                else:
                    samples.append(dict(index=doc['index'], method=method, prompt=doc['context'], completion=text, target=target))
                    row.update(correct=None, prediction=None)
                rows.append(row)
        assert len(rows) == 576 and len(samples) == 192
        write(ROOT / 'results/collected.json', rows)
        write(work / 'code_samples.json', samples)
        write(ROOT / 'results/raw_audit.json', dict(status='PASS', identity=identity, responses=576,
              every_step_current_length_payload=True, hierarchy_context_coverage=1.0,
              hierarchy_retention=.4, fts_retention=.5, full_retention=1.0,
              reused_controls=384, new_candidate_responses=192))
        print('RAW_AUDIT_PASS 576 responses; 192 code samples prepared')
        return
    rows = json.loads((ROOT / 'results/collected.json').read_text())
    code = json.loads((work / 'code_results.json').read_text())
    assert code['status'] == 'COMPLETE' and code['samples_sha256'] == sha(work / 'code_samples.json')
    code_rows = {(r['index'], r['method']): r for r in code['rows']}
    assert len(code_rows) == 192
    for row in rows:
        if row['task'] == 'mbpp_plus':
            row.update(correct=code_rows[row['index'], row['method']]['correct'],
                       prediction=code_rows[row['index'], row['method']]['status'])
        assert isinstance(row['correct'], bool)
    summary, comparisons = {}, {}
    for domain in sorted({r['domain'] for r in rows}):
        group = [r for r in rows if r['domain'] == domain]
        summary[domain] = {}
        for method in cfg['methods']:
            selected = [r for r in group if r['method'] == method]
            assert len(selected) == 64
            correct = sum(r['correct'] for r in selected)
            entry = dict(correct=correct, count=64, accuracy_pct=100 * correct / 64,
                         capped=sum(r['stop_reason'] == 'length' for r in selected))
            if method != 'full':
                entry.update(mean_final_payload_fraction=sum(r['final_payload_fraction'] for r in selected) / 64,
                             generated_tokens_per_second=sum(r['generated_tokens'] for r in selected) / sum(r['generation_seconds'] for r in selected))
            if method == 'hierarchy_price_exchange':
                for key in ('final_mean_K_versions', 'final_mean_V_versions', 'final_component_route_disagreement',
                            'final_auxiliary_to_full_fraction', 'final_local_context_coverage',
                            'final_pruned_layer_token_roots', 'final_pruned_adjacent_layer_bands',
                            'refinements_purchased_by_pruning', 'final_selected_K_extra_versions',
                            'final_selected_V_extra_versions'):
                    entry['mean_' + key] = sum(r[key] for r in selected) / 64
                updates = sum(r['generated_tokens'] for r in selected)
                for key in ('statistics_seconds', 'decision_seconds', 'allocator_policy_seconds', 'cache_update_seconds'):
                    entry[key.replace('_seconds', '_ms_per_update')] = 1000 * sum(r[key] for r in selected) / updates
            summary[domain][method] = entry
        comparisons[domain] = dict(joint60_vs_fts=paired(group, 'hierarchy_price_exchange', 'fts'),
                                   joint60_vs_full=paired(group, 'hierarchy_price_exchange', 'full'))
    result = dict(status='COMPLETE', identity=identity, summary=summary, paired=comparisons,
                  interpretation='Fixed training-only 64/domain mechanism test. The joint candidate uses 40% current-N Full-KV payload (60% compression); FTS uses an easier 50% payload. The candidate preserves global context union and prunes only bounded deep-layer Roots outside recent20%. Descriptive paired bootstrap intervals; no formal-test claim.')
    write(ROOT / 'results/per_document.json', rows)
    write(ROOT / 'results/summary.json', result)
    lines = ['# Band-Guard Version Tree + Root Exchange at 60% Compression', '', result['interpretation'], '',
             '| Domain | Full | FTS (50% compression) | Band-Guard Joint (60% compression) | Joint−FTS |', '|---|---:|---:|---:|---:|']
    for domain in sorted(summary):
        s = summary[domain]; delta = comparisons[domain]['joint60_vs_fts']['delta_pp']
        lines.append(f"| {domain} | {s['full']['correct']}/64 ({s['full']['accuracy_pct']:.2f}%) | {s['fts']['correct']}/64 ({s['fts']['accuracy_pct']:.2f}%) | {s['hierarchy_price_exchange']['correct']}/64 ({s['hierarchy_price_exchange']['accuracy_pct']:.2f}%) | {delta:+.2f} pp |")
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
