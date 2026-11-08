"""Requested Experiment7 tables, from complete fixed-denominator raw evidence."""
import collections
import csv
import json
from pathlib import Path
import numpy as np


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def export(root,cfg,rows,summary,paired,development=False):
    root=Path(root);accuracy=[];allocation=[];mapping=[];memory=[]
    for domain,methods in summary.items():
        for method,entry in methods.items():
            accuracy.append(dict(dataset=domain,method=method,correct=entry['correct'],count=entry['count'],
                                 accuracy_pct=entry['accuracy_pct'],capped=entry['capped']))
            selected=[r for r in rows if r['method']==method and r.get('domain',r.get('task'))==domain]
            assert len(selected)==entry['count']
            raw=[json.loads(Path(r['raw_path']).read_text()) for r in selected]
            base=dict(dataset=domain,method=method,samples=len(selected))
            if method=='full':
                memory.append(dict(base,mean_payload_fraction=1.,min_payload_fraction=1.,max_payload_fraction=1.,mean_auxiliary_bytes=None,max_peak_allocated_gpu_bytes=None,max_peak_reserved_gpu_bytes=None,process_peak_host_rss_bytes=None,telemetry_note='Reused Full accuracy reference lacks comparable cache/memory telemetry; unmeasured fields are NA'))
                continue
            lengths=np.array([r['final_cache']['logical_tokens'] for r in raw])
            if method!='full':
                coverage=np.mean([r['final_per_layer_head_context_fraction'] for r in selected],axis=0)
                kv=np.mean([np.array(r['final_KV_slots_by_layer_head'])/n for r,n in zip(selected,lengths)],axis=0)
                for layer in range(cfg['models']['ouro']['layers']):
                    for head in range(cfg['models']['ouro']['kv_heads']):
                        allocation.append(dict(base,layer=layer,head=head,root_coverage=coverage[layer,head],
                            mean_K_slots_per_context=kv[layer,0,head],mean_V_slots_per_context=kv[layer,1,head]))
                patterns=collections.defaultdict(collections.Counter)
                for r in selected:
                    for kind,hist in r['final_mapping_histograms'].items():patterns[kind].update(hist)
                for kind,hist in patterns.items():
                    for pattern,count in sorted(hist.items()):mapping.append(dict(base,component=kind,pattern=pattern,
                        pattern_count=count,total_local_positions=sum(hist.values()),fraction=count/sum(hist.values())))
            updates=sum(r['generated_tokens'] for r in raw)
            payload=[r['final_cache']['payload_ratio'] for r in raw]
            record=dict(base,mean_payload_fraction=np.mean(payload),min_payload_fraction=min(payload),max_payload_fraction=max(payload),
                        mean_auxiliary_bytes=np.mean([r['final_cache']['auxiliary_bytes'] for r in raw]),
                        max_peak_allocated_gpu_bytes=max(r['peak_allocated_bytes'] for r in raw),
                        max_peak_reserved_gpu_bytes=max(r['peak_reserved_bytes'] for r in raw),
                        process_peak_host_rss_bytes=max(r.get('process_peak_host_rss_bytes',0) for r in raw) or None)
            for key in ['allocation_seconds','statistics_seconds','decision_seconds','cache_update_seconds']:
                values=[r.get(key) for r in raw]
                record[key.replace('_seconds','_ms_per_update')]=1000*sum(values)/updates if all(v is not None for v in values) else None
            if method=='joint_hierarchy':
                for key in ['tree_seconds','scoring_seconds','heap_seconds','transfer_mask_seconds','compaction_seconds','importance_seconds']:
                    record[key.replace('_seconds','_ms_per_update')]=1000*sum(r['mechanism'][key] for r in raw)/updates
                for key in ['actions_K_sharing','actions_V_sharing','actions_pruning','pruning_before_terminal']:
                    record[key]=sum(r['mechanism'][key] for r in raw)
                for kind,j in [('K',0),('V',1)]:
                    for boundary in range(3):record[kind+'_removed_loop'+str(boundary+1)]=sum(r['mechanism']['removed_boundaries'][j][boundary] for r in raw)
                record['max_heap_transient_bytes']=max(r['final_cache']['heap_transient_peak_bytes'] for r in raw)
            memory.append(record)
    for filename,records in [('accuracy_comparison.csv',accuracy),('allocation_statistics.csv',allocation),
                             ('version_mapping_distribution.csv',mapping),('memory_and_overhead.csv',memory)]:
        write_csv(root/filename,records)
    lines=['# Experiment7: Hierarchical Joint KV Sharing and Token Pruning','',
           'Stage: '+('fixed96 training development (MBPP original training tests, not MBPP+)' if development else 'formal frozen four-cohort evaluation')+'.',
           'All compressed groups rerun with <=50%currentFullKV native payload. Full references are reused by input/source/scorer hashes. H2O uses whole-token floor rounding; proposed heap may undershoot. Actual memory is reported rather than assumed equal occupancy.',
           '', '| Dataset | Method | Correct | Accuracy |', '|---|---|---:|---:|']
    for r in accuracy:lines.append('| %s | %s | %d/%d | %.2f%% |'%(r['dataset'],r['method'],r['correct'],r['count'],r['accuracy_pct']))
    lines+=['','## Paired differences', '', '| Dataset | Comparison | Difference pp | Descriptive95%interval |','|---|---|---:|---|']
    for domain,comparisons in paired.items():
        for comparison,v in comparisons.items():lines.append('| %s | %s | %.2f | %s |'%(domain,comparison,v['delta_pp'],v['bootstrap95_pp']))
    lines+=['','## Interpretation boundaries','',
            'H1–H3 are hypotheses, not acceptance requirements. These comparisons estimate method-level effects. Proposed versus the old hierarchy also changes importance/root allocation and scheduling; a positive difference alone cannot isolate an independent causal contribution from every component.',
            'Every live non-recent root always has a pruning candidate. With nondecreasing merge costs clipped at1 and base utility2I, a remaining cheapest sharing action has density no greater than that root’s pruning density. Sharing-first tie handling therefore normally compresses a unit to1K1V before it is pruned. This is a consequence of the stated proxy, not an extra eligibility restriction.',
            'Tree ties merge the leftmost boundary; global heap ties order Ksharing, Vsharing, pruning, then token/layer/head birth index. No posthoc formula tuning or per-domain method selection.',
            'Tree construction, scalar scoring, heap scheduling, transfers/mask preparation and physical compaction are separately reported. Host heap and scalar buffers are metadata/temporary storage, never nativeKV backups. Process hostRSS is a process-lifetime maximum, not isolated allocator allocation.',
            'Routes retain native rightward representatives. A locally deleted token is attention-masked. Global union and layer/head-local context coverage are different quantities.',
            'Accuracy includes every completed capped response using the frozen scoring rules; hitting the limit is not automatically an incorrect answer. No claim of attentionFLOPs, HBMtraffic or end-to-end acceleration is made.',
            '', 'CPU correctness: results/cpu_correctness.json in the experiment root. Native uncompressed adapter/model forward audit: per-worker qualification records in the configured raw store. Detailed raw audits and paired document outcomes: results/.',
            'Per-layer/head allocation, mapping patterns, physical payload, metadata and timing are in the four CSVs next to this report. Reproducible setup and commands are in README.md; frozen protocol/source hashes in configs/.']
    (root/'EXPERIMENT_7_REPORT.md').write_text('\n'.join(lines)+'\n')
