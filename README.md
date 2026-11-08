# When to Share, When to Prune: Joint KV Cache Compression for Looped Transformers

Research snapshot of the current CoVerKV 60%-compression prototype for Ouro-1.4B. The current implementation is named `hierarchy_price_exchange` (historical experiment name: Band-Guard). This is a research prototype, not a standalone installable release.

## Method

- Build independent K and V contiguous Complete-Linkage version trees over four loops, retaining native BF16 representatives with rightward routing.
- Allocate extra versions globally using attention importance and tree residual scores. The numerical KV budget is 40% of Full KV at the **current** context length, growing during decoding.
- Compare low-value token-root removal with the marginal excluded version score. Roots are coherent across heads and may be removed only in fixed adjacent layer pairs in the deep half (layers 12–23 for the 24-layer model).
- Protect the recent 20% of tokens from root pruning. Shallow layers retain context; freed slots are reassigned to version refinements.
- Use a one-pass exchange heuristic rather than enumeration of sharing configurations or simulation of candidate attention outputs. Proxy value is not a guarantee of final accuracy.

Core implementation: [`price_exchange_cache.py`](experiments/streaming_hierarchy_band_guard60_formal/implementations/price_exchange_cache.py), with version-tree construction in `hierarchy.py` and storage/routing in `hierarchy_cache.py` and `streaming_cache.py`.

## Verified development evidence

Fixed training-only development sets, 64 questions per domain. FTS uses **50% compression**; our candidate uses **60% compression**. This is a compression–accuracy comparison, not a matched-budget comparison. Small-sample differences do not establish a statistically reliable advantage.

| Dataset | Full KV | FTS (50% compression) | Joint prototype (60% compression) |
|---|---:|---:|---:|
| GSM training | 47/64 | 46/64 | 47/64 |
| MATH training | 53/64 | 42/64 | 42/64 |
| MBPP training | 33/64 | 28/64 | 30/64 |

See the development experiment's `REPORT.md`, `results/summary.json` and `results/per_document.json` for evidence and paired statistics.

## Formal evaluation status

Saved checkpoint on 2026-11-06: GSM8K 400/400, MATH500 367/500, MBPP+ 0/378, HumanEval 0/164: **767/1,442 generated**. GPU work stopped while waiting for allocation. These are generation counts, not accuracy results. Full benchmark superiority is **not established**. The original formal REPORT.md is preserved as historical evidence.

## Layout and reproducibility

- `experiments/streaming_hierarchy_band_guard60_dev64/`: verified development snapshot.
- `experiments/streaming_hierarchy_band_guard60_formal/`: frozen full-evaluation snapshot and corrected GPU launch script `scripts/gpu_retry.sbatch`.
- `SOURCE_MANIFEST.json`: SHA256 of every exported experiment file, unchanged from the local snapshot.

These snapshots preserve the original absolute LUMI paths, source locks, prompt manifests, scoring rules and external dependency references. They require the original environment and external files referenced by the configurations. Simply cloning this repository is **not sufficient to run the experiment**: configure an equivalent environment, obtain model/data dependencies and explicitly version any path/configuration adaptation. Do not silently rewrite frozen identity locks. The optional compiled `libexp7_heap.so` is omitted; its C++ source/build helper is included and historical locks retain its original hash. Source locks are historical experiment provenance, not a promise that all external artifacts are distributed here.

Model weights, large raw generation/cache files, logs, and checkpoint archives are not uploaded. Local checkpoint backups and missing-question manifests remain on LUMI. Do not treat missing formal outputs as incorrect answers. Resume only after GPU allocation is available, skip completed identity-matching outputs, and attach scoring to the resumed job's successful completion.

Allocator time, KV updates/data movement, total generation time and auxiliary memory must be reported separately. This repository makes no measured ASIC speedup or end-to-end GPU speedup claim.
