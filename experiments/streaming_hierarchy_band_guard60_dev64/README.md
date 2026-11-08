# Band-Guard Joint Compression at 60%

This training-only successor follows the pre-declared conditional path:

1. Pure Version Tree at 40% current-length Full-KV payload was not stable on MATH.
2. Scattered single-layer Peak-Guard pruning recovered MATH but did not replicate on GSM8K.
3. This branch changes only the pruning structure: a token root may be removed only
   from a fixed adjacent pair of deep layers when both layers agree it is low risk.

Version allocation, K/V-independent Complete-Linkage trees, native rightward
routing, cumulative-attention statistics, recent-20% protection, 10% pruning cap,
and the global current-length payload formula remain unchanged. Released root
slots buy the highest-valued excluded Version-Tree refinements under the same
global budget.

The candidate retains exactly `ceil(0.40 * current Full KV components)` at each
decode length, corresponding to 60% compression. The reused FTS control retains
50% payload and is therefore an easier-budget reference, not a matched-budget
baseline. This is a fixed 64-example-per-domain training mechanism test; it is
not a formal benchmark.
