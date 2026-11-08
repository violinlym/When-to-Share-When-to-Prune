# CoVerKV 60% Compression Decision Record

All candidate payloads below are charged against the **current logical context
length**. A 60% compression candidate retains 40% of the native four-loop Full
KV payload at every prefill/decode step. FTS retains 50% payload and is therefore
an easier-budget reference, not an equal-budget baseline.

## Gate 1: version redundancy only

The frozen context-preserving Version Tree was first tested without token
pruning on 64 fixed training examples per domain.

| Domain | Full | FTS (50% compression) | Version Tree (60% compression) | VT − FTS |
|---|---:|---:|---:|---:|
| GSM8K | 47/64 | 46/64 | 48/64 | +2 questions |
| MATH | 53/64 | 42/64 | 40/64 | -2 questions |
| MBPP+ | 33/64 | 28/64 | 31/64 | +3 questions |

Conclusion: cross-loop version redundancy alone remains useful at 60%, but is
not uniformly sufficient. The MATH failure activates the pre-declared joint
version-sharing plus token-pruning branch.

## Gate 2: scattered single-layer root exchange

Peak-Guard prunes low-attention roots independently in eligible deep layers and
uses the released slots for Version-Tree refinements. It was frozen between two
non-overlapping 64-example training slices.

| Domain | Full (128) | FTS (128) | Peak-Guard (128) | Joint − FTS |
|---|---:|---:|---:|---:|
| GSM8K | 99 | 97 | 94 | -3 questions |
| MATH | 96 | 85 | 88 | +3 questions |
| MBPP+ | 73 | 65 | 69 | +4 questions |

Relative to pure Version Tree at the same payload, Peak-Guard recovered MATH
but slightly harmed GSM8K. Its first-slice success did not replicate. Therefore
the evidence supports joint allocation, but not scattered layer-local holes.

## Gate 3: adjacent-layer Band-Guard

Band-Guard changes only the pruning structure:

- Version Tree remains the first-stage allocator.
- K/V trees, residuals, attention statistics, rightward routing, recent-20%
  protection, global budget and pruning cap remain unchanged.
- A token root can be removed only from a fixed adjacent pair of deep layers
  when both layers pass the same low-risk gate.
- Every released root slot purchases the highest-valued excluded Version-Tree
  refinements under the same global 40% payload.

The candidate is promoted only if it exceeds the easier-budget FTS reference
on every fixed development domain and the gain is not explained by an
underfilled payload. Report local context coverage, pruned bands/roots,
refinements purchased, K/V version distribution, proxy-statistics time,
decision time and cache-update time. A positive first slice must be confirmed
on the non-overlapping second slice before any formal benchmark.

This record does not claim that structured pruning is better; it defines the
test that can falsify that hypothesis.
