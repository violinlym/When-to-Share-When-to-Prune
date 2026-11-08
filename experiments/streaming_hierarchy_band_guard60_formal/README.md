# Band-Guard full benchmark at 60% compression

This frozen run evaluates the Band-Guard candidate selected on training-only
development data. It combines K/V-independent Complete-Linkage Version Trees
with bounded token-root pruning across agreeing adjacent pairs in deep layers.
The recent 20% is protected from pruning; freed Root slots buy additional
Version-Tree refinements.

The candidate stores 40% of the native Full-KV payload at each current context
length (60% compression). Full KV and the existing FTS control are reused only
after input, model, output, scoring, and content-hash checks. FTS stores 50%
payload (50% compression), so this measures the compression-accuracy frontier;
it is not a matched-budget comparison. Every benchmark remains separately
reported, with paired gains/losses and uncertainty. No parameters are tuned on
these formal benchmark outputs.

Frozen cohorts: GSM8K-400, MATH500-500, MBPP+-378, HumanEval-164. The run uses
Ouro-1.4B, four loops, BF16/eager, greedy decoding and the existing task limits,
stops, and scorers. Candidate numerical KV is audited at every prefill/decode
length. Route/statistics metadata, allocator time, cache-update time, and total
generation time are reported separately.

The GPU job uses four nodes. CPU grading runs as a dependent job after generation
completes. Outputs are stored under the separate frozen run identity; development
outputs and previous formal results are not overwritten.
