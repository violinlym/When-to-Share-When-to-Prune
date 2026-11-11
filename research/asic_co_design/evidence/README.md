# Historical evidence, not new hardware measurements

`experiment8_*` files are small copies of the prior Experiment 8 **synthetic CPU**
results. Their original paths and SHA256 values are in `SOURCE_PROVENANCE.json`.
The copied CSV uses LF line endings; its separate copy hash and this normalization
are recorded in the manifest. Numerical contents are unchanged.
They are not outputs of the new integer selector and do not demonstrate model
accuracy, real-model refresh ratios, cycle latency, area or energy.

`experiment8_synthetic_totals.json` sums the copied refresh CSV. It documents why
25.39% fresh scoring does not imply a 75% reduction in total controller work.

The V-Rex PDF and full dense profiling data are referenced by hash, not copied.
The reproducible real profiling aggregation is in `../results/`, alongside
separately labeled new synthetic selector tests and formula-only storage sizes.
