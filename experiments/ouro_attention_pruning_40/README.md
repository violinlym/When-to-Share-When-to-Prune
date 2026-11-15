# Ouro attention-pruning adapters (40% compression)

Portable Ouro adapters for **H2O, SnapKV-D, R-KV, and TriAttention**, exported from the completed Ouro comparison. The adapter code uses PyTorch tensors and Transformers' `DynamicCache` API; the runner targets CUDA/NVIDIA, while the source experiment ran on ROCm. The CUDA path has not been validated on an NVIDIA host yet, so first run a short sample and compare model logits/cache shapes before a full evaluation.

## Frozen comparison semantics

- Ouro-1.4B, four recurrent loops, BF16, greedy decoding, eager attention hooks.
- 40% compression means retaining **60% of same-length four-loop Full KV payload**. If the current logical sequence has `N` positions, each layer/head keeps `ceil(0.6*N)` paired K/V positions.
- Prefill computes logits with the uncompressed causal cache, then compacts it. After every completed decode forward, the retained budget is recomputed from the new current sequence length and the cache is physically compacted. This is not a fixed 60% of prompt length.
- `bound` makes one token-selection decision shared by all four loops at each layer/head. `loop` selects independently for each loop; removed positions are masked and never routed to another loop.
- The four methods use the same payload target, model, prompt, and generation settings. Their native scoring/protection mechanisms differ, as detailed below. This is an Ouro adaptation, not a claim of unchanged official benchmark reproduction.

## Method settings

| Method | Importance / redundancy signal | Recent protection in this adapter |
|---|---|---:|
| H2O | Cumulative native attention; split retained positions between heavy hitters and recent tokens | Half the retained capacity |
| SnapKV-D | Mean causal attention from the latest 128 queries, average-pool kernel 5 | `min(128, capacity)` positions |
| R-KV | Latest 8 queries, max-pool kernel 7, importance/redundancy mixture `0.1 / 0.9`; suppress similar keys using cosine threshold 0.5 | `min(8, capacity)` positions |
| TriAttention | Official trigonometric/magnitude score over geometric offsets up to 65,536, with Ouro-specific query calibration | No separate recent window |

The cache budget parameter `ratio=0.3` is the per-pool heavy-hitter fraction used by the original implementation; the corresponding total retained capacity is 60% (`HH + recent`). Do not change it to `0.6`.

## Requirements and model setup

Use a CUDA-enabled PyTorch build, Transformers with `DynamicLayer`, and a local Ouro-1.4B checkpoint containing its `trust_remote_code` model files. The code uses eager attention because it observes attention weights and hooks the model's `eager_attention_forward`; it is not a vLLM integration. Install versions compatible with the local Ouro model implementation rather than assuming the ROCm environment's package versions will work unchanged on CUDA.

Edit `configs/ouro.example.json` or pass `--model-path`. The configured architecture is 24 physical layers × 4 loops, 16 query/KV heads, head dimension 128. The runner checks the loaded model's layer and KV-head counts.

## Input and invocation

Input is UTF-8 JSONL, one prompt per row:

```json
{"id":"example-1","context":"Question: ...\nAnswer:","max_new_tokens":512,"stop":["\nQuestion:"]}
```

Run all four methods with the bound policy:

```bash
python implementations/run_ouro.py \
  --model-path /path/to/Ouro-1.4B \
  --input /path/to/prompts.jsonl \
  --output /path/to/results.jsonl \
  --loop-policy bound
```

Use `--loop-policy loop` for independent per-loop pruning. Use `--methods h2o snapkv_d` to select a subset, and `--limit 2` for a local smoke run. The output is JSONL written after each method/input and includes generated text, stop reason, actual KV payload, context coverage, runtime, and peak CUDA memory. Choose a new output filename for each run; the runner starts a fresh file.

TriAttention requires calibration generated from generic training/calibration text (not evaluation inputs or labels). Provide JSONL rows with a `text` field:

```bash
python implementations/calibrate_ouro.py \
  --model-path /path/to/Ouro-1.4B \
  --texts /path/to/wikitext_train_texts.jsonl \
  --output /path/to/ouro_tri_calibration.pt \
  --limit 32

python implementations/run_ouro.py \
  --model-path /path/to/Ouro-1.4B \
  --input /path/to/prompts.jsonl \
  --output /path/to/results.jsonl \
  --tri-calibration /path/to/ouro_tri_calibration.pt
```

The calibration stores per-layer/loop/head pre-RoPE query moments and the model's RoPE inverse frequencies. Use the same exact model revision for calibration and evaluation.

## Files

- `implementations/engine.py`: Ouro eager-attention observer, prefill/decode, and generation adapter.
- `implementations/cache.py`, `h2o_reference.py`: physical KV compaction, H2O baseline, and shared/loop-wise masks.
- `implementations/policies.py`: SnapKV-D, R-KV, and TriAttention scoring ports.
- `implementations/run_ouro.py`: CUDA JSONL runner.
- `implementations/calibrate_ouro.py`: portable TriAttention calibration utility.
- `configs/ouro.example.json`: editable model and budget configuration.

Upstream method references: [H2O](https://github.com/FMInference/H2O), [SnapKV-D](https://arxiv.org/abs/2512.12008), [R-KV](https://github.com/Zefan-Cai/R-KV), and [TriAttention](https://github.com/WeianMao/triattention). This folder contains Ouro-oriented scoring/cache adapters; it does not redistribute upstream repositories or benchmark prompts.
