"""Ouro-1.4B controlled layer-wise context pruning and teacher-forced KL."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from core import masked_attention, output_kl, select_deleted


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = Path('/home/manager/A_liyanmin/Project/Loop transformer/independent-prototypes/model')
SEED = 20261116


def clone_cache(source):
    """Share immutable prefill tensors; DynamicLayer.update rebinds via torch.cat."""
    clone = DynamicCache()
    clone.layers = [DynamicLayer() for _ in range(96)]
    for dst, src in zip(clone.layers, source.layers):
        dst.keys, dst.values = src.keys, src.values
        dst.is_initialized = src.is_initialized
    return clone


@torch.inference_mode()
def run_sample(model, tokenizer, doc, sample_index, ratio=0.25, layers=range(24), max_tokens=32):
    module = sys.modules[type(model).__module__]
    original = module.eager_attention_forward
    ids = doc['context_ids']
    if ids != tokenizer.encode(doc['context'], add_special_tokens=False):
        raise AssertionError('Manifest IDs differ from the model tokenizer')
    n = len(ids)
    device = next(model.parameters()).device
    peaks = np.zeros((24, n), dtype=np.float32)
    masses = np.zeros((24, n), dtype=np.float64)
    calls = [0]

    def observe(attn_module, query, key, value, mask, *args, **kwargs):
        out, weights = original(attn_module, query, key, value, mask, *args, **kwargs)
        loop, physical = divmod(calls[0], 24)
        calls[0] += 1
        if loop >= 4 or physical != attn_module.layer_idx:
            raise AssertionError('Unexpected prefill loop/layer order')
        # Peak is a veto score. Mean mass is measured separately for auditing.
        peaks[physical] = np.maximum(peaks[physical], weights.amax((0, 1, 2)).float().cpu().numpy())
        masses[physical] += (weights.float().sum((0, 1, 2)) / (weights.shape[1] * weights.shape[2] * 4)).cpu().numpy()
        return out, weights

    native = DynamicCache()
    native.layers = [DynamicLayer() for _ in range(96)]
    try:
        module.eager_attention_forward = observe
        prefill_logits = model(torch.tensor([ids], device=device), past_key_values=native,
                               use_cache=True, exit_at_step=3, use_weighted_exit=False,
                               logits_to_keep=1).logits
    finally:
        module.eager_attention_forward = original
    if calls[0] != 96 or native.get_seq_length() != n:
        raise AssertionError(f'Expected 96 attention calls and {n} cached positions, saw {calls[0]} / {native.get_seq_length()}')
    if any(layer.get_seq_length() != n for layer in native.layers):
        raise AssertionError('Prefill cache has inconsistent loop/layer lengths')

    def forward_token(token, cache):
        return model(torch.tensor([[token]], device=device), past_key_values=cache,
                     use_cache=True, exit_at_step=3, use_weighted_exit=False,
                     logits_to_keep=1).logits[0, -1].detach()

    # The prompt's final logits are deliberately excluded: pruning starts after prefill.
    baseline = clone_cache(native)
    token = int(prefill_logits[0, -1].argmax())
    continuation, full_logits = [], []
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for _ in range(max_tokens):
        continuation.append(token)
        logits = forward_token(token, baseline)
        full_logits.append(logits.float().cpu())
        if token in eos:
            break
        token = int(logits.argmax())
    del baseline
    if not continuation:
        raise AssertionError('No post-pruning prediction positions')

    rows = []
    for physical in layers:
        physical = int(physical)
        deleted = select_deleted(peaks[physical], ratio)
        recent = int(np.ceil(n * 0.20))
        if np.any(deleted >= n - recent):
            raise AssertionError('Protected recent window was pruned')
        cache = clone_cache(native)
        audit, kls = set(), []
        try:
            for token, reference in zip(continuation, full_logits):
                calls = [0]
                module.eager_attention_forward = masked_attention(original, physical, deleted, n, calls, audit)
                logits = forward_token(token, cache)
                if calls[0] != 96:
                    raise AssertionError('Decode did not visit all 96 loop/layer slots')
                kls.append(output_kl(reference.to(logits.device), logits))
        finally:
            module.eager_attention_forward = original
        if len(deleted) and audit != {0, 1, 2, 3}:
            raise AssertionError(f'Pruning missed a loop: {audit}')
        if any(layer.get_seq_length() != n for layer in native.layers):
            raise AssertionError('A decode variant changed the original prefill positions')
        rows.append(dict(sample_index=sample_index, dataset=doc['task'], doc_id=doc['doc_id'],
                         layer=physical + 1, mean_kl=float(np.mean(kls)), valid_tokens=len(kls),
                         prompt_tokens=n, deleted_tokens=len(deleted), actual_pruning_ratio=len(deleted)/n,
                         deleted_attention_mass=float(masses[physical, deleted].sum()/masses[physical].sum()),
                         protected_recent_tokens=recent, selected_tokens_sha256=hashlib.sha256(deleted.tobytes()).hexdigest()))
        del cache
    return rows, continuation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.25)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    if not (0 <= args.ratio <= 0.8) or not (0 <= args.worker < args.workers):
        parser.error('Invalid ratio or worker assignment')
    torch.set_num_threads(2)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                 local_files_only=True, torch_dtype=torch.bfloat16,
                 attn_implementation='eager').cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    docs = json.loads((HERE / 'manifest48.json').read_text())
    indices = [0] if args.smoke else list(range(len(docs)))
    layers = [0, 12, 23] if args.smoke else range(24)
    output = args.output / ('smoke' if args.smoke else 'formal')
    output.mkdir(parents=True, exist_ok=True)
    for i in indices:
        if i % args.workers != args.worker:
            continue
        dest = output / f'{i:03d}.json'
        if dest.exists():
            continue
        start = time.monotonic()
        rows, tokens = run_sample(model, tokenizer, docs[i], i, args.ratio, layers, args.max_tokens)
        record = dict(rows=rows, continuation_ids=tokens, ratio=args.ratio,
                      seed=SEED, model=str(args.model.resolve()), seconds=time.monotonic()-start)
        temp = dest.with_suffix(f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(record, ensure_ascii=False))
        temp.replace(dest)
        print(f'{i:03d} {docs[i]["task"]} {len(tokens)} tokens {record["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
