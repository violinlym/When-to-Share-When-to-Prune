import unittest
from types import SimpleNamespace

import numpy as np
import torch

from core import masked_attention, output_kl, select_deleted
from run_profile import clone_cache
from measure_tau80 import root_masks
from transformers.cache_utils import DynamicCache, DynamicLayer


class ProtocolTests(unittest.TestCase):
    def test_selection_count_and_recent_protection(self):
        peak = np.arange(100, dtype=np.float32)[::-1]
        deleted = select_deleted(peak)
        self.assertEqual(len(deleted), 25)
        self.assertTrue(np.all(deleted < 80))
        self.assertEqual(len(select_deleted(peak, ratio=0)), 0)
        self.assertEqual(len(select_deleted(np.ones(8))), 2)

    def test_mask_all_heads_all_loops_and_other_layers_unchanged(self):
        calls, audit = [0], set()
        def original(module, query, key, value, mask, *args, **kwargs):
            logits = torch.zeros((1, 2, 1, 5))
            if mask is not None:
                logits += mask
            prob = logits.softmax(-1)
            return prob @ value, prob
        hook = masked_attention(original, 3, [0, 2], 4, calls, audit)
        q = torch.zeros((1, 2, 1, 1))
        k = torch.zeros((1, 2, 5, 1))
        v = torch.ones((1, 2, 5, 1))
        for loop in range(4):
            calls[0] = loop * 24
            for physical in range(24):
                _, prob = hook(SimpleNamespace(layer_idx=physical), q, k, v, None)
                if physical == 3:
                    self.assertTrue(torch.all(prob[..., [0, 2]] == 0))
                else:
                    self.assertTrue(torch.all(prob[..., [0, 2]] > 0))
        self.assertEqual(audit, {0, 1, 2, 3})

    def test_cache_clone_keeps_positions_and_is_independent(self):
        base = DynamicCache()
        base.layers = [DynamicLayer() for _ in range(96)]
        for layer in base.layers:
            layer.update(torch.zeros(1, 2, 5, 3), torch.zeros(1, 2, 5, 3))
        copy = clone_cache(base)
        copy.layers[0].update(torch.ones(1, 2, 1, 3), torch.ones(1, 2, 1, 3))
        self.assertEqual(base.layers[0].get_seq_length(), 5)
        self.assertEqual(copy.layers[0].get_seq_length(), 6)
        self.assertEqual(base.layers[95].get_seq_length(), 5)

    def test_identical_logits_zero_kl(self):
        logits = torch.tensor([1.1, -2.0, 0.3])
        self.assertLess(output_kl(logits, logits.clone()), 1e-7)

    def test_attention_mass_threshold_and_recent_protection(self):
        peak = np.arange(10, dtype=float)[::-1]
        fixed, adaptive, _ = root_masks(peak, np.ones(10))
        self.assertEqual(int(fixed.sum()), 7)
        self.assertEqual(int(adaptive.sum()), 8)
        self.assertTrue(fixed[-2:].all() and adaptive[-2:].all())
        mass = np.array([.01]*8 + [.46,.46])
        _, adaptive, _ = root_masks(peak,mass)
        self.assertEqual(int(adaptive.sum()), 2)
        self.assertGreaterEqual(mass[adaptive].sum()/mass.sum(),.8)


if __name__ == '__main__':
    unittest.main()
