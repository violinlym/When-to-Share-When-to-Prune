"""Deterministic synthetic functional tests. No model-accuracy claims."""
import json
import random
import unittest
from collections import Counter
from pathlib import Path
from selector import Tree, select, oracle, rightward_mapping
from scoring import sampled_tree, dyadic_ceiling


class FunctionalTests(unittest.TestCase):
    def test_integer_selector_parity(self):
        aggregate = Counter()
        peak_cached = 0
        cases = 0
        for seed in range(64):
            rng = random.Random(seed)
            trees = []
            for i in range(20):
                shifts = sorted(rng.randrange(8) for _ in range(3))
                if i % 5 == 0:
                    shifts[-1] = None
                mass = rng.randrange(1 << 16)
                low = max(0, mass - rng.randrange(4000))
                high = min((1 << 16) - 1, mass + rng.randrange(4000))
                trees.append(Tree(i, tuple(shifts), rng.randrange(4), low, high, mass))
            for b in range(sum(t.alive for t in trees) + 1):
                got, _, work = select(trees, b, 16)
                self.assertEqual(set(got), set(oracle(trees, b)))
                self.assertEqual(len(got), b)
                for t in trees:
                    js = sorted(j for tid, j in got if tid == t.tree_id)
                    self.assertEqual(js, list(range(len(js))))
                peak_cached = max(peak_cached, work.pop('peak_cached_mass_entries', 0))
                aggregate.update(work)
                cases += 1
        out = Path(__file__).parent / 'results'
        out.mkdir(exist_ok=True)
        (out / 'selector_synthetic.json').write_text(json.dumps({
            'evidence': 'SYNTHETIC_ONLY', 'cases': cases, 'seeds': 64,
            'integer_oracle_parity': True, 'logical_work': dict(aggregate),
            'max_cached_mass_entries_any_case': peak_cached,
            'note': 'Includes all budgets; no FP32 parity, root exchange, RTL, accuracy or latency validation.'
        }, indent=2) + '\n')

    def test_progressive_and_native_routing(self):
        for seed in range(100):
            rng = random.Random(seed)
            v = [[rng.randrange(-128, 128) for _ in range(16)] for _ in range(4)]
            shifts, anchors = sampled_tree(v, list(range(0, 16, 2)))
            self.assertEqual(sorted(anchors), [0, 1, 2])
            tree = Tree(0, shifts, 3, 100, 100, 100)
            previous = set(range(4))
            for budget in (3, 2, 1, 0):
                got, _, _ = select([tree], budget, 16)
                mapping = rightward_mapping(anchors, len(got))
                self.assertTrue(set(mapping) <= previous)
                self.assertTrue(all(target >= r for r, target in enumerate(mapping)))
                previous = set(mapping)
                tree = Tree(0, shifts, budget, 100, 100, 100)

    def test_zero_ties(self):
        trees = [Tree(i, (None,) * 3, 3, 0, 0, 0) for i in range(3)]
        self.assertEqual(select(trees, 4, 8)[0], [(0, 0), (0, 1), (0, 2), (1, 0)])
        self.assertEqual(sampled_tree([[1] * 4] * 4, [0, 1]), ((None,) * 3, (2, 1, 0)))

    def test_reject_bad_bounds_and_overflow(self):
        for tree in (Tree(0, (0, 1, 2), 3, 10, 20, 21),
                     Tree(0, (0, 1, 2), 3, 0, 256, 256),
                     Tree(0, (2, 1, 0), 3, 0, 10, 5)):
            with self.assertRaises(ValueError):
                select([tree], 2, 8)
        self.assertEqual(dyadic_ceiling(0, 0), None)
        self.assertEqual(dyadic_ceiling(3, 10), 1)
        self.assertEqual(dyadic_ceiling(11, 10), 0)


if __name__ == '__main__':
    unittest.main()
