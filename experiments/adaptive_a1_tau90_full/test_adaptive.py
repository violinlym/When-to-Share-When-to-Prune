"""CPU protocol checks for only-A1 replacement and immutable A2 sources."""

import hashlib
import math
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from adaptive_cache import AdaptiveCache
from sparse_cache import SparseCache as FixedCache, F


HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'anchor_route_kvonly'/'anchor75'


def synthetic(length=31):
    native=DynamicCache()
    native.layers=[DynamicLayer() for _ in range(96)]
    stats={}
    for loop in range(4):
        for layer in range(24):
            native.update(torch.randn(1,16,length,128,dtype=torch.bfloat16),
                          torch.randn(1,16,length,128,dtype=torch.bfloat16),loop*24+layer)
            mass=torch.arange(1,length+1,dtype=torch.float32)
            stats[loop,layer]={'peak':torch.arange(length,0,-1,dtype=torch.float32),
                               'mass':mass[None].expand(16,-1).clone(),
                               'coverage_mass':mass/(mass.sum()*4)}
    return native,stats


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20261119)

    def test_a2_files_identical_to_baseline_commit(self):
        for name in ('base_cache.py','sparse_cache.py','route_gain.py'):
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).digest(),
                             hashlib.sha256((SOURCE/name).read_bytes()).digest())

    def test_prefill_roots_match_threshold_and_budget(self):
        native,stats=synthetic()
        fixed=FixedCache(native,None,{},dict(stats),{},'anchor75')
        adaptive=AdaptiveCache(native,None,{},dict(stats),{},'adaptive90')
        n=native.get_seq_length()
        self.assertEqual(sum(int(((m[0]&8)!=0).sum()) for m in fixed.mask),24*math.floor(.75*n))
        self.assertGreaterEqual(min(adaptive.coverage_initial),.9-1e-6)
        for layer in range(24):
            roots=((adaptive.mask[layer][0]&8)!=0).cpu().numpy()
            self.assertTrue(roots[-math.ceil(.2*n):].all())
            peak=np.arange(n,0,-1)
            mass=np.arange(1,n+1,dtype=float)
            chosen=np.zeros(n,dtype=bool)
            chosen[-math.ceil(.2*n):]=True
            for token in np.arange(n-math.ceil(.2*n)):
                if mass[chosen].sum()/mass.sum()>=.9:break
                chosen[token]=True
            self.assertTrue(np.array_equal(roots,chosen))
        self.assertLessEqual(adaptive.nbytes(),F*n//4)
        self.assertEqual(adaptive.root_resurrections,0)

    def test_decode_history_weighting_and_no_resurrection(self):
        native,stats=synthetic()
        cache=AdaptiveCache(native,None,{},dict(stats),{},'adaptive90')
        old=[m.clone() for m in cache.mask]
        prior=[x.clone() for x in cache._coverage_cumulative]
        for layer in range(24):
            for loop in range(4):
                cache.current[loop,layer]=torch.randn(2,16,1,128,dtype=torch.bfloat16)
                cache.observed[loop,layer]=torch.full((16,cache.n+1),1/(cache.n+1))
        cache.finish()
        for layer in range(24):
            self.assertEqual(len(cache._coverage_cumulative[layer]),cache.n)
            self.assertTrue(torch.allclose(cache._coverage_cumulative[layer][:-1],
                                           prior[layer]+1/cache.n))
            self.assertTrue(bool(((cache.mask[layer][:,:-1]&old[layer])==cache.mask[layer][:,:-1]).all()))
        self.assertLessEqual(cache.nbytes(),F*cache.n//4)
        self.assertEqual(cache.root_resurrections,0)


if __name__=='__main__':unittest.main()
