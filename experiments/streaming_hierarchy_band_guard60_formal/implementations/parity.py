"""Zero-pruning invariant against frozen capped-version full responses."""
import json
from pathlib import Path
import numpy as np
from common import ROOT, sha


def check_parity(raw, index):
    with np.load(raw['decisions_path'], allow_pickle=False) as actual:
        no_pruning = bool((actual['final'][..., 3] < 4).all())
        if not no_pruning:
            return dict(zero_pruning_case=False, zero_pruning_consistency=None)
        refs = json.loads((ROOT/'configs/zero_pruning_references.json').read_text())
        ref = refs[str(index)]
        assert sha(ref['raw_path']) == ref['raw_sha256']
        expected = json.loads(Path(ref['raw_path']).read_text())
        assert sha(expected['decisions_path']) == expected['decisions_sha256']
        with np.load(expected['decisions_path'], allow_pickle=False) as before:
            consistent = (raw['input_ids'] == expected['input_ids'] and
                          raw['generated_ids'] == expected['generated_ids'] and
                          raw['generated_text'] == expected['generated_text'] and
                          np.array_equal(actual['initial'], before['initial']) and
                          np.array_equal(actual['final'], before['final']))
        return dict(zero_pruning_case=True, zero_pruning_consistency=consistent)
