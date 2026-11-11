"""Storage arithmetic only. No synthesized area, power or cycle claims."""
import argparse
import csv
from pathlib import Path


def estimate(n, layers=24, heads=16, dim=128):
    roots = layers * heads * n
    trees = 2 * roots
    full = 8 * roots * dim * 2
    return dict(context=n, full_bf16_bytes=full,
        retained_payload_bytes=(8 * roots * 2 // 5) * dim * 2,
        tree_descriptor_bytes=4 * trees,
        shared_mass48_bytes=6 * layers * n,
        root_loop_head_mass48_bytes=6 * 4 * roots,
        reduced_root_peak32_bytes=4 * layers * n,
        prediction_bitmap_bytes=(roots + 7) // 8,
        dense_logical_slot_pointer32_bytes=4 * 8 * roots,
        all_layer_current_token_staging_bytes=8 * layers * heads * dim * 2,
        one_descriptor_pass_bytes=4 * trees,
        sixteen_descriptor_pass_bytes=16 * 4 * trees)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/storage_scenarios.csv')
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    rows = [estimate(n) for n in (512, 1024, 2048, 4096, 8192, 16384)]
    with args.output.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
