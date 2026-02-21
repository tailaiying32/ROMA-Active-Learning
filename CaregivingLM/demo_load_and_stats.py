"""
Demo: Load HDF5 dataset and display statistics.

Shows how to:
- Load SIRS-enhanced joint limit samples from HDF5
- Display metadata and sample structure
- Compute statistics across all samples
"""

import numpy as np
import h5py
from pathlib import Path
from export_to_hdf5 import load_sample_from_hdf5


def load_and_display_stats(hdf5_path='output/sirs_sampling/sirs_samples_100.h5'):
    """
    Load HDF5 dataset and display comprehensive statistics.

    Args:
        hdf5_path: Path to HDF5 file
    """
    hdf5_path = Path(hdf5_path)

    if not hdf5_path.exists():
        print(f"Error: File not found: {hdf5_path}")
        print(f"Please run export_to_hdf5.py first to generate the dataset.")
        return

    print("=" * 70)
    print(f"Loading: {hdf5_path.name}")
    print("=" * 70)

    with h5py.File(hdf5_path, 'r') as f:
        # ====================================================================
        # Global Metadata
        # ====================================================================
        print("\n[Global Metadata]")
        meta = f['metadata']
        n_samples = meta.attrs['n_samples']
        joint_names = [s.decode('utf-8') for s in meta['joint_names'][:]]
        main_pairs = [s.decode('utf-8') for s in meta['main_pairs'][:]]

        print(f"  Total samples: {n_samples}")
        print(f"  Number of joints: {len(joint_names)}")
        print(f"  Joint names: {', '.join(joint_names[:4])}... ({len(joint_names)} total)")
        print(f"  Number of pairs: {len(main_pairs)}")
        print(f"  Pairs:")
        for pair_str in main_pairs:
            j1, j2 = pair_str.split('|')
            print(f"    - {j1} × {j2}")

        # Load configuration if available
        if 'config' in meta:
            import json
            config_json = meta['config'][()]
            if isinstance(config_json, bytes):
                config_json = config_json.decode('utf-8')
            config_dict = json.loads(config_json)
            print(f"\n  Configuration:")
            print(f"    Calibration grid: {config_dict.get('calibration_grid_n', 'N/A')}")
            print(f"    Target feasibility: [{config_dict.get('target_feasibility_min', 'N/A')}, "
                  f"{config_dict.get('target_feasibility_max', 'N/A')}]")

        # ====================================================================
        # Sample Statistics
        # ====================================================================
        print("\n[Sample Statistics]")

        # Collect statistics across all samples
        all_feasibilities = []
        all_n_bumps = []
        box_ranges = {joint: [] for joint in joint_names}

        for i in range(n_samples):
            sample_group = f['samples'][f'sample_{i:03d}']

            # Box limits
            box_group = sample_group['box_limits']
            joints = [s.decode('utf-8') for s in box_group['joint_names'][:]]
            lower = box_group['lower_bounds'][:]
            upper = box_group['upper_bounds'][:]

            for j, (l, u) in zip(joints, zip(lower, upper)):
                box_ranges[j].append(np.degrees(u - l))

            # SIRS metadata
            if 'metadata' in sample_group:
                meta_group = sample_group['metadata']
                target_feas = meta_group['target_feasibility'][:]
                actual_feas = meta_group['actual_feasibility'][:]

                all_feasibilities.extend(actual_feas)

            # Bump counts
            if 'sirs_bumps' in sample_group:
                bumps_group = sample_group['sirs_bumps']
                for pair_key in bumps_group.keys():
                    n_bumps = bumps_group[pair_key].attrs['n_bumps']
                    all_n_bumps.append(n_bumps)

        # Display statistics
        print(f"\n  Box Limit Ranges (degrees):")
        for joint in joint_names[:4]:
            ranges = box_ranges[joint]
            print(f"    {joint}: {np.mean(ranges):.1f}° ± {np.std(ranges):.1f}° "
                  f"[{np.min(ranges):.1f}°, {np.max(ranges):.1f}°]")
        print(f"    ... ({len(joint_names) - 4} more joints)")

        if all_feasibilities:
            print(f"\n  Feasibility Statistics:")
            print(f"    Mean: {np.mean(all_feasibilities):.1%}")
            print(f"    Std:  {np.std(all_feasibilities):.1%}")
            print(f"    Min:  {np.min(all_feasibilities):.1%}")
            print(f"    Max:  {np.max(all_feasibilities):.1%}")

        if all_n_bumps:
            print(f"\n  Bumps per Pair:")
            print(f"    Mean: {np.mean(all_n_bumps):.1f}")
            print(f"    Std:  {np.std(all_n_bumps):.1f}")
            print(f"    Min:  {np.min(all_n_bumps)}")
            print(f"    Max:  {np.max(all_n_bumps)}")

        # ====================================================================
        # Example Sample
        # ====================================================================
        print("\n[Example Sample: sample_000]")
        sample = load_sample_from_hdf5(hdf5_path, sample_id=0)

        print(f"  Sample ID: {sample.get('id', 'unknown')}")
        print(f"  Number of joints: {len(sample['joint_limits'])}")
        print(f"  Number of pairs: {len(sample.get('sirs_bumps', {}))}")

        print(f"\n  Box Limits (first 4 joints, degrees):")
        for i, (joint, (lower, upper)) in enumerate(list(sample['joint_limits'].items())[:4]):
            print(f"    {joint}: [{np.degrees(lower):.1f}°, {np.degrees(upper):.1f}°]")

        if 'sirs_bumps' in sample:
            print(f"\n  SIRS Constraints:")
            for pair, bumps in sample['sirs_bumps'].items():
                j1, j2 = pair
                print(f"    {j1} × {j2}: {len(bumps)} bumps")

                # Show first bump details
                if bumps:
                    bump = bumps[0]
                    print(f"      Bump 0: mu={np.degrees(bump['mu'])}, "
                          f"ls={np.degrees(bump['ls'])}, alpha={bump['alpha']:.1f}")

        if 'sirs_metadata' in sample:
            print(f"\n  Feasibility per Pair:")
            for pair, meta in sample['sirs_metadata'].items():
                j1, j2 = pair
                print(f"    {j1} × {j2}: "
                      f"target={meta['target_feasibility']:.1%}, "
                      f"actual={meta['actual_feasibility']:.1%}")

        # ====================================================================
        # File Info
        # ====================================================================
        print("\n[File Info]")
        file_size_mb = hdf5_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Compression: gzip (level 4)")
        print(f"  Samples per MB: {n_samples / file_size_mb:.1f}")

    print("\n" + "=" * 70)
    print("✓ Stats display complete!")
    print("=" * 70)


if __name__ == '__main__':
    # Default path
    hdf5_path = 'output/sirs_sampling/sirs_samples_100.h5'

    # Allow command line argument
    import sys
    if len(sys.argv) > 1:
        hdf5_path = sys.argv[1]

    load_and_display_stats(hdf5_path)
