"""
Demo: Dense sampling of feasible joint-space configurations.

Shows how to:
- Load SIRS constraints from HDF5
- Generate dense coverage using rejection sampling
- Export feasible samples to various formats
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add scone directory to path for Halton sampling
sys.path.insert(0, str(Path(__file__).parent / 'scone'))
from batch_joint_limit_sampler import generate_halton_samples

from feasibility_checker import SIRSFeasibilityChecker


def dense_sample_feasible_space(hdf5_path='output/sirs_sampling/sirs_samples_100.h5',
                                sample_id=0,
                                n_target=5000,
                                max_attempts=None,
                                output_format='csv',
                                output_path=None):
    """
    Generate dense coverage of feasible joint-space using Halton sequence rejection sampling.

    Args:
        hdf5_path: Path to HDF5 file with SIRS constraints
        sample_id: Which sample to use
        n_target: Target number of feasible samples
        max_attempts: Maximum rejection attempts (None = 10x target)
        output_format: 'csv', 'npy', or 'both'
        output_path: Output path (None = auto-generate)
    """
    hdf5_path = Path(hdf5_path)

    if not hdf5_path.exists():
        print(f"Error: File not found: {hdf5_path}")
        return

    print("=" * 70)
    print(f"Dense Feasible Joint-Space Sampling")
    print("=" * 70)

    # ====================================================================
    # Load Sample and Create Checker
    # ====================================================================
    print(f"\n[Loading Sample]")
    print(f"  File: {hdf5_path.name}")
    print(f"  Sample ID: {sample_id}")

    checker = SIRSFeasibilityChecker(hdf5_path, sample_id=sample_id)
    print(f"  ✓ Loaded: {checker}")

    box_limits = checker.get_box_limits()
    n_joints = len(box_limits)

    if max_attempts is None:
        max_attempts = n_target * 10  # Default: 10x oversampling

    print(f"\n[Sampling Configuration]")
    print(f"  Target samples: {n_target}")
    print(f"  Sampling method: Halton sequence + rejection sampling")
    print(f"  Number of dimensions: {n_joints}")
    print(f"  Max attempts: {max_attempts:,}")

    # ====================================================================
    # Dense Sampling using Halton Sequence
    # ====================================================================
    print(f"\n[Generating Feasible Samples]")
    print(f"  Generating Halton sequence...")

    # Generate Halton samples in [0,1]^n_joints
    halton_samples = generate_halton_samples(max_attempts, n_joints)

    # Scale to box limits
    box_limits_array = np.array(list(box_limits.values()))
    lower_bounds = box_limits_array[:, 0]
    upper_bounds = box_limits_array[:, 1]

    # Scale Halton points to box ranges
    scaled_samples = lower_bounds + halton_samples * (upper_bounds - lower_bounds)

    print(f"  ✓ Generated {max_attempts:,} Halton candidates")
    print(f"  Applying SIRS feasibility filter...")

    # Rejection sampling with SIRS feasibility
    feasible_samples = []
    total_attempts = 0

    for i, config in enumerate(scaled_samples):
        if len(feasible_samples) >= n_target:
            break

        # Check SIRS feasibility
        if checker.is_feasible(config):
            feasible_samples.append(config)

        total_attempts += 1

        # Progress reporting
        if (i + 1) % 500 == 0 or i == len(scaled_samples) - 1:
            coverage = len(feasible_samples) / n_target * 100
            acceptance = len(feasible_samples) / total_attempts * 100 if total_attempts > 0 else 0
            print(f"    Checked {total_attempts:,}/{max_attempts:,}, "
                  f"Found {len(feasible_samples)}/{n_target} ({coverage:.1f}% coverage, "
                  f"{acceptance:.1f}% acceptance)", end='\r')

    print()  # Newline

    print(f"  ✓ Generated {len(feasible_samples)} feasible samples")
    print(f"  Total attempts: {total_attempts:,}")
    print(f"  Acceptance rate: {len(feasible_samples) / total_attempts * 100:.2f}%")

    if len(feasible_samples) < n_target:
        print(f"  ⚠ Warning: Only generated {len(feasible_samples)}/{n_target} samples "
              f"({len(feasible_samples)/n_target*100:.1f}% of target)")
        print(f"  Try increasing max_attempts (current: {max_attempts:,})")

    # ====================================================================
    # Statistics
    # ====================================================================
    print(f"\n[Coverage Statistics]")

    samples_array = np.array(feasible_samples)

    print(f"  Joint ranges (degrees):")
    for i, (joint_name, (box_min, box_max)) in enumerate(list(box_limits.items())):
        sample_min = np.degrees(samples_array[:, i].min())
        sample_max = np.degrees(samples_array[:, i].max())
        box_min_deg = np.degrees(box_min)
        box_max_deg = np.degrees(box_max)
        coverage = (sample_max - sample_min) / (box_max_deg - box_min_deg) * 100

        print(f"    {joint_name}:")
        print(f"      Box: [{box_min_deg:.1f}°, {box_max_deg:.1f}°]")
        print(f"      Samples: [{sample_min:.1f}°, {sample_max:.1f}°]")
        print(f"      Coverage: {coverage:.1f}%")
    print(f"    ... ({n_joints - 4} more joints)")

    # ====================================================================
    # Export
    # ====================================================================
    if output_path is None:
        output_dir = Path('output/dense_samples')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_base = output_dir / f'feasible_samples_s{sample_id}_n{len(feasible_samples)}'
    else:
        output_base = Path(output_path).with_suffix('')

    print(f"\n[Exporting Samples]")

    if output_format in ['csv', 'both']:
        # Export to CSV with joint names as columns
        samples_deg = np.degrees(samples_array)
        df = pd.DataFrame(samples_deg, columns=list(box_limits.keys()))

        csv_path = output_base.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"  ✓ CSV: {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")

    if output_format in ['npy', 'both']:
        # Export to NPY (raw radians)
        npy_path = output_base.with_suffix('.npy')
        np.save(npy_path, samples_array)
        print(f"  ✓ NPY: {npy_path} ({npy_path.stat().st_size / 1024:.1f} KB)")

    print("\n" + "=" * 70)
    print("✓ Dense sampling complete!")
    print("=" * 70)

    return samples_array


if __name__ == '__main__':
    import sys

    # Parse command line arguments
    hdf5_path = 'path'
    sample_id = 0
    n_target = 50000

    if len(sys.argv) > 1:
        hdf5_path = sys.argv[1]
    if len(sys.argv) > 2:
        sample_id = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_target = int(sys.argv[3])

    dense_sample_feasible_space(
        hdf5_path=hdf5_path,
        sample_id=sample_id,
        n_target=n_target,
        output_format='both'
    )
