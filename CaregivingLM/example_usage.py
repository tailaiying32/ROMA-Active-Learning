"""
Example usage of SIRS-enhanced joint limit sampling.

Demonstrates end-to-end workflow from generation to querying.
"""

import numpy as np
from pathlib import Path

# Configuration
import sirs_sampling_config as config

# Generation
from sirs_batch_sampler import generate_sirs_enhanced_joint_limits

# Export/Load
from export_to_hdf5 import export_samples_to_hdf5, load_sample_from_hdf5

# Query
from feasibility_checker import SIRSFeasibilityChecker

# Visualization
from visualize_sirs_samples import visualize_single_sample_pairwise


def example_1_generate_samples():
    """Example 1: Generate SIRS-enhanced samples."""
    print("\n" + "=" * 70)
    print("Example 1: Generate 10 SIRS-Enhanced Samples")
    print("=" * 70)

    samples = generate_sirs_enhanced_joint_limits(
        n_samples_per_batch=10,  # 10 samples
        reject_disconnected=True,
        max_rejection_attempts=10,
        verbose=True
    )

    print(f"\n✓ Generated {len(samples)} samples")

    # Inspect first sample
    sample = samples[0]
    print(f"\nSample 0 structure:")
    print(f"  ID: {sample['id']}")
    print(f"  Joints: {len(sample['joint_limits'])}")
    print(f"  SIRS pairs: {len(sample['sirs_bumps'])}")

    # Show one pair's metadata
    pair = list(sample['sirs_metadata'].keys())[0]
    meta = sample['sirs_metadata'][pair]
    print(f"\n  Pair {pair[0]} × {pair[1]}:")
    print(f"    Bumps: {meta['n_bumps']}")
    print(f"    Target feasibility: {meta['target_feasibility']:.1%}")
    print(f"    Actual feasibility: {meta['actual_feasibility']:.1%}")

    return samples


def example_2_export_to_hdf5(samples):
    """Example 2: Export samples to HDF5."""
    print("\n" + "=" * 70)
    print("Example 2: Export to HDF5")
    print("=" * 70)

    output_path = Path('example_samples.h5')

    export_samples_to_hdf5(
        samples,
        output_path,
        compression='gzip',
        compression_opts=4
    )

    print(f"\n✓ Exported to {output_path}")
    return output_path


def example_3_load_and_query(hdf5_path):
    """Example 3: Load sample and query feasibility."""
    print("\n" + "=" * 70)
    print("Example 3: Load and Query Feasibility")
    print("=" * 70)

    # Load using low-level function
    print("\n[Method 1: Direct load]")
    sample = load_sample_from_hdf5(hdf5_path, sample_id=0)
    print(f"✓ Loaded sample 0: {len(sample['joint_limits'])} joints")

    # Load using feasibility checker
    print("\n[Method 2: Feasibility checker]")
    checker = SIRSFeasibilityChecker(hdf5_path, sample_id=0)
    print(f"✓ Loaded: {checker}")

    # Sample a feasible configuration
    print("\n[Sampling feasible configuration]")
    config = checker.sample_feasible_config(max_attempts=10000)

    if config:
        print("✓ Found feasible configuration:")
        for joint, value in list(config.items())[:4]:
            print(f"  {joint}: {np.degrees(value):.1f}°")
        print(f"  ... ({len(config) - 4} more)")

        # Query with details
        result = checker.is_feasible(config, return_details=True)
        print(f"\n✓ Feasibility check:")
        print(f"  Overall: {result['is_feasible']}")
        print(f"  Min h-value: {result['min_h_value']:.4f}")

        # Show per-pair h-values
        print("\n  Per-pair h-values:")
        for pair, pair_result in result['pair_results'].items():
            h = pair_result['h_value']
            status = "✓" if pair_result['is_feasible'] else "✗"
            print(f"    {status} {pair[0]} × {pair[1]}: {h:.4f}")

        return config

    else:
        print("✗ Failed to find feasible configuration")
        return None


def example_4_visualize(hdf5_path):
    """Example 4: Visualize a sample."""
    print("\n" + "=" * 70)
    print("Example 4: Visualize Sample")
    print("=" * 70)

    # Load sample
    sample = load_sample_from_hdf5(hdf5_path, sample_id=0)

    # Visualize
    output_path = Path('example_sample_visualization.png')
    print(f"\nGenerating visualization...")

    visualize_single_sample_pairwise(
        sample,
        output_path=output_path,
        grid_n=200
    )

    print(f"✓ Saved to {output_path}")


def example_5_batch_query(hdf5_path, joint_configs):
    """Example 5: Batch feasibility checking."""
    print("\n" + "=" * 70)
    print("Example 5: Batch Feasibility Checking")
    print("=" * 70)

    checker = SIRSFeasibilityChecker(hdf5_path, sample_id=0)

    # Check multiple configurations
    feasible_count = 0

    print(f"\nChecking {len(joint_configs)} configurations...")

    for i, config in enumerate(joint_configs):
        is_feasible = checker.is_feasible(config)

        if is_feasible:
            feasible_count += 1

        if i < 3:  # Show first 3
            status = "✓" if is_feasible else "✗"
            print(f"  {status} Config {i}: {is_feasible}")

    feasibility_rate = feasible_count / len(joint_configs)
    print(f"\n✓ Feasibility rate: {feasibility_rate:.1%} ({feasible_count}/{len(joint_configs)})")


def example_6_compare_samples(hdf5_path):
    """Example 6: Compare feasibility across different samples."""
    print("\n" + "=" * 70)
    print("Example 6: Compare Feasibility Across Samples")
    print("=" * 70)

    # Create checkers for 3 different samples
    checkers = [
        SIRSFeasibilityChecker(hdf5_path, sample_id=i)
        for i in [0, 5, 9]
    ]

    # Sample a configuration from sample 0
    config = checkers[0].sample_feasible_config()

    if config:
        print("\nConfiguration from Sample 0:")
        for joint, value in list(config.items())[:3]:
            print(f"  {joint}: {np.degrees(value):.1f}°")

        print("\nFeasibility in different samples:")
        for i, checker in enumerate(checkers):
            result = checker.is_feasible(config, return_details=True)
            sample_id = [0, 5, 9][i]

            status = "✓" if result['is_feasible'] else "✗"
            print(f"  {status} Sample {sample_id}: {result['is_feasible']} (min h={result['min_h_value']:.4f})")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SIRS-Enhanced Joint Limit Sampling - Example Usage")
    print("=" * 70)

    # Example 1: Generate samples
    samples = example_1_generate_samples()

    # Example 2: Export to HDF5
    hdf5_path = example_2_export_to_hdf5(samples)

    # Example 3: Load and query
    feasible_config = example_3_load_and_query(hdf5_path)

    # Example 4: Visualize
    example_4_visualize(hdf5_path)

    # Example 5: Batch query
    if feasible_config:
        # Generate some random configs near the feasible one
        joint_configs = []
        for _ in range(10):
            config_dict = {}
            for joint, value in feasible_config.items():
                # Add small noise
                noise = np.random.uniform(-0.1, 0.1)  # ±5.7°
                config_dict[joint] = value + noise
            joint_configs.append(config_dict)

        example_5_batch_query(hdf5_path, joint_configs)

    # Example 6: Compare samples
    example_6_compare_samples(hdf5_path)

    print("\n" + "=" * 70)
    print("✓ All examples complete!")
    print("=" * 70)

    print("\nGenerated files:")
    print("  - example_samples.h5")
    print("  - example_sample_visualization.png")

    print("\nNext steps:")
    print("  - Integrate with SCONE for forward kinematics")
    print("  - Use feasibility checker in motion planning")
    print("  - Generate more samples with custom parameters")


if __name__ == '__main__':
    main()
