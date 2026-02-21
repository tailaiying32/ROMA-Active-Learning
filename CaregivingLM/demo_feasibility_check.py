"""
Demo: Feasibility checking with SIRS constraints.

Shows how to:
- Load SIRS constraints from HDF5
- Check if joint configurations are feasible
- Identify which constraints are violated
"""

import numpy as np
from pathlib import Path
from feasibility_checker import SIRSFeasibilityChecker


def demo_feasibility_check(hdf5_path='output/sirs_sampling/sirs_samples_100.h5',
                           sample_id=0, n_test_points=1000):
    """
    Demonstrate feasibility checking with random joint configurations.

    Args:
        hdf5_path: Path to HDF5 file
        sample_id: Which sample to use
        n_test_points: Number of random points to test
    """
    hdf5_path = Path(hdf5_path)

    if not hdf5_path.exists():
        print(f"Error: File not found: {hdf5_path}")
        return

    print("=" * 70)
    print(f"SIRS Feasibility Checking Demo")
    print("=" * 70)

    # ====================================================================
    # Load Sample and Create Checker
    # ====================================================================
    print(f"\n[Loading Sample]")
    print(f"  File: {hdf5_path.name}")
    print(f"  Sample ID: {sample_id}")

    checker = SIRSFeasibilityChecker(hdf5_path, sample_id=sample_id)
    print(f"  ✓ Loaded: {checker}")

    # Display box limits
    print(f"\n[Box Limits]")
    box_limits = checker.get_box_limits()
    for joint_name, (lower, upper) in list(box_limits.items())[:4]:
        print(f"  {joint_name}: [{np.degrees(lower):.1f}°, {np.degrees(upper):.1f}°]")
    print(f"  ... ({len(box_limits) - 4} more joints)")

    # Display pairwise constraints
    print(f"\n[Pairwise Constraints]")
    pairwise_info = checker.get_pairwise_info()
    for (j1, j2), info in pairwise_info.items():
        target = info.get('target_feasibility', 0) * 100
        actual = info.get('actual_feasibility', 0) * 100
        print(f"  {j1} × {j2}:")
        print(f"    Bumps: {info['n_bumps']}, "
              f"Target: {target:.1f}%, Actual: {actual:.1f}%")

    # ====================================================================
    # Test Random Points
    # ====================================================================
    print(f"\n[Testing Random Points]")
    print(f"  Generating {n_test_points} random configurations...")

    rng = np.random.default_rng(42)
    feasible_count = 0
    infeasible_count = 0
    violation_counts = {pair: 0 for pair in pairwise_info.keys()}

    # Sample uniformly from box
    n_joints = len(checker.joint_names)
    test_configs = []

    for _ in range(n_test_points):
        config = np.array([
            rng.uniform(lower, upper)
            for lower, upper in box_limits.values()
        ])
        test_configs.append(config)

    # Check feasibility
    print(f"  Checking feasibility...")
    for i, config in enumerate(test_configs):
        result = checker.is_feasible(config, return_details=True)

        if result['is_feasible']:
            feasible_count += 1
        else:
            infeasible_count += 1

            # Track which pairs are violated
            for pair, pair_result in result['pair_results'].items():
                if not pair_result['is_feasible']:
                    violation_counts[pair] += 1

        if (i + 1) % 100 == 0:
            print(f"    Checked {i+1}/{n_test_points}...", end='\r')

    print(f"\n  ✓ Checked {n_test_points} configurations")

    # ====================================================================
    # Results
    # ====================================================================
    print(f"\n[Results]")
    print(f"  Feasible:   {feasible_count} ({feasible_count/n_test_points*100:.1f}%)")
    print(f"  Infeasible: {infeasible_count} ({infeasible_count/n_test_points*100:.1f}%)")

    if infeasible_count > 0:
        print(f"\n  Constraint Violations:")
        for pair, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
            j1, j2 = pair
            print(f"    {j1} × {j2}: {count} violations "
                  f"({count/infeasible_count*100:.1f}% of infeasible)")

    # ====================================================================
    # Detailed Example
    # ====================================================================
    print(f"\n[Detailed Example: First Infeasible Point]")

    # Find first infeasible point
    for config in test_configs:
        result = checker.is_feasible(config, return_details=True)
        if not result['is_feasible']:
            print(f"  Joint configuration:")
            for i, (name, value) in enumerate(result['joint_values'].items()):
                if i < 4:
                    print(f"    {name}: {np.degrees(value):.1f}°")
            print(f"    ... ({len(result['joint_values']) - 4} more joints)")

            print(f"\n  Constraint violations:")
            for pair, pair_result in result['pair_results'].items():
                j1, j2 = pair
                h = pair_result['h_value']
                status = "✓" if pair_result['is_feasible'] else "✗"
                print(f"    {status} {j1} × {j2}: h = {h:.4f}")

            print(f"\n  Most violated constraint: {result['limiting_pair']}")
            print(f"  Minimum h-value: {result['min_h_value']:.4f}")
            break

    # ====================================================================
    # Sample a Feasible Point
    # ====================================================================
    print(f"\n[Sampling Feasible Configuration]")
    print(f"  Using rejection sampling (max 10000 attempts)...")

    feasible_config = checker.sample_feasible_config(max_attempts=10000)

    if feasible_config:
        print(f"  ✓ Found feasible configuration:")
        for i, (name, value) in enumerate(list(feasible_config.items())[:4]):
            print(f"    {name}: {np.degrees(value):.1f}°")
        print(f"    ... ({len(feasible_config) - 4} more joints)")

        # Verify it's feasible
        result = checker.is_feasible(feasible_config, return_details=True)
        print(f"\n  Verification:")
        print(f"    Is feasible: {result['is_feasible']}")
        print(f"    Min h-value: {result['min_h_value']:.4f}")
    else:
        print(f"  ✗ Failed to find feasible configuration after 10000 attempts")

    print("\n" + "=" * 70)
    print("✓ Feasibility check demo complete!")
    print("=" * 70)


if __name__ == '__main__':
    import sys

    # Parse command line arguments
    hdf5_path = 'output/sirs_sampling/sirs_samples_100.h5'
    sample_id = 0
    n_test_points = 1000

    if len(sys.argv) > 1:
        hdf5_path = sys.argv[1]
    if len(sys.argv) > 2:
        sample_id = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_test_points = int(sys.argv[3])

    demo_feasibility_check(hdf5_path, sample_id, n_test_points)
