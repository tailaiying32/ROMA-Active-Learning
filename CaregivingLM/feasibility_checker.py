"""
SIRS Feasibility Checker

Loads SIRS-enhanced joint limit samples from HDF5 and provides a query
interface to check if a given joint configuration is feasible.
"""

import numpy as np
import h5py
from pathlib import Path

from sirs2d.sirs import h_value
from export_to_hdf5 import load_sample_from_hdf5
import sirs_sampling_config as config


class SIRSFeasibilityChecker:
    """
    Query interface for checking joint configuration feasibility.

    Usage:
        checker = SIRSFeasibilityChecker('sirs_samples_100.h5', sample_id=0)
        result = checker.is_feasible(joint_config)
    """

    def __init__(self, hdf5_path, sample_id=0, use_smooth=True, smoothing_k='auto'):
        """
        Initialize the feasibility checker.

        Args:
            hdf5_path: Path to HDF5 file containing samples
            sample_id: Which sample to use (0-based index)
            use_smooth: Use smooth corners for h-value computation
            smoothing_k: Smoothing parameter ('auto' or numeric value)
        """
        self.hdf5_path = Path(hdf5_path)
        self.sample_id = sample_id
        self.use_smooth = use_smooth
        self.smoothing_k = smoothing_k

        # Load the sample
        self.sample = load_sample_from_hdf5(self.hdf5_path, sample_id)

        # Store joint names in order
        self.joint_names = list(self.sample['joint_limits'].keys())

        # Precompute pairwise constraint info
        self._build_constraint_index()

    def _build_constraint_index(self):
        """Build an index of pairwise constraints for fast lookup."""
        self.pairwise_constraints = {}

        if 'sirs_bumps' not in self.sample:
            return

        for pair_key, bumps in self.sample['sirs_bumps'].items():
            j1, j2 = pair_key

            # Get indices in joint_names
            if j1 in self.joint_names and j2 in self.joint_names:
                i1 = self.joint_names.index(j1)
                i2 = self.joint_names.index(j2)

                # Store box and bumps
                lower1, upper1 = self.sample['joint_limits'][j1]
                lower2, upper2 = self.sample['joint_limits'][j2]

                box = {
                    'q1_range': (lower1, upper1),
                    'q2_range': (lower2, upper2)
                }

                self.pairwise_constraints[(i1, i2)] = {
                    'box': box,
                    'bumps': bumps,
                    'joint_names': (j1, j2)
                }

    def is_feasible(self, joint_config, return_details=False):
        """
        Check if a joint configuration is feasible under SIRS constraints.

        Args:
            joint_config: Dictionary mapping joint names to values (in radians),
                         or array of values in same order as self.joint_names
            return_details: If True, return detailed information about each constraint

        Returns:
            If return_details=False: Boolean, True if feasible
            If return_details=True: Dictionary with:
                - 'is_feasible': Boolean
                - 'pair_results': Dict mapping pair to h-value and feasibility
                - 'limiting_pair': Which pair most violates constraints (if infeasible)
                - 'joint_values': The joint configuration used
        """
        # Convert input to array
        if isinstance(joint_config, dict):
            q = np.array([joint_config[name] for name in self.joint_names])
        else:
            q = np.array(joint_config)

        # Check box limits
        for i, (name, (lower, upper)) in enumerate(self.sample['joint_limits'].items()):
            if q[i] < lower or q[i] > upper:
                if return_details:
                    return {
                        'is_feasible': False,
                        'reason': f'Out of box bounds for {name}',
                        'joint_values': {name: q[i] for i, name in enumerate(self.joint_names)},
                        'pair_results': {},
                        'limiting_pair': None
                    }
                else:
                    return False

        # Check pairwise SIRS constraints
        pair_results = {}
        min_h_value = float('inf')
        limiting_pair = None

        for (i1, i2), constraint in self.pairwise_constraints.items():
            box = constraint['box']
            bumps = constraint['bumps']
            j1, j2 = constraint['joint_names']

            # Extract the 2D point
            q_pair = np.array([q[i1], q[i2]])

            # Compute h-value
            h = h_value(q_pair, box, bumps,
                       use_smooth=self.use_smooth,
                       smoothing_k=self.smoothing_k)

            pair_results[(j1, j2)] = {
                'h_value': float(h),
                'is_feasible': h >= 0,
                'joint_values': {j1: float(q[i1]), j2: float(q[i2])}
            }

            # Track most violated constraint
            if h < min_h_value:
                min_h_value = h
                limiting_pair = (j1, j2)

        # Overall feasibility
        is_feasible = all(result['is_feasible'] for result in pair_results.values())

        if return_details:
            return {
                'is_feasible': is_feasible,
                'pair_results': pair_results,
                'limiting_pair': limiting_pair if not is_feasible else None,
                'joint_values': {name: float(q[i]) for i, name in enumerate(self.joint_names)},
                'min_h_value': float(min_h_value)
            }
        else:
            return is_feasible

    def sample_feasible_config(self, max_attempts=1000):
        """
        Sample a random feasible joint configuration.

        Args:
            max_attempts: Maximum number of rejection sampling attempts

        Returns:
            Dictionary mapping joint names to feasible values, or None if failed
        """
        rng = np.random.default_rng()

        for _ in range(max_attempts):
            # Sample uniformly from box
            q = np.array([
                rng.uniform(lower, upper)
                for lower, upper in self.sample['joint_limits'].values()
            ])

            # Check feasibility
            if self.is_feasible(q):
                return {name: float(q[i]) for i, name in enumerate(self.joint_names)}

        return None

    def get_box_limits(self):
        """Return box limits for all joints."""
        return self.sample['joint_limits'].copy()

    def get_pairwise_info(self):
        """Return information about pairwise constraints."""
        info = {}
        for (i1, i2), constraint in self.pairwise_constraints.items():
            j1, j2 = constraint['joint_names']
            info[(j1, j2)] = {
                'n_bumps': len(constraint['bumps']),
                'box': constraint['box'].copy()
            }

            # Add metadata if available
            if 'sirs_metadata' in self.sample:
                pair_key = (j1, j2)
                if pair_key in self.sample['sirs_metadata']:
                    meta = self.sample['sirs_metadata'][pair_key]
                    info[(j1, j2)]['target_feasibility'] = meta['target_feasibility']
                    info[(j1, j2)]['actual_feasibility'] = meta['actual_feasibility']

        return info

    def __repr__(self):
        n_joints = len(self.joint_names)
        n_pairs = len(self.pairwise_constraints)
        return (f"SIRSFeasibilityChecker(sample_id={self.sample_id}, "
                f"n_joints={n_joints}, n_constraints={n_pairs})")


def demo_feasibility_checker():
    """Demonstrate usage of the feasibility checker."""
    print("=" * 70)
    print("SIRS Feasibility Checker Demo")
    print("=" * 70)

    # Find HDF5 file in output directory
    output_dir = Path(config.OUTPUT_DIR)
    hdf5_files = list(output_dir.glob('sirs_samples_*.h5'))

    if not hdf5_files:
        print(f"\nError: No HDF5 files found in {output_dir}")
        print("Please run export_to_hdf5.py first to generate the file.")
        return

    # Use the most recent file
    hdf5_path = max(hdf5_files, key=lambda p: p.stat().st_mtime)

    # Create checker for sample 0
    print(f"\nLoading sample 0 from {hdf5_path.name}...")
    checker = SIRSFeasibilityChecker(hdf5_path, sample_id=0)
    print(f"✓ Loaded: {checker}")

    # Show box limits
    print("\n[Box Limits]")
    for joint_name, (lower, upper) in list(checker.get_box_limits().items())[:4]:
        print(f"  {joint_name}: [{np.degrees(lower):.1f}°, {np.degrees(upper):.1f}°]")
    print(f"  ... ({len(checker.get_box_limits()) - 4} more joints)")

    # Show pairwise constraints
    print("\n[Pairwise Constraints]")
    for (j1, j2), info in checker.get_pairwise_info().items():
        target_feas = info.get('target_feasibility', 0) * 100
        actual_feas = info.get('actual_feasibility', 0) * 100
        print(f"  {j1} × {j2}:")
        print(f"    Bumps: {info['n_bumps']}, Target: {target_feas:.1f}%, Actual: {actual_feas:.1f}%")

    # Sample a feasible configuration
    print("\n[Sampling Feasible Configuration]")
    feasible_config = checker.sample_feasible_config(max_attempts=10000)

    if feasible_config:
        print("✓ Found feasible configuration:")
        for joint_name, value in list(feasible_config.items())[:4]:
            print(f"  {joint_name}: {np.degrees(value):.1f}°")
        print(f"  ... ({len(feasible_config) - 4} more joints)")

        # Check it (with details)
        print("\n[Checking Feasibility with Details]")
        result = checker.is_feasible(feasible_config, return_details=True)
        print(f"  Overall feasible: {result['is_feasible']}")
        print(f"  Min h-value: {result['min_h_value']:.4f}")

        print("\n  Per-pair results:")
        for pair, pair_result in result['pair_results'].items():
            h = pair_result['h_value']
            status = "✓" if pair_result['is_feasible'] else "✗"
            print(f"    {status} {pair[0]} × {pair[1]}: h = {h:.4f}")

        # Now test with a random infeasible point (outside box)
        print("\n[Testing Infeasible Configuration]")
        infeasible_config = feasible_config.copy()
        first_joint = list(infeasible_config.keys())[0]

        # Set first joint way outside box
        box_limits = checker.get_box_limits()
        lower, upper = box_limits[first_joint]
        infeasible_config[first_joint] = upper + np.radians(50)  # 50° beyond limit

        result = checker.is_feasible(infeasible_config, return_details=True)
        print(f"  Overall feasible: {result['is_feasible']}")
        if 'reason' in result:
            print(f"  Reason: {result['reason']}")
        elif result['limiting_pair']:
            print(f"  Limiting pair: {result['limiting_pair']}")
            print(f"  Min h-value: {result['min_h_value']:.4f}")

    else:
        print("✗ Failed to find feasible configuration after 10000 attempts")

    print("\n" + "=" * 70)
    print("✓ Demo complete!")


if __name__ == '__main__':
    demo_feasibility_checker()
