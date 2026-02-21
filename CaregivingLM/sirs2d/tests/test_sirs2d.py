#!/usr/bin/env python3
"""
Unit tests for SIRS 2D system.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sirs2d import config
from sirs2d.sampler import sample_box, sample_bumps, calibrate_alpha
from sirs2d.sirs import (
    inside_box, box_margin, h_value, feasible_mask_grid,
    compute_feasible_fraction
)


def test_outside_box_infeasible():
    """Test that points outside box are infeasible."""
    print("\nTest 1: Outside box → infeasible")

    # Create simple box
    box = {
        'q1_range': (0.0, 100.0),
        'q2_range': (0.0, 100.0)
    }

    # No bumps
    bumps = []

    # Points outside box
    q_outside = np.array([
        [-10.0, 50.0],   # q1 too low
        [110.0, 50.0],   # q1 too high
        [50.0, -10.0],   # q2 too low
        [50.0, 110.0],   # q2 too high
        [-10.0, -10.0],  # Both too low
    ])

    # Check feasibility
    h_vals = h_value(q_outside, box, bumps)

    # All should be infeasible (h < 0)
    assert np.all(h_vals < 0), "Points outside box should be infeasible"

    print(f"  ✓ All {len(q_outside)} points outside box are infeasible")
    print(f"  h values: {h_vals}")


def test_no_bumps_box_shape():
    """Test that with K=0, feasible region matches box exactly."""
    print("\nTest 2: K=0 → feasible region = box")

    # Create box
    box = {
        'q1_range': (-50.0, 50.0),
        'q2_range': (0.0, 100.0)
    }

    # No bumps
    bumps = []

    # Generate grid
    X, Y, H, M = feasible_mask_grid(box, bumps, grid_n=100)

    # All points inside box should be feasible
    frac = np.mean(M)

    # Should be exactly 1.0 (allowing small numerical error)
    assert abs(frac - 1.0) < 0.01, f"Expected frac=1.0, got {frac:.4f}"

    print(f"  ✓ Feasible fraction with K=0: {frac:.4f}")
    print(f"  Box: q1={box['q1_range']}, q2={box['q2_range']}")


def test_alpha_scaling():
    """Test that increasing alpha reduces feasible fraction."""
    print("\nTest 3: Increasing α → reduced feasibility")

    rng = np.random.default_rng(42)

    # Sample box and bumps
    box = sample_box(rng)
    bumps = sample_bumps(box, rng)

    # Ensure bumps have measurable effect by boosting alpha
    q1_width = box['q1_range'][1] - box['q1_range'][0]
    q2_width = box['q2_range'][1] - box['q2_range'][0]
    min_alpha = 0.3 * min(q1_width, q2_width)

    for bump in bumps:
        bump['alpha'] = max(bump['alpha'], min_alpha)

    # Compute initial feasible fraction
    frac_initial = compute_feasible_fraction(box, bumps, grid_n=200)

    # Scale alpha by 2x
    bumps_scaled = []
    for bump in bumps:
        bumps_scaled.append({
            'mu': bump['mu'].copy(),
            'ls': bump['ls'].copy(),
            'alpha': bump['alpha'] * 2.0
        })

    # Compute scaled feasible fraction
    frac_scaled = compute_feasible_fraction(box, bumps_scaled, grid_n=200)

    print(f"  Initial α: frac = {frac_initial:.2%}")
    print(f"  Scaled α (2x): frac = {frac_scaled:.2%}")

    # Scaled should be less than initial
    assert frac_scaled < frac_initial, "2x alpha should reduce feasible fraction"

    print(f"  ✓ Scaling α by 2x reduced feasibility by {frac_initial - frac_scaled:.2%}")


def test_calibration():
    """Test that calibration reaches target within tolerance."""
    print("\nTest 4: Calibration accuracy")

    rng = np.random.default_rng(43)

    # Sample box and bumps
    box = sample_box(rng)
    bumps = sample_bumps(box, rng)

    # Target fractions to test
    target_fractions = [0.3, 0.5, 0.7]

    for target_frac in target_fractions:
        # Calibrate
        bumps_calibrated = calibrate_alpha(box, bumps, target_frac, rng, grid_n=200)

        # Compute actual fraction
        actual_frac = compute_feasible_fraction(box, bumps_calibrated, grid_n=200)

        error = abs(actual_frac - target_frac)
        tolerance = config.CALIBRATION_TOLERANCE

        print(f"  Target: {target_frac:.2%}, Actual: {actual_frac:.2%}, Error: {error:.2%}")

        assert error <= tolerance, f"Calibration error {error:.2%} exceeds tolerance {tolerance:.2%}"

    print(f"  ✓ All calibrations within ±{tolerance:.0%} tolerance")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SIRS 2D Unit Tests")
    print("=" * 60)

    try:
        test_outside_box_infeasible()
        test_no_bumps_box_shape()
        test_alpha_scaling()
        test_calibration()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return True

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
