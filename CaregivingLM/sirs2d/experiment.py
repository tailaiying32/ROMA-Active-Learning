#!/usr/bin/env python3
"""
Main experiment script for SIRS 2D demonstrations.

Generates visualizations of:
1. Single-user demo
2. Calibration before/after
3. Diversity montage
4. Edge coupling example
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from . import config
from .sampler import sample_box, sample_bumps, calibrate_alpha
from .sirs import feasible_mask_grid, compute_auto_smoothing_k, check_2d_connectivity
from .visualize import compose_panel, save_figure


def demo_single_user(output_dir='sirs2d/outputs', seed=None):
    """
    Generate single-user demonstration.

    Creates one random box with bumps and visualizes the feasible region.
    """
    print("\n=== Single-User Demo ===")

    if seed is None:
        seed = config.DEFAULT_RANDOM_SEED

    rng = np.random.default_rng(seed)

    # Sample box and bumps
    box = sample_box(rng)
    bumps = sample_bumps(box, rng, edge_bias=True)

    print(f"Box: q1=[{box['q1_range'][0]:.1f}, {box['q1_range'][1]:.1f}], "
          f"q2=[{box['q2_range'][0]:.1f}, {box['q2_range'][1]:.1f}]")
    print(f"K = {len(bumps)} bumps")

    # Evaluate on grid
    grid_n = config.GRID_RESOLUTION_MAX
    X, Y, H, M = feasible_mask_grid(box, bumps, grid_n, use_smooth=True)

    frac = np.mean(M)
    print(f"Empirical feasible fraction: {frac:.2%}")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    compose_panel(box, bumps, X, Y, H, M, ax=ax)

    # Save
    output_path = Path(output_dir) / 'single_user.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(output_path)
    plt.close()


def demo_calibration(output_dir='sirs2d/outputs', seed=None, target_frac=0.5):
    """
    Generate calibration demonstration (before/after).

    Shows effect of alpha scaling to reach target feasible fraction.
    """
    print("\n=== Calibration Demo ===")

    if seed is None:
        seed = config.DEFAULT_RANDOM_SEED + 1

    rng = np.random.default_rng(seed)

    # Sample box and bumps
    box = sample_box(rng)
    bumps = sample_bumps(box, rng, edge_bias=True)

    print(f"Target feasible fraction: {target_frac:.2%}")

    # Before calibration
    grid_n = config.GRID_RESOLUTION_MAX
    X, Y, H_before, M_before = feasible_mask_grid(box, bumps, grid_n, use_smooth=True)
    frac_before = np.mean(M_before)
    print(f"Before calibration: {frac_before:.2%}")

    # Calibrate
    bumps_calibrated = calibrate_alpha(box, bumps, target_frac, rng, grid_n)

    # After calibration
    X, Y, H_after, M_after = feasible_mask_grid(box, bumps_calibrated, grid_n, use_smooth=True)
    frac_after = np.mean(M_after)
    print(f"After calibration: {frac_after:.2%}")

    # Visualize side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    compose_panel(
        box, bumps, X, Y, H_before, M_before,
        title=f'Before: K={len(bumps)}, frac={frac_before:.2%}',
        ax=ax1
    )

    compose_panel(
        box, bumps_calibrated, X, Y, H_after, M_after,
        title=f'After: K={len(bumps_calibrated)}, frac={frac_after:.2%} (target={target_frac:.2%})',
        ax=ax2
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'calibration_before_after.png'
    save_figure(output_path)
    plt.close()


def demo_diversity_montage(output_dir='sirs2d/outputs', seed=None, n_samples=9):
    """
    Generate diversity montage showing multiple random configurations.

    Args:
        output_dir: Output directory
        seed: Random seed
        n_samples: Number of configurations to show (must be perfect square)
    """
    print("\n=== Diversity Montage Demo ===")

    if seed is None:
        seed = config.DEFAULT_RANDOM_SEED + 2

    rng = np.random.default_rng(seed)

    # Determine grid layout
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]

    print(f"Generating {n_samples} random configurations...")

    for i in range(n_samples):
        ax = axes[i]

        # Sample box and bumps
        box = sample_box(rng)
        bumps = sample_bumps(box, rng, edge_bias=True)

        # Evaluate on grid (lower resolution for speed)
        grid_n = config.GRID_RESOLUTION_MIN
        X, Y, H, M = feasible_mask_grid(box, bumps, grid_n, use_smooth=True)

        # Visualize (minimal elements for montage)
        compose_panel(
            box, bumps, X, Y, H, M,
            title=f'Config {i+1}: K={len(bumps)}',
            ax=ax,
            show_bumps=True,
            show_box=True
        )

        print(f"  Config {i+1}: K={len(bumps)}, frac={np.mean(M):.2%}")

    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'montage.png'
    save_figure(output_path, dpi=100)  # Lower DPI for montage
    plt.close()


def demo_edge_coupling(output_dir='sirs2d/outputs', seed=None):
    """
    Generate edge coupling demonstration.

    Manually places bumps near corners and edges to show realistic
    joint-limit coupling behavior.
    """
    print("\n=== Edge Coupling Demo ===")

    if seed is None:
        seed = config.DEFAULT_RANDOM_SEED + 3

    rng = np.random.default_rng(seed)

    # Sample box
    box = sample_box(rng)
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    q1_width = q1_max - q1_min
    q2_width = q2_max - q2_min

    # Manually create bumps near edges/corners

    bumps = []

    # Bump 1: Near upper-right corner
    bumps.append({
        'mu': np.array([q1_max - 0.15 * q1_width, q2_max - 0.15 * q2_width]),
        'ls': np.array([0.2 * q1_width, 0.2 * q2_width]),
        'alpha': 0.3 * min(q1_width, q2_width)  # Moderate strength
    })

    # Bump 2: Elongated bump along left edge (simulates joint coupling)
    bumps.append({
        'mu': np.array([q1_min + 0.2 * q1_width, q2_min + 0.5 * q2_width]),
        'ls': np.array([0.15 * q1_width, 0.4 * q2_width]),  # Elongated in q2
        'alpha': 0.25 * min(q1_width, q2_width)
    })

    # Bump 3: Near bottom edge
    bumps.append({
        'mu': np.array([q1_min + 0.6 * q1_width, q2_min + 0.15 * q2_width]),
        'ls': np.array([0.25 * q1_width, 0.15 * q2_width]),
        'alpha': 0.2 * min(q1_width, q2_width)
    })

    print(f"Box: q1=[{q1_min:.1f}, {q1_max:.1f}], q2=[{q2_min:.1f}, {q2_max:.1f}]")
    print(f"K = {len(bumps)} manually placed bumps")

    # Evaluate on grid
    grid_n = config.GRID_RESOLUTION_MAX
    X, Y, H, M = feasible_mask_grid(box, bumps, grid_n, use_smooth=True)

    frac = np.mean(M)
    print(f"Feasible fraction: {frac:.2%}")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    compose_panel(
        box, bumps, X, Y, H, M,
        title=f'Edge Coupling Example: K={len(bumps)}, frac={frac:.2%}',
        ax=ax
    )

    # Save
    output_path = Path(output_dir) / 'edge_coupling.png'
    save_figure(output_path)
    plt.close()


def demo_edge_bias(output_dir='sirs2d/outputs', seed=None):
    """
    Demonstrate edge-biased sampling vs uniform sampling.

    Shows side-by-side comparison of bump placement with and without edge bias.
    """
    print("\n=== Edge Bias Demo ===")

    if seed is None:
        seed = config.DEFAULT_RANDOM_SEED + 4

    rng = np.random.default_rng(seed)

    # Sample same box twice
    box = sample_box(rng)

    # Bumps without edge bias (uniform)
    rng_uniform = np.random.default_rng(seed + 100)
    bumps_uniform = sample_bumps(box, rng_uniform, edge_bias=False)

    # Bumps with edge bias
    rng_biased = np.random.default_rng(seed + 100)  # Same seed for comparison
    bumps_biased = sample_bumps(box, rng_biased, edge_bias=True)

    print(f"Box: q1=[{box['q1_range'][0]:.1f}, {box['q1_range'][1]:.1f}], "
          f"q2=[{box['q2_range'][0]:.1f}, {box['q2_range'][1]:.1f}]")
    print(f"K = {len(bumps_uniform)} bumps (same for both)")

    # Evaluate both
    grid_n = config.GRID_RESOLUTION_MAX
    X_u, Y_u, H_u, M_u = feasible_mask_grid(box, bumps_uniform, grid_n, use_smooth=True)
    X_b, Y_b, H_b, M_b = feasible_mask_grid(box, bumps_biased, grid_n, use_smooth=True)

    frac_uniform = np.mean(M_u)
    frac_biased = np.mean(M_b)
    print(f"Uniform sampling: frac={frac_uniform:.2%}")
    print(f"Edge-biased sampling: frac={frac_biased:.2%}")

    # Visualize side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    compose_panel(
        box, bumps_uniform, X_u, Y_u, H_u, M_u,
        title=f'Uniform Sampling: K={len(bumps_uniform)}, frac={frac_uniform:.2%}',
        ax=ax1
    )

    compose_panel(
        box, bumps_biased, X_b, Y_b, H_b, M_b,
        title=f'Edge-Biased Sampling: K={len(bumps_biased)}, frac={frac_biased:.2%}',
        ax=ax2
    )

    # Add text annotation explaining the difference
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    # Compute normalized positions for uniform
    positions_uniform = []
    for bump in bumps_uniform:
        norm_x = (bump['mu'][0] - q1_min) / (q1_max - q1_min)
        norm_y = (bump['mu'][1] - q2_min) / (q2_max - q2_min)
        positions_uniform.append((norm_x, norm_y))

    # Compute normalized positions for biased
    positions_biased = []
    for bump in bumps_biased:
        norm_x = (bump['mu'][0] - q1_min) / (q1_max - q1_min)
        norm_y = (bump['mu'][1] - q2_min) / (q2_max - q2_min)
        positions_biased.append((norm_x, norm_y))

    # Add subtitle with position info
    avg_dist_uniform = np.mean([min(p[0], 1-p[0], p[1], 1-p[1]) for p in positions_uniform])
    avg_dist_biased = np.mean([min(p[0], 1-p[0], p[1], 1-p[1]) for p in positions_biased])

    fig.suptitle(f'Edge Bias Comparison (Beta(α={config.EDGE_BIAS_BETA_ALPHA}, α={config.EDGE_BIAS_BETA_ALPHA}))\n'
                 f'Uniform: avg_edge_dist={avg_dist_uniform:.2f} | Biased: avg_edge_dist={avg_dist_biased:.2f}',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'edge_bias_comparison.png'
    save_figure(output_path)
    plt.close()


def demo_smooth_corners(output_dir='sirs2d/outputs', seed=None):
    """
    Generate smooth corners demonstration.

    Shows comparison of sharp corners (k=0) vs smooth corners with different
    smoothing parameters (k=auto, k=5, k=10).
    """
    print("\n=== Smooth Corners Demo ===")

    if seed is None:
        seed = config.DEFAULT_RANDOM_SEED + 5

    rng = np.random.default_rng(seed)

    # Sample box and bumps
    box = sample_box(rng)
    bumps = sample_bumps(box, rng, edge_bias=True)

    # Auto-compute k
    k_auto = compute_auto_smoothing_k(box)
    k_values = [0.0, k_auto, 5.0, 10.0]
    k_labels = ['Sharp (k=0)', f'Auto (k={k_auto:.1f})', 'Smooth (k=5)', 'Very Smooth (k=10)']

    print(f"Box: q1=[{box['q1_range'][0]:.1f}, {box['q1_range'][1]:.1f}], "
          f"q2=[{box['q2_range'][0]:.1f}, {box['q2_range'][1]:.1f}]")
    print(f"Auto-computed k: {k_auto:.2f}")
    print(f"K = {len(bumps)} bumps")

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    grid_n = config.GRID_RESOLUTION_MAX

    for i, (k, label) in enumerate(zip(k_values, k_labels)):
        ax = axes[i]

        # Evaluate with smooth corners
        use_smooth = (k > 0)
        X, Y, H, M = feasible_mask_grid(box, bumps, grid_n, use_smooth=use_smooth, smoothing_k=k)

        frac = np.mean(M)
        print(f"  {label}: frac = {frac:.2%}")

        # Visualize
        compose_panel(
            box, bumps, X, Y, H, M,
            title=f'{label}: frac={frac:.2%}',
            ax=ax
        )

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'smooth_corners_comparison.png'
    save_figure(output_path)
    plt.close()


def demo_connectivity(output_dir='sirs2d/outputs', seed=None):
    """
    Generate connectivity demonstration.

    Shows side-by-side comparison of connected vs disconnected feasible regions.
    """
    print("\n=== Connectivity Demo ===")

    if seed is None:
        seed = config.DEFAULT_RANDOM_SEED + 6

    rng = np.random.default_rng(seed)

    # Sample common box
    box = sample_box(rng)
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']
    q1_width = q1_max - q1_min
    q2_width = q2_max - q2_min

    print(f"Box: q1=[{q1_min:.1f}, {q1_max:.1f}], q2=[{q2_min:.1f}, {q2_max:.1f}]")

    # Configuration 1: Small bumps (should remain connected)
    bumps_connected = [{
        'mu': np.array([q1_min + 0.3 * q1_width, q2_min + 0.5 * q2_width]),
        'ls': np.array([0.15 * q1_width, 0.15 * q2_width]),
        'alpha': 0.2 * min(q1_width, q2_width)
    }, {
        'mu': np.array([q1_min + 0.7 * q1_width, q2_min + 0.5 * q2_width]),
        'ls': np.array([0.15 * q1_width, 0.15 * q2_width]),
        'alpha': 0.2 * min(q1_width, q2_width)
    }]

    # Configuration 2: Wall of bumps creating left/right disconnection
    # Create a vertical "wall" that separates left and right regions
    bumps_disconnected = [{
        'mu': np.array([q1_min + 0.5 * q1_width, q2_min + 0.25 * q2_width]),
        'ls': np.array([0.12 * q1_width, 0.2 * q2_width]),
        'alpha': 0.6 * min(q1_width, q2_width)
    }, {
        'mu': np.array([q1_min + 0.5 * q1_width, q2_min + 0.5 * q2_width]),
        'ls': np.array([0.12 * q1_width, 0.2 * q2_width]),
        'alpha': 0.6 * min(q1_width, q2_width)
    }, {
        'mu': np.array([q1_min + 0.5 * q1_width, q2_min + 0.75 * q2_width]),
        'ls': np.array([0.12 * q1_width, 0.2 * q2_width]),
        'alpha': 0.6 * min(q1_width, q2_width)
    }]

    grid_n = config.GRID_RESOLUTION_MAX

    # Evaluate both configurations
    # Use smooth corners for connected case, sharp corners for disconnected case
    # (sharp corners make disconnection more likely)
    X_c, Y_c, H_c, M_c = feasible_mask_grid(box, bumps_connected, grid_n, use_smooth=True)
    X_d, Y_d, H_d, M_d = feasible_mask_grid(box, bumps_disconnected, grid_n, use_smooth=False)

    # Check connectivity
    conn_result_c = check_2d_connectivity(box, bumps_connected, 1, 2, grid_n=100, use_smooth=True)
    conn_result_d = check_2d_connectivity(box, bumps_disconnected, 1, 2, grid_n=100, use_smooth=False)

    frac_c = np.mean(M_c)
    frac_d = np.mean(M_d)

    print(f"Connected config: K={len(bumps_connected)}, frac={frac_c:.2%}, "
          f"{conn_result_c['num_components']} component(s)")
    print(f"Disconnected config: K={len(bumps_disconnected)}, frac={frac_d:.2%}, "
          f"{conn_result_d['num_components']} component(s)")

    # Visualize side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    compose_panel(
        box, bumps_connected, X_c, Y_c, H_c, M_c,
        title=f'Connected: K={len(bumps_connected)}, frac={frac_c:.2%}, '
              f'{conn_result_c["num_components"]} component',
        ax=ax1
    )

    compose_panel(
        box, bumps_disconnected, X_d, Y_d, H_d, M_d,
        title=f'Disconnected: K={len(bumps_disconnected)}, frac={frac_d:.2%}, '
              f'{conn_result_d["num_components"]} components',
        ax=ax2
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'connectivity_comparison.png'
    save_figure(output_path)
    plt.close()


def run_all_demos(output_dir='sirs2d/outputs'):
    """
    Run all demonstration scenarios.
    """
    print("=" * 60)
    print("SIRS 2D Demonstrations")
    print("=" * 60)

    demo_single_user(output_dir)
    demo_calibration(output_dir)
    demo_diversity_montage(output_dir)
    demo_edge_coupling(output_dir)
    demo_edge_bias(output_dir)
    demo_smooth_corners(output_dir)
    demo_connectivity(output_dir)

    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    run_all_demos()
