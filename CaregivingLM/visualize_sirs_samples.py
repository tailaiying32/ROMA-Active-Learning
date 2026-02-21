"""
Visualization utilities for SIRS-enhanced joint limit samples.

Creates comprehensive visualizations showing feasible regions with SIRS bumps.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sirs2d.sirs import feasible_mask_grid
from sirs2d.visualize import compose_panel
import sirs_sampling_config as config


def visualize_single_sample_pairwise(sample, output_path=None, grid_n=None):
    """
    Visualize all joint pairs for a single sample in a 2×2 grid.

    Args:
        sample: SIRS-enhanced sample dictionary
        output_path: Path to save figure (None = show only)
        grid_n: Grid resolution (None = use config)
    """
    if 'sirs_bumps' not in sample or len(sample['sirs_bumps']) == 0:
        print("Warning: No SIRS bumps in sample")
        return

    grid_n = grid_n or config.VIZ_GRID_RESOLUTION

    # Get joint pairs
    joint_pairs = list(sample['sirs_bumps'].keys())
    n_pairs = len(joint_pairs)

    # Create 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i, pair in enumerate(joint_pairs):
        if i >= 4:
            break

        ax = axes[i]
        j1, j2 = pair

        # Extract box
        lower1, upper1 = sample['joint_limits'][j1]
        lower2, upper2 = sample['joint_limits'][j2]

        box = {
            'q1_range': (lower1, upper1),
            'q2_range': (lower2, upper2)
        }

        # Get bumps
        bumps = sample['sirs_bumps'][pair]

        # Get metadata
        metadata = sample['sirs_metadata'][pair]

        # Evaluate feasible region
        X, Y, H, M = feasible_mask_grid(
            box, bumps, grid_n,
            use_smooth=config.USE_SMOOTH_CORNERS
        )

        # Visualize
        compose_panel(box, bumps, X, Y, H, M, ax=ax)

        # Title with feasibility info
        title = f'{j1.replace("_r", "")}\n×\n{j2.replace("_r", "")}'
        subtitle = f'Target: {metadata["target_feasibility"]:.1%}, ' \
                   f'Actual: {metadata["actual_feasibility"]:.1%}, ' \
                   f'{metadata["n_bumps"]} bumps'

        ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold')

        # Convert axes to degrees
        ax.set_xlabel(f'{j1.replace("_r", "")} (degrees)', fontsize=10)
        ax.set_ylabel(f'{j2.replace("_r", "")} (degrees)', fontsize=10)

        # Get current tick labels and convert to degrees
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        ax.set_xticklabels([f'{np.degrees(x):.0f}' for x in xticks], fontsize=9)
        ax.set_yticklabels([f'{np.degrees(y):.0f}' for y in yticks], fontsize=9)

    # Hide unused subplots
    for i in range(n_pairs, 4):
        axes[i].axis('off')

    # Add overall title
    sample_id = sample['id']
    fig.suptitle(f'SIRS-Enhanced Joint Limits (Sample {sample_id})',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.VIZ_DPI, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_multi_sample_comparison(samples, pair_index=0, n_samples=12,
                                      output_path=None, grid_n=None):
    """
    Visualize one joint pair across multiple samples (montage).

    Args:
        samples: List of SIRS-enhanced samples
        pair_index: Which joint pair to visualize (0-3)
        n_samples: Number of samples to show
        output_path: Path to save figure (None = show only)
        grid_n: Grid resolution (None = use config)
    """
    grid_n = grid_n or config.VIZ_GRID_RESOLUTION

    # Select samples
    if len(samples) < n_samples:
        selected_samples = samples
        n_samples = len(samples)
    else:
        # Select evenly spaced samples
        indices = np.linspace(0, len(samples) - 1, n_samples, dtype=int)
        selected_samples = [samples[i] for i in indices]

    # Get the joint pair from first sample
    first_sample = selected_samples[0]
    if 'sirs_bumps' not in first_sample:
        print("Warning: No SIRS bumps in samples")
        return

    joint_pairs = list(first_sample['sirs_bumps'].keys())
    if pair_index >= len(joint_pairs):
        pair_index = 0

    pair = joint_pairs[pair_index]
    j1, j2 = pair

    # Create montage grid (3 rows × 4 cols)
    n_rows = 3
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    axes = axes.flatten()

    for i, sample in enumerate(selected_samples):
        if i >= n_samples:
            break

        ax = axes[i]

        # Extract box
        lower1, upper1 = sample['joint_limits'][j1]
        lower2, upper2 = sample['joint_limits'][j2]

        box = {
            'q1_range': (lower1, upper1),
            'q2_range': (lower2, upper2)
        }

        # Get bumps
        bumps = sample['sirs_bumps'].get(pair, [])

        # Get metadata
        metadata = sample['sirs_metadata'].get(pair, {})

        # Evaluate feasible region
        X, Y, H, M = feasible_mask_grid(
            box, bumps, grid_n,
            use_smooth=config.USE_SMOOTH_CORNERS
        )

        # Visualize
        compose_panel(box, bumps, X, Y, H, M, ax=ax, show_box=True, show_bumps=True)

        # Title
        sample_id = sample['id']
        feas = metadata.get('actual_feasibility', 0)
        n_bumps = metadata.get('n_bumps', 0)

        ax.set_title(f'Sample {sample_id}\n{feas:.1%} feasible, {n_bumps} bumps',
                     fontsize=9, fontweight='bold')

        # Smaller labels
        ax.set_xlabel('', fontsize=8)
        ax.set_ylabel('', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    # Add overall title
    pair_name = f'{j1.replace("_r", "")} × {j2.replace("_r", "")}'
    fig.suptitle(f'Multi-Sample Comparison: {pair_name}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.VIZ_DPI, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    print("Testing visualization utilities...")

    from sirs_batch_sampler import generate_sirs_enhanced_joint_limits

    # Generate test samples
    print("\nGenerating test samples...")
    samples = generate_sirs_enhanced_joint_limits(
        n_samples_per_batch=1,  # 10 samples
        verbose=False
    )

    print(f"Generated {len(samples)} samples\n")

    # Visualize first sample
    print("[Phase 4] Visualizing single sample (all 4 pairs)...")
    output_dir = Path(config.OUTPUT_DIR)
    output_path = output_dir / 'sample_0_pairwise_regions.png'
    visualize_single_sample_pairwise(samples[0], output_path)

    # Visualize multi-sample comparison
    print("\n[Phase 5] Visualizing multi-sample comparison...")
    output_path = output_dir / 'multi_sample_comparison.png'
    visualize_multi_sample_comparison(samples, pair_index=0, n_samples=10, output_path=output_path)

    print("\n✓ Visualizations complete!")
