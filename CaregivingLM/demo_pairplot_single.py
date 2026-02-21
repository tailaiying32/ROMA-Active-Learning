"""
Demo: Generate pairplot visualization for a single sample.

Shows how to:
- Load a specific sample from HDF5
- Generate feasible points using rejection sampling
- Create 4×4 pairplot for main joints (shoulder + elbow)
- Save visualization to file
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from export_to_hdf5 import load_sample_from_hdf5
from create_pairplots import sample_feasible_points_from_region, JOINT_ORDER, JOINT_LABELS, FULL_JOINT_RANGES


def create_pairplot_for_sample(hdf5_path='output/sirs_sampling/sirs_samples_100.h5',
                               sample_index=0,
                               n_points=5000,
                               max_attempts=None,
                               output_dir=None,
                               dpi=150,
                               show=False):
    """
    Create pairplot visualization for a single sample.

    Args:
        hdf5_path: Path to HDF5 file
        sample_index: Index of sample to visualize
        n_points: Number of feasible points to sample
        max_attempts: Maximum Halton candidates (None = 10x n_points)
        output_dir: Output directory (None = auto)
        dpi: DPI for saved figure
        show: If True, display plot interactively
    """
    hdf5_path = Path(hdf5_path)

    if not hdf5_path.exists():
        print(f"Error: File not found: {hdf5_path}")
        return

    print("=" * 70)
    print(f"Pairplot Visualization for Sample {sample_index}")
    print("=" * 70)

    # ====================================================================
    # Load Sample
    # ====================================================================
    print(f"\n[Loading Sample]")
    print(f"  File: {hdf5_path.name}")
    print(f"  Sample index: {sample_index}")

    sample = load_sample_from_hdf5(hdf5_path, sample_id=sample_index)

    print(f"  Sample ID: {sample.get('id', f'sample_{sample_index}')}")
    print(f"  Number of joints: {len(sample['joint_limits'])}")
    print(f"  Number of SIRS pairs: {len(sample.get('sirs_bumps', {}))}")

    # Display feasibility info
    if 'sirs_metadata' in sample:
        print(f"\n  Feasibility per pair:")
        for pair, meta in sample['sirs_metadata'].items():
            j1, j2 = pair
            print(f"    {j1} × {j2}: "
                  f"target={meta['target_feasibility']:.1%}, "
                  f"actual={meta['actual_feasibility']:.1%}, "
                  f"bumps={meta['n_bumps']}")

    # ====================================================================
    # Sample Feasible Points
    # ====================================================================
    print(f"\n[Sampling Feasible Points]")
    print(f"  Target: {n_points} points")
    print(f"  Method: Halton sequence + rejection sampling")

    df = sample_feasible_points_from_region(
        sample,
        n_points=n_points,
        max_attempts=max_attempts
    )

    if df is None:
        print(f"\n✗ Failed to generate sufficient feasible points")
        print(f"  This sample may have very low feasibility or disconnected regions")
        return

    print(f"  ✓ Generated {len(df)} feasible points")

    # ====================================================================
    # Create Pairplot
    # ====================================================================
    print(f"\n[Creating Pairplot]")
    print(f"  Joints: {', '.join(JOINT_ORDER)}")

    # Configure seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("flare")

    # Create pairplot
    pairplot = sns.pairplot(
        df,
        diag_kind='hist',
        plot_kws={
            'alpha': 0.7,
            's': 15,
            'edgecolor': 'none',
            'color': '#E74C3C',  # Vibrant coral/red
        },
        diag_kws={
            'bins': 30,
            'edgecolor': '#C0392B',
            'color': '#E74C3C',
            'alpha': 0.8
        }
    )

    # Set axis limits to full physiological ranges
    n_vars = len(JOINT_ORDER)
    for i in range(n_vars):
        for j in range(n_vars):
            ax = pairplot.axes[i, j]

            # Set x-axis limits (column joint)
            x_joint = JOINT_ORDER[j]
            x_min, x_max = FULL_JOINT_RANGES[x_joint]
            ax.set_xlim(x_min, x_max)

            # Set y-axis limits (row joint) - except diagonal
            if i != j:
                y_joint = JOINT_ORDER[i]
                y_min, y_max = FULL_JOINT_RANGES[y_joint]
                ax.set_ylim(y_min, y_max)

    # Add title
    sample_id = sample.get('id', f'sample_{sample_index}')
    pairplot.fig.suptitle(
        f'4-Joint Pairplot: Sample {sample_index} ({len(df)} feasible points)',
        fontsize=16,
        fontweight='bold',
        y=1.01
    )

    # Tight layout
    pairplot.fig.tight_layout()

    # ====================================================================
    # Save Figure
    # ====================================================================
    print(f"\n[Saving Visualization]")

    if output_dir is None:
        output_dir = Path('output/pairplots')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'pairplot_sample_{sample_index}.png'
    pairplot.savefig(output_path, dpi=dpi, bbox_inches='tight')

    file_size_kb = output_path.stat().st_size / 1024
    print(f"  ✓ Saved: {output_path}")
    print(f"  Size: {file_size_kb:.1f} KB")
    print(f"  DPI: {dpi}")

    # ====================================================================
    # Display (optional)
    # ====================================================================
    if show:
        print(f"\n[Displaying Plot]")
        plt.show()
    else:
        plt.close()

    print("\n" + "=" * 70)
    print("✓ Pairplot generation complete!")
    print("=" * 70)

    return output_path


if __name__ == '__main__':
    import sys

    # Parse command line arguments
    hdf5_path = 'output/sirs_sampling/sirs_samples_100.h5'
    sample_index = 0
    n_points = 5000
    show = False

    print("\nUsage: python demo_pairplot_single.py [hdf5_path] [sample_index] [n_points] [--show]")
    print(f"  hdf5_path: Path to HDF5 file (default: {hdf5_path})")
    print(f"  sample_index: Sample index to visualize (default: {sample_index})")
    print(f"  n_points: Number of points to sample (default: {n_points})")
    print(f"  --show: Display plot interactively\n")

    # Parse arguments
    args = [arg for arg in sys.argv[1:] if arg != '--show']
    if '--show' in sys.argv:
        show = True

    if len(args) > 0:
        hdf5_path = args[0]
    if len(args) > 1:
        sample_index = int(args[1])
    if len(args) > 2:
        n_points = int(args[2])

    create_pairplot_for_sample(
        hdf5_path=hdf5_path,
        sample_index=sample_index,
        n_points=n_points,
        dpi=150,
        show=show
    )
