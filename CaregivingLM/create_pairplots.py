"""
Create pairplot visualizations for SIRS-enhanced joint limit samples.

Uses seaborn.pairplot to show the full 4-joint space (shoulder_flexion,
shoulder_abduction, shoulder_rotation, elbow_flexion) by sampling points
from the SIRS-constrained feasible regions.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add scone directory to path for Halton sampling
sys.path.insert(0, str(Path(__file__).parent / 'scone'))
from batch_joint_limit_sampler import generate_halton_samples

from sirs2d.sirs import h_value
import sirs_sampling_config as config


# Map joint names to indices
JOINT_ORDER = [
    'shoulder_flexion_r',
    'shoulder_abduction_r',
    'shoulder_rotation_r',
    'elbow_flexion_r'
]

# Short labels for plots (in degrees)
JOINT_LABELS = [
    'Flexion (°)',
    'Abduction (°)',
    'Rotation (°)',
    'Elbow (°)'
]

# Full physiological joint ranges (from batch_joint_limit_sampler.py)
FULL_JOINT_RANGES = {
    'shoulder_flexion_r': (-45, 150),      # degrees
    'shoulder_abduction_r': (-60, 120),    # degrees
    'shoulder_rotation_r': (-60, 60),      # degrees
    'elbow_flexion_r': (0, 130),           # degrees
}


def sample_feasible_points_from_region(sample, n_points=5000, max_attempts=None):
    """
    Sample feasible points from a SIRS-enhanced joint limit set using Halton sequences.

    Uses Halton sequence sampling for better coverage with O(n) complexity instead
    of O(grid_size^n) for grid-based methods.

    Args:
        sample: SIRS-enhanced sample dictionary
        n_points: Target number of feasible points to sample
        max_attempts: Maximum Halton candidates to generate (None = 10x n_points)

    Returns:
        DataFrame with columns for each joint (in degrees) containing feasible points
    """
    if max_attempts is None:
        max_attempts = n_points * 10  # Default: 10x oversampling
    # Extract ranges for the 4 main joints
    ranges = []
    for joint_name in JOINT_ORDER:
        if joint_name in sample['joint_limits']:
            ranges.append(sample['joint_limits'][joint_name])
        else:
            # Fallback to full range if not in sample
            from scone.batch_joint_limit_sampler import BASE_JOINT_RANGES
            ranges.append(BASE_JOINT_RANGES[joint_name])

    # Create mapping from joint pair to bumps
    sirs_constraints = {}
    if 'sirs_bumps' in sample:
        for pair_key, bumps in sample['sirs_bumps'].items():
            j1, j2 = pair_key
            if j1 in JOINT_ORDER and j2 in JOINT_ORDER:
                i = JOINT_ORDER.index(j1)
                j = JOINT_ORDER.index(j2)
                # Store with canonical ordering
                key = (min(i, j), max(i, j))
                sirs_constraints[key] = bumps

    def is_point_feasible(q):
        """Check if a point satisfies all SIRS constraints."""
        for (i, j), bumps in sirs_constraints.items():
            box = {
                'q1_range': ranges[i],
                'q2_range': ranges[j]
            }
            q_pair = np.array([q[i], q[j]])
            h = h_value(q_pair, box, bumps,
                       use_smooth=config.USE_SMOOTH_CORNERS,
                       smoothing_k='auto' if config.SMOOTHING_K_AUTO else 0.0)
            if h < 0:
                return False
        return True

    # Generate Halton samples in [0,1]^4
    n_dims = len(ranges)
    print(f"    Using Halton sampling: {max_attempts:,} candidates for {n_dims} joints")

    halton_samples = generate_halton_samples(max_attempts, n_dims)

    # Scale to joint ranges
    ranges_array = np.array(ranges)
    lower_bounds = ranges_array[:, 0]
    upper_bounds = ranges_array[:, 1]

    scaled_samples = lower_bounds + halton_samples * (upper_bounds - lower_bounds)

    # Rejection sampling with SIRS feasibility
    feasible_points = []
    for i, q in enumerate(scaled_samples):
        if len(feasible_points) >= n_points:
            break

        if is_point_feasible(q):
            feasible_points.append(q)

    coverage_pct = len(feasible_points) / n_points * 100
    acceptance_pct = len(feasible_points) / min(i+1, max_attempts) * 100
    print(f"    Halton sampling: {len(feasible_points)}/{n_points} points ({coverage_pct:.1f}% coverage, {acceptance_pct:.1f}% acceptance)")

    if len(feasible_points) < n_points:
        print(f"    Warning: Only sampled {len(feasible_points)}/{n_points} points")

    # Check if we have enough points
    if len(feasible_points) < 100:  # Need at least 100 points for meaningful visualization
        print(f"    Error: Insufficient feasible points ({len(feasible_points)}), cannot create pairplot")
        return None

    # Convert to DataFrame with degrees
    points_deg = np.degrees(np.array(feasible_points))
    df = pd.DataFrame(points_deg, columns=JOINT_LABELS)

    return df


def create_pairplot_4joints(sample, output_path=None, n_points=5000, dpi=150):
    """
    Create a 4×4 pairplot for one sample using seaborn.pairplot.

    Args:
        sample: SIRS-enhanced sample dictionary
        output_path: Where to save the plot
        n_points: Number of points to sample from feasible region
        dpi: DPI for saved figure
    """
    # Sample feasible points from the region
    print(f"    Sampling {n_points} feasible points...")
    df = sample_feasible_points_from_region(sample, n_points=n_points)

    # Check if sampling failed
    if df is None:
        print(f"    ✗ Skipping sample {sample.get('id', 'unknown')} - insufficient feasible points")
        return

    print(f"    Creating seaborn pairplot...")

    # Create pairplot using seaborn
    sample_id = sample.get('id', 'unknown')

    # Use a fancy color palette - sunset/thermal gradient
    # Options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'rocket', 'mako', 'flare'
    sns.set_palette("flare")  # Beautiful warm gradient: purple -> red -> orange -> yellow

    # Use seaborn's pairplot
    pairplot = sns.pairplot(
        df,
        diag_kind='hist',  # Histogram on diagonal
        plot_kws={
            'alpha': 0.7,
            's': 15,
            'edgecolor': 'none',
            'color': '#E74C3C',  # Vibrant coral/red color
        },
        diag_kws={
            'bins': 30,
            'edgecolor': '#C0392B',  # Darker red for edges
            'color': '#E74C3C',      # Match scatter color
            'alpha': 0.8
        }
    )

    # Set axis limits to full physiological ranges for all subplots
    n_vars = len(JOINT_ORDER)
    for i in range(n_vars):
        for j in range(n_vars):
            ax = pairplot.axes[i, j]

            # Set x-axis limits (column joint)
            x_joint = JOINT_ORDER[j]
            x_min, x_max = FULL_JOINT_RANGES[x_joint]
            ax.set_xlim(x_min, x_max)

            # Set y-axis limits (row joint) - except for diagonal
            if i != j:
                y_joint = JOINT_ORDER[i]
                y_min, y_max = FULL_JOINT_RANGES[y_joint]
                ax.set_ylim(y_min, y_max)

    # Add title
    pairplot.fig.suptitle(f'4-Joint Pairplot (Sample {sample_id})',
                         fontsize=16, fontweight='bold', y=1.01)

    # Adjust layout
    pairplot.fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pairplot.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def create_multiple_pairplots(samples, sample_indices=None, output_dir=None):
    """
    Create pairplots for multiple samples.

    Args:
        samples: List of SIRS-enhanced samples
        sample_indices: List of original sample IDs (for filenames). If None, use sequential indices.
        output_dir: Directory to save plots
    """
    if sample_indices is None:
        # Use sequential indices
        sample_indices = list(range(len(samples)))

    output_dir = Path(output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(samples)} pairplots...")

    for i, sample in enumerate(samples):
        # Use original sample ID for filename
        sample_id = sample_indices[i] if i < len(sample_indices) else i
        output_path = output_dir / f'pairplot_sample_{sample_id}.png'

        print(f"  Creating pairplot for sample {sample_id}...")
        create_pairplot_4joints(sample, output_path=output_path, n_points=5000)


if __name__ == '__main__':
    from export_to_hdf5 import load_sample_from_hdf5
    import h5py

    print("=" * 70)
    print("Phase 7: Pairplot Visualizations")
    print("=" * 70)

    # Find HDF5 file in output directory
    output_dir = Path(config.OUTPUT_DIR)
    hdf5_files = list(output_dir.glob('sirs_samples_*.h5'))

    if not hdf5_files:
        print(f"\n✗ Error: No HDF5 files found in {output_dir}")
        print("Please run export_to_hdf5.py first to generate the file.")
        exit(1)

    # Use the most recent file if multiple exist
    hdf5_path = max(hdf5_files, key=lambda p: p.stat().st_mtime)

    print(f"\nLoading samples from {hdf5_path.name}...")

    # Get number of samples from HDF5
    with h5py.File(hdf5_path, 'r') as f:
        n_samples = f['metadata'].attrs['n_samples']

    print(f"✓ Found {n_samples} samples in HDF5 file")

    # Load sample indices - choose samples with higher feasibility
    # sample_indices = [0, 25, 50, 75, 99]
    # Randomly select additional samples
    rng = np.random.default_rng(123)
    sample_indices = rng.choice(n_samples, size=50, replace=False).tolist()
    # Filter out indices that don't exist
    sample_indices = [idx for idx in sample_indices if idx < n_samples]

    print(f"\nGenerating pairplots for samples: {sample_indices}")
    print("(Samples with low feasibility will be skipped)")

    # Load samples and check feasibility
    samples_to_plot = []
    indices_to_plot = []

    for idx in sample_indices:
        sample = load_sample_from_hdf5(hdf5_path, sample_id=idx)

        # Check average feasibility
        if 'sirs_metadata' in sample:
            avg_feas = np.mean([meta['actual_feasibility']
                               for meta in sample['sirs_metadata'].values()])
            print(f"  Sample {idx}: avg feasibility = {avg_feas:.1%}")

            # Only include samples with reasonable feasibility
            if avg_feas >= 0.3:  # At least 30% feasible
                samples_to_plot.append(sample)
                indices_to_plot.append(idx)
            else:
                print(f"    ⚠ Skipping - too low feasibility")
        else:
            samples_to_plot.append(sample)
            indices_to_plot.append(idx)

    if not samples_to_plot:
        print("\n✗ No samples with sufficient feasibility found!")
        exit(1)

    print(f"\n✓ Selected {len(samples_to_plot)} samples for visualization: {indices_to_plot}")

    # Create pairplots
    create_multiple_pairplots(
        samples_to_plot,
        sample_indices=indices_to_plot,  # Use original sample IDs
        output_dir=config.OUTPUT_DIR
    )

    print("\n✓ Phase 7 complete!")
5 