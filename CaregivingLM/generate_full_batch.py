"""
Generate full batch of 100 SIRS-enhanced samples and visualize statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from sirs_batch_sampler import generate_sirs_enhanced_joint_limits
import sirs_sampling_config as config


def collect_statistics(samples):
    """
    Collect statistics from SIRS-enhanced samples.

    Returns:
        Dictionary with statistics for each joint pair
    """
    stats = {
        'per_pair': defaultdict(lambda: {
            'target_feasibilities': [],
            'actual_feasibilities': [],
            'num_bumps': [],
            'box_widths_q1': [],
            'box_widths_q2': [],
        }),
        'n_samples': len(samples),
    }

    for sample in samples:
        if 'sirs_bumps' not in sample:
            continue

        for pair, bumps in sample['sirs_bumps'].items():
            metadata = sample['sirs_metadata'][pair]

            stats['per_pair'][pair]['target_feasibilities'].append(
                metadata['target_feasibility']
            )
            stats['per_pair'][pair]['actual_feasibilities'].append(
                metadata['actual_feasibility']
            )
            stats['per_pair'][pair]['num_bumps'].append(
                metadata['n_bumps']
            )
            stats['per_pair'][pair]['box_widths_q1'].append(
                np.degrees(metadata['box_width_q1'])
            )
            stats['per_pair'][pair]['box_widths_q2'].append(
                np.degrees(metadata['box_width_q2'])
            )

    # Convert lists to arrays
    for pair in stats['per_pair']:
        for key in stats['per_pair'][pair]:
            stats['per_pair'][pair][key] = np.array(stats['per_pair'][pair][key])

    return stats


def visualize_batch_statistics(stats, output_path=None):
    """
    Visualize statistics for the full batch.

    Creates a dashboard showing:
    - Target feasibility distributions per pair
    - Target vs actual feasibility scatter
    - Number of bumps distribution per pair
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    joint_pairs = list(stats['per_pair'].keys())

    # Row 1: Target feasibility histograms
    for i, pair in enumerate(joint_pairs):
        ax = fig.add_subplot(gs[0, i])
        data = stats['per_pair'][pair]['target_feasibilities']

        ax.hist(data * 100, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Target Feasibility (%)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{pair[0].replace("_r", "")}\n×\n{pair[1].replace("_r", "")}',
                     fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add mean line
        mean_val = np.mean(data) * 100
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}%')
        ax.legend(fontsize=8)

    # Row 2: Target vs Actual feasibility scatter
    for i, pair in enumerate(joint_pairs):
        ax = fig.add_subplot(gs[1, i])
        target = stats['per_pair'][pair]['target_feasibilities'] * 100
        actual = stats['per_pair'][pair]['actual_feasibilities'] * 100

        ax.scatter(target, actual, alpha=0.6, s=30, color='steelblue')

        # Add diagonal line (perfect calibration)
        min_val = min(target.min(), actual.min())
        max_val = max(target.max(), actual.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Add ±5% tolerance bands
        ax.fill_between([min_val, max_val],
                        [min_val - 5, max_val - 5],
                        [min_val + 5, max_val + 5],
                        alpha=0.2, color='gray', label='±5% tolerance')

        ax.set_xlabel('Target Feasibility (%)', fontsize=10)
        ax.set_ylabel('Actual Feasibility (%)', fontsize=10)
        ax.set_title('Calibration Accuracy', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

        # Compute error
        error = np.abs(target - actual)
        mean_error = np.mean(error)
        ax.text(0.98, 0.02, f'Mean error: {mean_error:.1f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=8)

    # Row 3: Number of bumps per pair
    for i, pair in enumerate(joint_pairs):
        ax = fig.add_subplot(gs[2, i])
        data = stats['per_pair'][pair]['num_bumps']

        bins = np.arange(data.min(), data.max() + 2) - 0.5
        ax.hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Bumps', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Bump Count Distribution', fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(range(int(data.min()), int(data.max()) + 1))

        # Add mean
        mean_bumps = np.mean(data)
        ax.text(0.98, 0.98, f'Mean: {mean_bumps:.1f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)

    # Overall title
    fig.suptitle(f'Batch Statistics ({stats["n_samples"]} samples)',
                 fontsize=18, fontweight='bold')

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.VIZ_DPI, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_statistics_summary(stats):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print(f"Batch Statistics Summary ({stats['n_samples']} samples)")
    print("=" * 70)

    for pair in stats['per_pair']:
        j1, j2 = pair
        data = stats['per_pair'][pair]

        print(f"\n{j1} × {j2}:")
        print(f"  Target feasibility: {np.mean(data['target_feasibilities']):.2%} "
              f"± {np.std(data['target_feasibilities']):.2%}")
        print(f"  Actual feasibility: {np.mean(data['actual_feasibilities']):.2%} "
              f"± {np.std(data['actual_feasibilities']):.2%}")

        # Calibration error
        error = np.abs(data['target_feasibilities'] - data['actual_feasibilities'])
        print(f"  Calibration error: {np.mean(error):.2%} ± {np.std(error):.2%}")
        print(f"  Max error: {np.max(error):.2%}")

        print(f"  Number of bumps: {np.mean(data['num_bumps']):.1f} "
              f"± {np.std(data['num_bumps']):.1f}")
        print(f"  Box width (q1): {np.mean(data['box_widths_q1']):.1f}° "
              f"± {np.std(data['box_widths_q1']):.1f}°")
        print(f"  Box width (q2): {np.mean(data['box_widths_q2']):.1f}° "
              f"± {np.std(data['box_widths_q2']):.1f}°")

    print("=" * 70)


if __name__ == '__main__':
    print("=" * 70)
    print("Generating Full 100-Sample Batch")
    print("=" * 70)

    # Generate samples
    print("\nGenerating 100 SIRS-enhanced samples...")
    print("(This may take a few minutes)\n")

    samples = generate_sirs_enhanced_joint_limits(
        n_samples_per_batch=config.N_SAMPLES_PER_BATCH,  # 10 per batch × 10 batches = 100
        verbose=True
    )

    print(f"\n✓ Generated {len(samples)} samples")

    # Collect statistics
    print("\nCollecting statistics...")
    stats = collect_statistics(samples)

    # Print summary
    print_statistics_summary(stats)

    # Visualize
    print("\nGenerating statistics dashboard...")
    output_dir = Path(config.OUTPUT_DIR)
    output_path = output_dir / 'batch_statistics.png'
    visualize_batch_statistics(stats, output_path)

    print("\n✓ Phase 6 complete!")
