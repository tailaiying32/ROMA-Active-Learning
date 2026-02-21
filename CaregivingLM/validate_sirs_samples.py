"""
Validation utilities for SIRS-enhanced joint limit samples.

Checks connectivity and other properties to ensure samples are valid.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sirs2d.sirs import check_2d_connectivity
import sirs_sampling_config as config


def check_sample_connectivity(sample, grid_n=100, use_smooth=True, early_exit=True):
    """
    Check connectivity for all joint pairs in a single sample.

    Args:
        sample: SIRS-enhanced sample dictionary
        grid_n: Grid resolution for connectivity check
        use_smooth: Use smooth corners for connectivity check
        early_exit: If True, return immediately when first disconnected pair is found (faster)

    Returns:
        Dictionary with connectivity results for each pair (may be incomplete if early_exit=True)
    """
    if 'sirs_bumps' not in sample:
        return {}

    results = {}

    for pair, bumps in sample['sirs_bumps'].items():
        j1, j2 = pair

        # Extract box
        lower1, upper1 = sample['joint_limits'][j1]
        lower2, upper2 = sample['joint_limits'][j2]

        box = {
            'q1_range': (lower1, upper1),
            'q2_range': (lower2, upper2)
        }

        # Check connectivity
        conn_result = check_2d_connectivity(
            box, bumps, 1, 2,
            grid_n=grid_n,
            use_smooth=use_smooth
        )

        results[pair] = conn_result

        # Early exit if disconnected (optimization for rejection sampling)
        if early_exit and not conn_result['is_connected']:
            break

    return results


def validate_batch_connectivity(samples, grid_n=100, use_smooth=True, verbose=True):
    """
    Validate connectivity for entire batch of samples.

    Args:
        samples: List of SIRS-enhanced samples
        grid_n: Grid resolution for connectivity check
        use_smooth: Use smooth corners
        verbose: Print progress

    Returns:
        Dictionary with validation statistics and detailed results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("Connectivity Validation")
        print("=" * 70)
        print(f"Validating {len(samples)} samples...")
        print(f"Grid resolution: {grid_n}×{grid_n}")
        print()

    all_results = []
    disconnected_count = 0
    total_pairs = 0

    for i, sample in enumerate(samples):
        if verbose and (i % 10 == 0 or i == len(samples) - 1):
            print(f"  Checking sample {i+1}/{len(samples)}...", end='\r')

        conn_results = check_sample_connectivity(sample, grid_n, use_smooth)
        all_results.append(conn_results)

        # Count disconnected pairs
        for pair, result in conn_results.items():
            total_pairs += 1
            if not result['is_connected']:
                disconnected_count += 1

    if verbose:
        print()

    # Compute statistics
    disconnected_fraction = disconnected_count / total_pairs if total_pairs > 0 else 0

    # Find most problematic pairs
    pair_disconnect_counts = {}
    for conn_results in all_results:
        for pair, result in conn_results.items():
            if pair not in pair_disconnect_counts:
                pair_disconnect_counts[pair] = {'disconnected': 0, 'total': 0}

            pair_disconnect_counts[pair]['total'] += 1
            if not result['is_connected']:
                pair_disconnect_counts[pair]['disconnected'] += 1

    # Sort pairs by disconnection rate
    pair_stats = []
    for pair, counts in pair_disconnect_counts.items():
        rate = counts['disconnected'] / counts['total']
        pair_stats.append((pair, counts['disconnected'], counts['total'], rate))

    pair_stats.sort(key=lambda x: x[3], reverse=True)

    # Print summary
    if verbose:
        print(f"\n[Summary]")
        print(f"  Total samples: {len(samples)}")
        print(f"  Total pair checks: {total_pairs}")
        print(f"  Disconnected pairs: {disconnected_count}")
        print(f"  Disconnection rate: {disconnected_fraction:.2%}")

        print(f"\n[Per-Pair Statistics]")
        for pair, disc, total, rate in pair_stats:
            j1, j2 = pair
            status = "⚠" if rate > 0.1 else "✓"
            print(f"  {status} {j1} × {j2}:")
            print(f"      {disc}/{total} disconnected ({rate:.1%})")

        # Decision
        print(f"\n[Validation Result]")
        threshold = config.MAX_DISCONNECTED_FRACTION
        if disconnected_fraction <= threshold:
            print(f"  ✓ PASS: Disconnection rate {disconnected_fraction:.1%} ≤ {threshold:.0%} threshold")
        else:
            print(f"  ✗ FAIL: Disconnection rate {disconnected_fraction:.1%} > {threshold:.0%} threshold")
            print(f"  Recommendation: Adjust SIRS parameters to reduce disconnection")

    return {
        'all_results': all_results,
        'disconnected_count': disconnected_count,
        'total_pairs': total_pairs,
        'disconnected_fraction': disconnected_fraction,
        'pair_stats': pair_stats,
        'passed': disconnected_fraction <= config.MAX_DISCONNECTED_FRACTION
    }


def visualize_connectivity_stats(validation_results, output_path=None):
    """
    Visualize connectivity validation results.

    Args:
        validation_results: Output from validate_batch_connectivity()
        output_path: Path to save figure (None = show only)
    """
    pair_stats = validation_results['pair_stats']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Bar chart of disconnection rates
    pair_labels = [f"{j1}\n×\n{j2}" for (j1, j2), _, _, _ in pair_stats]
    rates = [rate * 100 for _, _, _, rate in pair_stats]
    colors = ['red' if r > 10 else 'green' for r in rates]

    ax1.bar(range(len(pair_labels)), rates, color=colors, alpha=0.7)
    ax1.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='10% threshold')
    ax1.set_xticks(range(len(pair_labels)))
    ax1.set_xticklabels(pair_labels, fontsize=9)
    ax1.set_ylabel('Disconnection Rate (%)', fontsize=12)
    ax1.set_title('Disconnection Rate by Joint Pair', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: Component count distribution
    all_results = validation_results['all_results']
    component_counts = []

    for conn_results in all_results:
        for result in conn_results.values():
            component_counts.append(result['num_components'])

    ax2.hist(component_counts, bins=range(1, max(component_counts) + 2),
             alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Connected Components', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add text annotation
    n_connected = sum(1 for c in component_counts if c == 1)
    pct_connected = n_connected / len(component_counts) * 100
    ax2.text(0.98, 0.98, f'{pct_connected:.1f}% fully connected',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=config.VIZ_DPI, bbox_inches='tight')
        print(f"  Saved connectivity stats to: {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Test connectivity validation
    print("Testing connectivity validation...")

    from sirs_batch_sampler import generate_sirs_enhanced_joint_limits

    # Generate test samples
    samples = generate_sirs_enhanced_joint_limits(
        n_samples_per_batch=1,  # 10 samples
        verbose=False
    )

    # Validate connectivity
    validation_results = validate_batch_connectivity(
        samples,
        grid_n=config.CONNECTIVITY_GRID_N,
        use_smooth=config.USE_SMOOTH_CORNERS,
        verbose=True
    )

    # Visualize
    output_dir = Path(config.OUTPUT_DIR)
    output_path = output_dir / 'connectivity_stats.png'
    visualize_connectivity_stats(validation_results, output_path)

    print(f"\n✓ Validation complete!")
