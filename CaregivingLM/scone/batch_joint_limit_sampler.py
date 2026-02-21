#!/usr/bin/env python3
"""
Batch Joint Limit Sampler

Generates diverse sets of joint limits using Halton sequences to ensure good coverage
of the joint limit space. Each joint contributes 2 dimensions (lower, upper) for a
total of 20D sampling space for 10 DOFs.

Target joints (10 DOFs):
- clavicle_protraction_r, clavicle_elevation_r, clavicle_rotation_r
- scapula_abduction_r, scapula_elevation_r, scapula_winging_r
- shoulder_flexion_r, shoulder_abduction_r, shoulder_rotation_r
- elbow_flexion_r

Usage:
    python batch_joint_limit_sampler.py --n-samples 1000 --visualize
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

def halton_sequence(index, base):
    """Generate the index-th number in the Halton sequence for given base."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result

def generate_halton_samples(n_samples, n_dims):
    """Generate n_samples of n_dims using Halton sequence with different prime bases."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    if n_dims > len(primes):
        raise ValueError(f"Too many dimensions ({n_dims}). Max supported: {len(primes)}")

    samples = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        for j in range(n_dims):
            samples[i, j] = halton_sequence(i + 1, primes[j])
    return samples

def get_base_joint_limits():
    """Get the physiological joint limits from joint_coverage_sampling.py as reference ranges."""
    deg_to_rad = np.pi / 180.0

    base_limits = {
        'clavicle_protraction_r': (-30*deg_to_rad, 10*deg_to_rad),
        'clavicle_elevation_r': (-15*deg_to_rad, 45*deg_to_rad),
        'clavicle_rotation_r': (-15*deg_to_rad, 60*deg_to_rad),
        'scapula_abduction_r': (-35*deg_to_rad, 45*deg_to_rad),
        'scapula_elevation_r': (-45*deg_to_rad, 45*deg_to_rad),
        'scapula_winging_r': (-45*deg_to_rad, 45*deg_to_rad),
        'shoulder_flexion_r': (-45*deg_to_rad, 150*deg_to_rad),
        'shoulder_abduction_r': (-60*deg_to_rad, 120*deg_to_rad),
        'shoulder_rotation_r': (-60*deg_to_rad, 60*deg_to_rad),
        'elbow_flexion_r': (0*deg_to_rad, 130*deg_to_rad)
    }

    return base_limits

def generate_procedural_joint_limits(n_samples_per_batch, coverage_factor=0.8, min_range_factor=0.1, 
                                   main_dofs_only=False, seed=42):
    """
    Generate joint limit sets using procedural sampling approach.
    Creates batches with decreasing numbers of limited joints (13 -> 12 -> ... -> 1).

    Args:
        n_samples_per_batch: Number of joint limit sets to generate per batch
        coverage_factor: What fraction of the base range to use for sampling (0.8 = 80%)
        min_range_factor: Minimum range as fraction of base range (0.1 = 10%)
        main_dofs_only: If True, generate limits only for 3 shoulder DOFs + elbow (default: False)
        seed: Random seed for reproducible joint selection (default: 42)

    Returns:
        List of joint limit dictionaries with metadata about which joints are limited
    """
    base_limits = get_base_joint_limits()
    
    # Filter to only main DOFs if requested
    if main_dofs_only:
        main_dofs = ['shoulder_flexion_r', 'shoulder_abduction_r', 'shoulder_rotation_r', 'elbow_flexion_r']
        base_limits = {name: limits for name, limits in base_limits.items() if name in main_dofs}
        print(f"Main DOFs only mode: generating limits for {list(base_limits.keys())}")
    
    all_joint_names = list(base_limits.keys())
    n_total_joints = len(all_joint_names)
    
    print(f"Procedural sampling: {n_total_joints} joints, {n_samples_per_batch} samples per batch")
    print(f"Coverage factor: {coverage_factor:.1%}, Min range factor: {min_range_factor:.1%}")
    
    joint_limit_sets = []
    sample_id = 0
    
    # Use the maximum number of dimensions needed (all joints * 2)
    max_dims = len(all_joint_names) * 2
    total_samples = n_samples_per_batch * n_total_joints
    
    # Generate all Halton samples at once for consistency
    all_halton_samples = generate_halton_samples(total_samples, max_dims)
    halton_index = 0
    
    # Generate batches from n_total_joints down to 1 limited joint
    for n_limited_joints in range(n_total_joints, 0, -1):
        print(f"\nBatch {n_total_joints - n_limited_joints + 1}: Limiting {n_limited_joints} out of {n_total_joints} joints")
        
        # Generate multiple samples for this number of limited joints
        for batch_sample in range(n_samples_per_batch):
            # Set unique random seed for each sample to ensure diversity
            sample_seed = seed + sample_id * 7  # Use prime number offset for better distribution
            np.random.seed(sample_seed)
            
            # Randomly select which joints to limit for THIS SAMPLE
            if n_limited_joints == n_total_joints:
                # Limit all joints
                limited_joints = all_joint_names.copy()
            else:
                # Randomly select joints to limit (different selection for each sample)
                limited_joints = np.random.choice(all_joint_names, size=n_limited_joints, replace=False).tolist()
            
            # Get the Halton sample for this iteration
            halton_sample = all_halton_samples[halton_index]
            halton_index += 1
            
            joint_limits = {}
            
            for joint_name in limited_joints:
                base_min, base_max = base_limits[joint_name]
                base_range = base_max - base_min
                
                # Find the index of this joint in the full joint list
                joint_idx = all_joint_names.index(joint_name)
                u, v = halton_sample[joint_idx * 2], halton_sample[joint_idx * 2 + 1]
                
                # Parameters
                min_range = min_range_factor * base_range
                effective_range = coverage_factor * base_range
                effective_min = base_min + (base_range - effective_range) / 2
                effective_max = effective_min + effective_range
                
                # SIMPLE MIDPOINT + RANGE SAMPLING
                range_size = min_range + v * (effective_range - min_range)
                
                # Calculate valid midpoint bounds
                midpoint_min = effective_min + range_size / 2
                midpoint_max = effective_max - range_size / 2
                
                # Sample midpoint within valid bounds
                if midpoint_max > midpoint_min:
                    midpoint = midpoint_min + u * (midpoint_max - midpoint_min)
                else:
                    midpoint = (effective_min + effective_max) / 2
                    range_size = effective_range
                
                # Convert to lower/upper limits
                lower_limit = midpoint - range_size / 2
                upper_limit = midpoint + range_size / 2
                
                joint_limits[joint_name] = (lower_limit, upper_limit)
            
            # For non-limited joints, use the full base range
            for joint_name in all_joint_names:
                if joint_name not in joint_limits:
                    joint_limits[joint_name] = base_limits[joint_name]
            
            joint_limit_sets.append({
                'id': sample_id,
                'joint_limits': joint_limits,
                'metadata': {
                    'n_limited_joints': n_limited_joints,
                    'limited_joints': limited_joints,
                    'unlimited_joints': [j for j in all_joint_names if j not in limited_joints],
                    'batch_number': n_total_joints - n_limited_joints + 1,
                    'sample_in_batch': batch_sample
                }
            })
            
            sample_id += 1
    
    print(f"\nGenerated {len(joint_limit_sets)} joint limit sets across {n_total_joints} batches")
    return joint_limit_sets


def generate_halton_joint_limits(n_samples, coverage_factor=0.8, min_range_factor=0.1, main_dofs_only=False):
    """
    Generate diverse joint limit sets using Halton sampling (original method).

    Args:
        n_samples: Number of joint limit sets to generate
        coverage_factor: What fraction of the base range to use for sampling (0.8 = 80%)
        min_range_factor: Minimum range as fraction of base range (0.1 = 10%)
        main_dofs_only: If True, generate limits only for 3 shoulder DOFs + elbow (default: False)

    Returns:
        List of joint limit dictionaries
    """
    base_limits = get_base_joint_limits()

    # Filter to only main DOFs if requested
    if main_dofs_only:
        main_dofs = ['shoulder_flexion_r', 'shoulder_abduction_r', 'shoulder_rotation_r', 'elbow_flexion_r']
        base_limits = {name: limits for name, limits in base_limits.items() if name in main_dofs}
        print(f"Main DOFs only mode: generating limits for {list(base_limits.keys())}")

    joint_names = list(base_limits.keys())
    n_joints = len(joint_names)

    print(f"Generating {n_samples} joint limit sets for {n_joints} joints...")
    print(f"Coverage factor: {coverage_factor:.1%}, Min range factor: {min_range_factor:.1%}")

    # Generate 20D Halton samples (10 joints × 2 limits each)
    halton_samples = generate_halton_samples(n_samples, n_joints * 2)

    joint_limit_sets = []

    for i in range(n_samples):
        joint_limits = {}

        for j, joint_name in enumerate(joint_names):
            base_min, base_max = base_limits[joint_name]
            base_range = base_max - base_min

            u, v = halton_samples[i, j * 2], halton_samples[i, j * 2 + 1]

            # Parameters
            min_range = min_range_factor * base_range
            effective_range = coverage_factor * base_range
            effective_min = base_min + (base_range - effective_range) / 2
            effective_max = effective_min + effective_range

            # SIMPLE MIDPOINT + RANGE SAMPLING (NO CLAMPING)
            # Sample range size first
            range_size = min_range + v * (effective_range - min_range)

            # Calculate valid midpoint bounds to avoid any clamping
            midpoint_min = effective_min + range_size / 2
            midpoint_max = effective_max - range_size / 2

            # Sample midpoint only within valid bounds (no clamping needed)
            if midpoint_max > midpoint_min:
                midpoint = midpoint_min + u * (midpoint_max - midpoint_min)
            else:
                # If range is too large, use center
                midpoint = (effective_min + effective_max) / 2
                range_size = effective_range

            # Convert to lower/upper limits (guaranteed to be within bounds)
            lower_limit = midpoint - range_size / 2
            upper_limit = midpoint + range_size / 2

            joint_limits[joint_name] = (lower_limit, upper_limit)

        joint_limit_sets.append({
            'id': i,
            'joint_limits': joint_limits
        })

    print(f"Generated {len(joint_limit_sets)} joint limit sets")
    return joint_limit_sets

def analyze_coverage_statistics(joint_limit_sets):
    """Analyze coverage statistics for the generated joint limit sets."""
    # Use actual joint names from the generated sets, not base limits
    if not joint_limit_sets:
        return {}

    joint_names = list(joint_limit_sets[0]['joint_limits'].keys())
    base_limits = get_base_joint_limits()

    stats = {}

    for joint_name in joint_names:
        # Extract all lower and upper limits for this joint
        lower_limits = [jls['joint_limits'][joint_name][0] for jls in joint_limit_sets]
        upper_limits = [jls['joint_limits'][joint_name][1] for jls in joint_limit_sets]
        ranges = [upper - lower for lower, upper in zip(lower_limits, upper_limits)]

        base_min, base_max = base_limits[joint_name]
        base_range = base_max - base_min

        # Convert to degrees for readability
        deg_factor = 180.0 / np.pi

        stats[joint_name] = {
            'base_range_deg': base_range * deg_factor,
            'lower_limits': {
                'min_deg': np.min(lower_limits) * deg_factor,
                'max_deg': np.max(lower_limits) * deg_factor,
                'mean_deg': np.mean(lower_limits) * deg_factor,
                'std_deg': np.std(lower_limits) * deg_factor
            },
            'upper_limits': {
                'min_deg': np.min(upper_limits) * deg_factor,
                'max_deg': np.max(upper_limits) * deg_factor,
                'mean_deg': np.mean(upper_limits) * deg_factor,
                'std_deg': np.std(upper_limits) * deg_factor
            },
            'ranges': {
                'min_deg': np.min(ranges) * deg_factor,
                'max_deg': np.max(ranges) * deg_factor,
                'mean_deg': np.mean(ranges) * deg_factor,
                'std_deg': np.std(ranges) * deg_factor,
                'min_as_fraction': np.min(ranges) / base_range,
                'max_as_fraction': np.max(ranges) / base_range
            }
        }

    return stats

def visualize_limit_coverage(joint_limit_sets, save_path=None):
    """Create comprehensive coverage visualization dashboard."""
    if not joint_limit_sets:
        print("No joint limit sets to visualize")
        return

    # Use actual joint names from the generated sets
    joint_names = list(joint_limit_sets[0]['joint_limits'].keys())
    base_limits = get_base_joint_limits()
    n_joints = len(joint_names)

    # Convert to degrees for visualization
    deg_factor = 180.0 / np.pi

    # Create figure with subplots (adjust layout based on number of joints)
    if n_joints <= 4:
        # For 4 joints or fewer, use 2x6 layout (2 rows, up to 6 columns per row)
        fig = plt.figure(figsize=(24, 12))
        cols = max(4, n_joints)  # At least 4 columns
        rows = 6  # 3 types of plots × 2 rows each
    else:
        # For more joints, use original 6x5 layout
        fig = plt.figure(figsize=(20, 20))
        cols = 5
        rows = 6

    # Panel A: Lower limit distributions
    for i, joint_name in enumerate(joint_names):
        if n_joints <= 4:
            ax = plt.subplot(rows, cols, i + 1)
        else:
            ax = plt.subplot(rows, cols, i + 1)

        lower_limits = [jls['joint_limits'][joint_name][0] * deg_factor for jls in joint_limit_sets]
        base_min, base_max = base_limits[joint_name]

        ax.hist(lower_limits, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(base_min * deg_factor, color='red', linestyle='--', label='Base min')
        ax.set_title(f'{joint_name}\nLower Limits')
        ax.set_xlabel('Degrees')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    # Panel B: Upper limit distributions
    for i, joint_name in enumerate(joint_names):
        if n_joints <= 4:
            ax = plt.subplot(rows, cols, i + cols + 1)
        else:
            ax = plt.subplot(rows, cols, i + 11)

        upper_limits = [jls['joint_limits'][joint_name][1] * deg_factor for jls in joint_limit_sets]
        base_min, base_max = base_limits[joint_name]

        ax.hist(upper_limits, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(base_max * deg_factor, color='red', linestyle='--', label='Base max')
        ax.set_title(f'{joint_name}\nUpper Limits')
        ax.set_xlabel('Degrees')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    # Panel C: Range distributions (upper - lower)
    for i, joint_name in enumerate(joint_names):
        if n_joints <= 4:
            ax = plt.subplot(rows, cols, i + 2*cols + 1)
        else:
            ax = plt.subplot(rows, cols, i + 21)

        ranges = [(jls['joint_limits'][joint_name][1] - jls['joint_limits'][joint_name][0]) * deg_factor
                  for jls in joint_limit_sets]
        base_min, base_max = base_limits[joint_name]
        base_range = (base_max - base_min) * deg_factor

        ax.hist(ranges, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(base_range, color='red', linestyle='--', label='Base range')
        ax.set_title(f'{joint_name}\nRanges (Upper - Lower)')
        ax.set_xlabel('Degrees')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Coverage visualization saved to {save_path}")

    plt.show()

def save_joint_limit_sets(joint_limit_sets, output_path="joint_limit_sets.json"):
    """Save joint limit sets to JSON file."""

    # Convert numpy types to native Python types for JSON serialization
    serializable_sets = []
    for jls in joint_limit_sets:
        serializable_jls = {
            'id': int(jls['id']),
            'joint_limits': {}
        }

        for joint_name, (lower, upper) in jls['joint_limits'].items():
            serializable_jls['joint_limits'][joint_name] = [float(lower), float(upper)]

        # Include metadata if present (for procedural sampling)
        if 'metadata' in jls:
            serializable_jls['metadata'] = jls['metadata']

        serializable_sets.append(serializable_jls)

    # Determine generation method
    generation_method = 'procedural_sampling' if 'metadata' in joint_limit_sets[0] else 'halton_20d_sampling'
    description = ('Joint limit sets generated using procedural sampling with decreasing number of limited joints' 
                   if generation_method == 'procedural_sampling'
                   else 'Joint limit sets generated using 20D Halton sampling for kinematic reachability analysis')

    # Add metadata
    output_data = {
        'metadata': {
            'n_samples': len(joint_limit_sets),
            'joint_names': list(joint_limit_sets[0]['joint_limits'].keys()),
            'generation_method': generation_method,
            'description': description
        },
        'joint_limit_sets': serializable_sets
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(joint_limit_sets)} joint limit sets to {output_path}")

def main():
    """Generate joint limit sets with coverage visualization."""
    parser = argparse.ArgumentParser(description='Generate diverse joint limit sets using Halton sampling')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of joint limit sets to generate (default: 1000)')
    parser.add_argument('--coverage-factor', type=float, default=1.0, help='Coverage factor for sampling range (default: 0.8)')
    parser.add_argument('--min-range-factor', type=float, default=0.05, help='Minimum range as fraction of base range (default: 0.1)')
    parser.add_argument('--visualize', action='store_true', help='Show coverage visualization')
    parser.add_argument('--output', default='joint_limit_sets.json', help='Output JSON file path')
    parser.add_argument('--save-plot', help='Save coverage plot to file')
    parser.add_argument('--main-dofs-only', action='store_true', help='Generate limits only for 3 shoulder DOFs + elbow')
    
    # Procedural sampling arguments
    parser.add_argument('--procedural', action='store_true', help='Use procedural sampling (decreasing number of limited joints)')
    parser.add_argument('--n-samples-per-batch', type=int, default=10, help='Number of samples per batch in procedural mode (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for procedural joint selection (default: 42)')

    args = parser.parse_args()

    print("=== Batch Joint Limit Sampler ===")

    # Generate joint limit sets using chosen method
    if args.procedural:
        print("Using procedural sampling approach")
        joint_limit_sets = generate_procedural_joint_limits(
            n_samples_per_batch=args.n_samples_per_batch,
            coverage_factor=args.coverage_factor,
            min_range_factor=args.min_range_factor,
            main_dofs_only=args.main_dofs_only,
            seed=args.seed
        )
    else:
        print("Using traditional Halton sampling approach")
        joint_limit_sets = generate_halton_joint_limits(
            n_samples=args.n_samples,
            coverage_factor=args.coverage_factor,
            min_range_factor=args.min_range_factor,
            main_dofs_only=args.main_dofs_only
        )

    # Analyze coverage statistics
    print("\nAnalyzing coverage statistics...")
    stats = analyze_coverage_statistics(joint_limit_sets)

    # Print summary
    print(f"\n=== COVERAGE SUMMARY ===")
    for joint_name, joint_stats in stats.items():
        range_stats = joint_stats['ranges']
        print(f"{joint_name:25s}: Range {range_stats['min_deg']:5.1f}° - {range_stats['max_deg']:5.1f}° "
              f"(mean: {range_stats['mean_deg']:5.1f}°, {range_stats['min_as_fraction']:.1%} - {range_stats['max_as_fraction']:.1%} of base)")

    # Save joint limit sets
    save_joint_limit_sets(joint_limit_sets, args.output)

    # Create visualization if requested
    if args.visualize:
        print("\nCreating coverage visualization...")
        visualize_limit_coverage(joint_limit_sets, save_path=args.save_plot)

    print(f"\nGenerated {len(joint_limit_sets)} joint limit sets successfully!")

if __name__ == "__main__":
    main()