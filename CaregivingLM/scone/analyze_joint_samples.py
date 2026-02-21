#!/usr/bin/env python3
"""
Analyze Joint Configuration Samples

This script loads the joint configuration data from kinematic sampling
and searches for poses similar to specified target configurations.
"""

import numpy as np
import json
import os
import argparse

def load_joint_data(data_dir="kinematic_reachability", filename_prefix="hand_reach"):
    """Load joint configurations and metadata"""
    joint_file = f"{data_dir}/{filename_prefix}_joint_configs.npy"
    hand_file = f"{data_dir}/{filename_prefix}_hand_positions.npy"
    metadata_file = f"{data_dir}/{filename_prefix}_metadata.json"

    if not os.path.exists(joint_file):
        print(f"Error: Joint configuration file not found: {joint_file}")
        return None, None, None

    joint_configs = np.load(joint_file)
    hand_positions = np.load(hand_file) if os.path.exists(hand_file) else None

    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    return joint_configs, hand_positions, metadata

def find_similar_poses(joint_configs, target_pose, dof_names, tolerance=5.0):
    """Find poses similar to target within tolerance (degrees)"""
    target_array = np.array([target_pose[name] for name in dof_names])

    # Calculate distances to target pose
    distances = np.linalg.norm(joint_configs - target_array, axis=1)

    # Find poses within tolerance
    similar_indices = np.where(distances <= tolerance)[0]
    similar_poses = joint_configs[similar_indices]
    similar_distances = distances[similar_indices]

    return similar_indices, similar_poses, similar_distances

def analyze_joint_coverage(joint_configs, joint_limits, dof_names):
    """Analyze how well the joint space is covered"""
    print("Joint Space Coverage Analysis:")
    print("=" * 50)

    for i, dof_name in enumerate(dof_names):
        min_limit, max_limit = joint_limits[dof_name]
        sampled_min = joint_configs[:, i].min()
        sampled_max = joint_configs[:, i].max()
        sampled_range = sampled_max - sampled_min
        expected_range = max_limit - min_limit

        print(f"\n{dof_name}:")
        print(f"  Expected range: [{min_limit:.1f}, {max_limit:.1f}] = {expected_range:.1f}°")
        print(f"  Sampled range:  [{sampled_min:.1f}, {sampled_max:.1f}] = {sampled_range:.1f}°")
        print(f"  Coverage: {100 * sampled_range / expected_range:.1f}%")

        # Check if target value is within sampled range
        target_value = -45 if dof_name == "shoulder_flexion_r" else \
                      -11.3 if dof_name == "shoulder_abduction_r" else \
                      60.0 if dof_name == "shoulder_rotation_r" else \
                      88.4 if dof_name == "elbow_flexion_r" else None

        if target_value is not None:
            in_sampled = sampled_min <= target_value <= sampled_max
            in_limits = min_limit <= target_value <= max_limit
            print(f"  Target value {target_value:.1f}°: ", end="")
            if in_limits:
                if in_sampled:
                    print("✓ Within sampled range")
                else:
                    print("✗ Outside sampled range (but within limits)")
            else:
                print("✗ Outside joint limits")

def main():
    parser = argparse.ArgumentParser(description='Analyze Joint Configuration Samples')
    parser.add_argument('--data-dir', type=str, default='kinematic_reachability',
                       help='Directory containing the sample data')
    parser.add_argument('--prefix', type=str, default='hand_reach',
                       help='Filename prefix for the data files')
    parser.add_argument('--tolerance', type=float, default=5.0,
                       help='Tolerance in degrees for finding similar poses')

    args = parser.parse_args()

    # Load joint configuration data
    print("Loading joint configuration data...")
    joint_configs, hand_positions, metadata = load_joint_data(args.data_dir, args.prefix)

    if joint_configs is None:
        return 1

    print(f"Loaded {len(joint_configs)} joint configurations")

    # Get DOF names and joint limits from metadata
    dof_names = metadata.get('dof_names', ['shoulder_flexion_r', 'shoulder_abduction_r', 'shoulder_rotation_r', 'elbow_flexion_r'])
    joint_limits = metadata.get('joint_limits', {})

    print(f"DOFs: {dof_names}")

    # Define target pose (back space position)
    target_pose = {
        'shoulder_flexion_r': -45.0,
        'shoulder_abduction_r': -11.3,
        'shoulder_rotation_r': 60.0,
        'elbow_flexion_r': 88.4
    }

    print(f"\nTarget pose (back space):")
    for dof, angle in target_pose.items():
        print(f"  {dof}: {angle:.1f}°")

    # Analyze joint space coverage
    if joint_limits:
        analyze_joint_coverage(joint_configs, joint_limits, dof_names)

    # Find similar poses
    print(f"\nSearching for poses within {args.tolerance:.1f}° tolerance...")
    similar_indices, similar_poses, similar_distances = find_similar_poses(
        joint_configs, target_pose, dof_names, args.tolerance)

    print(f"\nFound {len(similar_indices)} similar poses:")

    if len(similar_indices) == 0:
        print("No poses found within tolerance!")

        # Find closest pose
        target_array = np.array([target_pose[name] for name in dof_names])
        all_distances = np.linalg.norm(joint_configs - target_array, axis=1)
        closest_idx = np.argmin(all_distances)
        closest_pose = joint_configs[closest_idx]
        closest_distance = all_distances[closest_idx]

        print(f"\nClosest pose (distance: {closest_distance:.2f}°):")
        for i, dof in enumerate(dof_names):
            target_val = target_pose[dof]
            closest_val = closest_pose[i]
            diff = closest_val - target_val
            print(f"  {dof}: {closest_val:.1f}° (target: {target_val:.1f}°, diff: {diff:+.1f}°)")

        if hand_positions is not None:
            closest_hand_pos = hand_positions[closest_idx]
            print(f"  Hand position: ({closest_hand_pos[0]:.3f}, {closest_hand_pos[1]:.3f}, {closest_hand_pos[2]:.3f})")

    else:
        # Show the best matches
        sorted_indices = np.argsort(similar_distances)
        print(f"\nTop 5 closest matches:")

        for rank, idx in enumerate(sorted_indices[:5]):
            sample_idx = similar_indices[idx]
            pose = similar_poses[idx]
            distance = similar_distances[idx]

            print(f"\nRank {rank + 1} (sample #{sample_idx}, distance: {distance:.2f}°):")
            for i, dof in enumerate(dof_names):
                target_val = target_pose[dof]
                sample_val = pose[i]
                diff = sample_val - target_val
                print(f"  {dof}: {sample_val:.1f}° (target: {target_val:.1f}°, diff: {diff:+.1f}°)")

            if hand_positions is not None:
                hand_pos = hand_positions[sample_idx]
                print(f"  Hand position: ({hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f})")

    # Statistics about the sampled space
    print(f"\nSampling Statistics:")
    print(f"  Total samples: {len(joint_configs)}")
    print(f"  Halton samples: {metadata.get('n_halton_samples', 'N/A')}")
    print(f"  Boundary samples: {metadata.get('n_boundary_samples', 'N/A')}")

    return 0

if __name__ == "__main__":
    exit(main())