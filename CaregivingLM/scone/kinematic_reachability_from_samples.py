#!/usr/bin/env python3
"""
Extract kinematic reachability from existing joint coverage samples.
This script reads joint angles from .zml files and filters against joint limits.
"""

import numpy as np
import glob
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def read_joint_angles_from_zml(filepath):
    """Extract joint angles from a .zml file."""
    joint_angles = {}

    with open(filepath, 'r') as f:
        in_values_section = False

        for line in f:
            line = line.strip()

            if line == "values {":
                in_values_section = True
                continue
            elif line == "}":
                if in_values_section:
                    break
                continue

            if in_values_section and "=" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    joint_name = parts[0].strip()
                    joint_value = float(parts[1].strip())
                    joint_angles[joint_name] = joint_value

    return joint_angles

def get_joint_limits():
    """Get the same joint limits used in joint_coverage_sampling.py."""
    deg_to_rad = np.pi / 180.0

    joint_limits = {
        'clavicle_protraction_r': (-30*deg_to_rad, 10*deg_to_rad),
        'clavicle_elevation_r': (-15*deg_to_rad, 45*deg_to_rad),
        'clavicle_rotation_r': (-15*deg_to_rad, 60*deg_to_rad),
        'scapula_abduction_r': (-35*deg_to_rad, 45*deg_to_rad),
        'scapula_elevation_r': (-45*deg_to_rad, 45*deg_to_rad),
        'scapula_winging_r': (-45*deg_to_rad, 45*deg_to_rad),
        'shoulder_flexion_r': (-45*deg_to_rad, 150*deg_to_rad),
        'shoulder_abduction_r': (-60*deg_to_rad, 120*deg_to_rad),
        'shoulder_rotation_r': (-60*deg_to_rad, 60*deg_to_rad),
        'elbow_flexion_r': (0*deg_to_rad, 130*deg_to_rad),
        'forearm_pronation_r': (-90*deg_to_rad, 90*deg_to_rad),
        'wrist_flexion_r': (-70*deg_to_rad, 70*deg_to_rad),
        'radial_deviation_r': (-30*deg_to_rad, 15*deg_to_rad)
    }

    return joint_limits

def check_joint_limits_compliance(joint_angles, joint_limits):
    """Check if joint angles are within specified limits."""
    violations = {}
    within_limits = True

    for joint_name, angle in joint_angles.items():
        if joint_name in joint_limits:
            min_limit, max_limit = joint_limits[joint_name]

            if angle < min_limit or angle > max_limit:
                violations[joint_name] = {
                    'angle': angle,
                    'limits': (min_limit, max_limit),
                    'violation': angle - max_limit if angle > max_limit else min_limit - angle
                }
                within_limits = False

    return within_limits, violations

def analyze_samples(samples_dir="halton_sample_joint", hand_positions_file="halton_sample_joint/halton_sample_hand_positions.txt"):
    """Analyze existing samples for kinematic reachability."""

    # Get joint limits
    joint_limits = get_joint_limits()

    # Find all .zml files
    zml_files = sorted(glob.glob(f"{samples_dir}/halton_sample_*.zml"))

    if not zml_files:
        print(f"No .zml files found in {samples_dir}")
        return

    print(f"Found {len(zml_files)} sample files")

    # Load hand positions
    try:
        hand_positions = np.loadtxt(hand_positions_file)
        print(f"Loaded {len(hand_positions)} hand positions")
    except Exception as e:
        print(f"Error loading hand positions: {e}")
        return

    # Analyze each sample
    reachable_samples = []
    unreachable_samples = []
    reachable_positions = []
    unreachable_positions = []

    all_joint_angles = []
    violation_stats = {joint: {'count': 0, 'total_violation': 0.0} for joint in joint_limits}

    for i, zml_file in enumerate(zml_files):
        # Extract sample number from filename
        filename = Path(zml_file).name
        sample_num = int(filename.split('_')[-1].split('.')[0])

        # Read joint angles
        joint_angles = read_joint_angles_from_zml(zml_file)
        all_joint_angles.append(joint_angles)

        # Check compliance
        within_limits, violations = check_joint_limits_compliance(joint_angles, joint_limits)

        if within_limits:
            reachable_samples.append(sample_num)
            if sample_num < len(hand_positions):
                reachable_positions.append(hand_positions[sample_num])
        else:
            unreachable_samples.append(sample_num)
            if sample_num < len(hand_positions):
                unreachable_positions.append(hand_positions[sample_num])

            # Track violation statistics
            for joint, violation_data in violations.items():
                violation_stats[joint]['count'] += 1
                violation_stats[joint]['total_violation'] += abs(violation_data['violation'])

    # Convert to numpy arrays
    reachable_positions = np.array(reachable_positions) if reachable_positions else np.array([])
    unreachable_positions = np.array(unreachable_positions) if unreachable_positions else np.array([])

    # Print summary
    total_samples = len(zml_files)
    reachable_count = len(reachable_samples)
    unreachable_count = len(unreachable_samples)

    print(f"\n=== KINEMATIC REACHABILITY ANALYSIS ===")
    print(f"Total samples analyzed: {total_samples}")
    print(f"Reachable (within joint limits): {reachable_count} ({reachable_count/total_samples*100:.1f}%)")
    print(f"Unreachable (violate joint limits): {unreachable_count} ({unreachable_count/total_samples*100:.1f}%)")

    # Violation statistics
    print(f"\n=== JOINT LIMIT VIOLATION ANALYSIS ===")
    for joint, stats in violation_stats.items():
        if stats['count'] > 0:
            avg_violation = stats['total_violation'] / stats['count']
            violation_rate = stats['count'] / total_samples * 100

            min_limit, max_limit = joint_limits[joint]
            limit_range = max_limit - min_limit

            print(f"{joint:25s}: {stats['count']:3d} violations ({violation_rate:4.1f}%) "
                  f"- avg violation: {avg_violation:.3f} rad ({avg_violation*180/np.pi:.1f}°) "
                  f"- range: [{min_limit:.3f}, {max_limit:.3f}] ({limit_range*180/np.pi:.1f}° span)")

    # Save results
    output_dir = Path("kinematic_reachability")
    output_dir.mkdir(exist_ok=True)

    # Save reachable positions
    if len(reachable_positions) > 0:
        np.savetxt(output_dir / "reachable_hand_positions.txt",
                   reachable_positions, header="x y z", fmt="%.6f")
        print(f"\nSaved {len(reachable_positions)} reachable positions to {output_dir}/reachable_hand_positions.txt")

        # Position statistics for reachable samples
        print(f"\nReachable hand position range:")
        print(f"  X: [{reachable_positions[:,0].min():.3f}, {reachable_positions[:,0].max():.3f}] (span: {reachable_positions[:,0].max()-reachable_positions[:,0].min():.3f})")
        print(f"  Y: [{reachable_positions[:,1].min():.3f}, {reachable_positions[:,1].max():.3f}] (span: {reachable_positions[:,1].max()-reachable_positions[:,1].min():.3f})")
        print(f"  Z: [{reachable_positions[:,2].min():.3f}, {reachable_positions[:,2].max():.3f}] (span: {reachable_positions[:,2].max()-reachable_positions[:,2].min():.3f})")

    # Save unreachable positions
    if len(unreachable_positions) > 0:
        np.savetxt(output_dir / "unreachable_hand_positions.txt",
                   unreachable_positions, header="x y z", fmt="%.6f")
        print(f"Saved {len(unreachable_positions)} unreachable positions to {output_dir}/unreachable_hand_positions.txt")

    # Save sample indices
    with open(output_dir / "reachability_classification.json", 'w') as f:
        json.dump({
            'total_samples': total_samples,
            'reachable_count': reachable_count,
            'unreachable_count': unreachable_count,
            'reachable_samples': reachable_samples,
            'unreachable_samples': unreachable_samples,
            'joint_limits': {k: list(v) for k, v in joint_limits.items()},  # Convert tuples to lists for JSON
            'violation_statistics': violation_stats
        }, f, indent=2)

    print(f"Saved detailed analysis to {output_dir}/reachability_classification.json")

    return reachable_positions, unreachable_positions, violation_stats

def set_equal_aspect_3d(ax, x_data, y_data, z_data):
    """Set equal aspect ratio for 3D plot."""
    # Get the range of each axis
    x_range = np.max(x_data) - np.min(x_data)
    y_range = np.max(y_data) - np.min(y_data)
    z_range = np.max(z_data) - np.min(z_data)

    # Get the maximum range
    max_range = max(x_range, y_range, z_range)

    # Get the center of each axis
    x_center = (np.max(x_data) + np.min(x_data)) / 2
    y_center = (np.max(y_data) + np.min(y_data)) / 2
    z_center = (np.max(z_data) + np.min(z_data)) / 2

    # Set the limits
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)

def create_reachability_visualization(reachable_positions, unreachable_positions, mode="gif", output_path="kinematic_reachability/reachability_visualization.gif"):
    """Create 3D visualization with GIF output or interactive mode."""

    # Apply SCONE coordinate transformation: Y-up to Z-up
    if len(reachable_positions) > 0:
        reach_x = reachable_positions[:, 0]   # X stays X (right)
        reach_y = -reachable_positions[:, 2]  # -Z becomes Y (forward)
        reach_z = reachable_positions[:, 1]   # Y becomes Z (up)
    else:
        reach_x = reach_y = reach_z = np.array([])

    if len(unreachable_positions) > 0:
        unreach_x = unreachable_positions[:, 0]   # X stays X (right)
        unreach_y = -unreachable_positions[:, 2]  # -Z becomes Y (forward)
        unreach_z = unreachable_positions[:, 1]   # Y becomes Z (up)
    else:
        unreach_x = unreach_y = unreach_z = np.array([])

    # Combine all points for aspect ratio calculation
    all_x = np.concatenate([reach_x, unreach_x]) if len(reach_x) > 0 or len(unreach_x) > 0 else np.array([0])
    all_y = np.concatenate([reach_y, unreach_y]) if len(reach_y) > 0 or len(unreach_y) > 0 else np.array([0])
    all_z = np.concatenate([reach_z, unreach_z]) if len(reach_z) > 0 or len(unreach_z) > 0 else np.array([0])

    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot reachable positions (green)
    if len(reach_x) > 0:
        ax.scatter(reach_x, reach_y, reach_z, c='green', alpha=0.6, s=15, label=f'Reachable ({len(reach_x)})')

    # Plot unreachable positions (red)
    if len(unreach_x) > 0:
        ax.scatter(unreach_x, unreach_y, unreach_z, c='red', alpha=0.6, s=15, label=f'Unreachable ({len(unreach_x)})')

    # Add shoulder reference point using the same position as kinematic_reach_visualizer.py
    # Original shoulder position from SCONE model: [0.0, 1.25, 0.15]
    shoulder_x_orig, shoulder_y_orig, shoulder_z_orig = 0.0, 1.25, 0.15

    # Apply the same coordinate transformation: Y-up to Z-up
    shoulder_x_rot = shoulder_x_orig      # x stays x (right)
    shoulder_y_rot = -shoulder_z_orig     # -z becomes y (forward)
    shoulder_z_rot = shoulder_y_orig      # y becomes z (up)

    ax.scatter([shoulder_x_rot], [shoulder_y_rot], [shoulder_z_rot],
               c='blue', s=100, marker='o', label='Shoulder Center')

    # Set equal aspect ratio
    set_equal_aspect_3d(ax, all_x, all_y, all_z)

    # Labels and title
    ax.set_xlabel('X (Right) [m]')
    ax.set_ylabel('Y (Forward) [m]')
    ax.set_zlabel('Z (Up) [m]')
    ax.set_title('Kinematic Reachability Analysis\n(Joint Limit Filtering)')
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    if mode == "interactive":
        print("Opening interactive visualization...")
        print("Use mouse to rotate, zoom, and pan. Close window when done.")
        plt.show()
    elif mode == "gif":
        # Animation function
        def animate(frame):
            ax.view_init(elev=20, azim=frame * 2)  # Rotate 2 degrees per frame
            return []

        # Create animation (180 frames for 360 degrees)
        print("Creating rotating GIF visualization...")
        anim = animation.FuncAnimation(fig, animate, frames=180, interval=50, blit=False)

        # Save as GIF
        anim.save(output_path, writer='pillow', fps=20, dpi=100)
        plt.close()
        print(f"GIF visualization saved to {output_path}")
    else:
        raise ValueError(f"Unknown visualization mode: {mode}. Use 'gif' or 'interactive'.")

def main():
    """Run kinematic reachability analysis with visualization."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze kinematic reachability from existing joint coverage samples')
    parser.add_argument('--mode', choices=['gif', 'interactive', 'both'], default='gif',
                       help='Visualization mode: gif (rotating animation), interactive (mouse controls), or both (default: gif)')
    parser.add_argument('--samples-dir', default='init/halton_sample_joint',
                       help='Directory containing sample .zml files (default: init/halton_sample_joint)')
    parser.add_argument('--hand-positions-file', default='init/halton_sample_joint/halton_sample_hand_positions.txt',
                       help='File containing hand positions (default: init/halton_sample_joint/halton_sample_hand_positions.txt)')
    parser.add_argument('--output-gif', default='kinematic_reachability/reachability_visualization.gif',
                       help='Output path for GIF file (default: kinematic_reachability/reachability_visualization.gif)')

    args = parser.parse_args()

    print("Analyzing kinematic reachability from existing joint coverage samples...")

    reachable_positions, unreachable_positions, violation_stats = analyze_samples(
        samples_dir=args.samples_dir,
        hand_positions_file=args.hand_positions_file
    )

    # Create visualization(s)
    if len(reachable_positions) > 0 or len(unreachable_positions) > 0:
        if args.mode in ['gif', 'both']:
            create_reachability_visualization(reachable_positions, unreachable_positions,
                                            mode="gif", output_path=args.output_gif)

        if args.mode in ['interactive', 'both']:
            create_reachability_visualization(reachable_positions, unreachable_positions,
                                            mode="interactive")
    else:
        print("No position data available for visualization")

    print(f"\nAnalysis complete! Results saved to kinematic_reachability/ directory")

if __name__ == "__main__":
    main()