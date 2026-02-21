#!/usr/bin/env python3
"""
3D Interactive Visualization of Hand Trajectories

This script creates an interactive 3D plot of hand trajectory data from SCONE simulation results.
You can rotate, zoom, and pan around the 3D visualization.

Usage:
    python visualize_trajectories.py trajectory_file.txt [--sample-ratio 0.1]

Arguments:
    trajectory_file: Path to the hand trajectory file (e.g., condition_1_hand_trajectories.txt)
    --sample-ratio: Fraction of points to randomly sample (0.0-1.0, default: 1.0 = all points)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import random


def load_trajectory_data(file_path):
    """Load trajectory data from file"""
    trajectory_points = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            trajectory_points.append((x, y, z))
                        except ValueError:
                            continue

    except Exception as e:
        print(f"Error reading trajectory file: {e}")
        return None

    return np.array(trajectory_points)


def sample_points(points, sample_ratio):
    """Randomly sample a fraction of the points"""
    if sample_ratio >= 1.0:
        return points

    n_total = len(points)
    n_sample = int(n_total * sample_ratio)

    if n_sample == 0:
        n_sample = 1

    # Random sampling without replacement
    indices = random.sample(range(n_total), min(n_sample, n_total))
    return points[indices]


def create_3d_visualization(trajectory_points, file_path, sample_ratio):
    """Create interactive 3D visualization"""

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Rotate around X-axis by 90 degrees to convert Y-up to Z-up
    # Original: x=right, y=up, z=forward
    # Rotated: x=right, y=forward, z=up
    x = trajectory_points[:, 0]   # x stays x (right)
    y = -trajectory_points[:, 2]  # -z becomes y (forward)
    z = trajectory_points[:, 1]   # y becomes z (up)

    # Create the 3D scatter plot with larger points
    scatter = ax.scatter(x, y, z, c=range(len(x)), cmap='viridis',
                        s=10, alpha=0.7, edgecolors='none')

    # Add colorbar to show temporal progression
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Trajectory Progression', rotation=270, labelpad=15)

    # Set labels and title (rotated coordinates)
    ax.set_xlabel('X Position (m) [Right]')
    ax.set_ylabel('Y Position (m) [Forward]')
    ax.set_zlabel('Z Position (m) [Up]')

    filename = os.path.basename(file_path)
    sample_info = f" (sampled {sample_ratio:.1%})" if sample_ratio < 1.0 else ""
    ax.set_title(f'Hand Trajectory Visualization\n{filename}{sample_info}\n{len(trajectory_points):,} points')

    # Set equal aspect ratio for better visualization
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add grid for better spatial reference
    ax.grid(True, alpha=0.3)

    # Add statistics text (rotated coordinates)
    stats_text = f"Statistics:\n"
    stats_text += f"X [Right]: [{x.min():.3f}, {x.max():.3f}]\n"
    stats_text += f"Y [Forward]: [{y.min():.3f}, {y.max():.3f}]\n"
    stats_text += f"Z [Up]: [{z.min():.3f}, {z.max():.3f}]\n"
    stats_text += f"Total points: {len(trajectory_points):,}"

    # Add text box with statistics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='top', bbox=props)

    # Improve the viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    return fig, ax


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='3D Interactive Visualization of Hand Trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_trajectories.py compiled_results_coverage_200/condition_1_hand_trajectories.txt
    python visualize_trajectories.py condition_0_hand_trajectories.txt --sample-ratio 0.1
    python visualize_trajectories.py trajectory.txt --sample-ratio 0.05
        """
    )

    parser.add_argument('trajectory_file',
                       help='Path to the hand trajectory file (e.g., condition_1_hand_trajectories.txt)')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                       help='Fraction of points to randomly sample (0.0-1.0, default: 1.0 = all points)')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file not found: {args.trajectory_file}")
        return 1

    if args.sample_ratio <= 0 or args.sample_ratio > 1.0:
        print(f"Error: Sample ratio must be between 0.0 and 1.0, got: {args.sample_ratio}")
        return 1

    print(f"Loading trajectory data from: {args.trajectory_file}")

    # Load trajectory data
    trajectory_points = load_trajectory_data(args.trajectory_file)

    if trajectory_points is None or len(trajectory_points) == 0:
        print("Error: No valid trajectory data found in the file")
        return 1

    print(f"Loaded {len(trajectory_points):,} trajectory points")

    # Sample points if requested
    if args.sample_ratio < 1.0:
        print(f"Sampling {args.sample_ratio:.1%} of points...")
        trajectory_points = sample_points(trajectory_points, args.sample_ratio)
        print(f"Using {len(trajectory_points):,} sampled points for visualization")

    # Create visualization
    print("Creating 3D visualization...")
    fig, ax = create_3d_visualization(trajectory_points, args.trajectory_file, args.sample_ratio)

    print("\nVisualization ready!")
    print("Instructions:")
    print("- Left mouse button: Rotate the view")
    print("- Right mouse button: Zoom in/out")
    print("- Middle mouse button: Pan the view")
    print("- Close the window to exit")

    # Show the interactive plot
    plt.show()

    return 0


if __name__ == "__main__":
    exit(main())