#!/usr/bin/env python3
"""
Kinematic Reachability Visualizer

This script combines the kinematic reachability sampler with the SVM boundary
visualization pipeline to create rotating 3D GIFs of hand reachability.

Usage:
    python kinematic_reach_visualizer.py [options]

Options:
    --config: Configuration file (default: kinematic_reachability_config.json)
    --vis-mode: Visualization mode - endpoints, trajectories, reachable (default: endpoints)
    --save-gif: Save rotating GIF file path
    --gamma-percentile: SVM gamma percentile for gap-filling (default: 100.0)
    --resolution: Grid resolution for SVM visualization (default: 50)
    --visualize: Show interactive plot
"""

import numpy as np
import argparse
import os
import tempfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kinematic_reachability_sampler import sample_kinematic_reachability
from generate_svm_boundary import (
    train_svm_boundary, visualize_svm_boundary, create_rotating_gif
)

def save_hand_positions_as_text(hand_positions, filename):
    """Save hand positions in the format expected by generate_svm_boundary.py"""
    with open(filename, 'w') as f:
        f.write("# Hand position data from kinematic reachability sampling\n")
        f.write("# Format: x y z (meters)\n")
        for pos in hand_positions:
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

def set_equal_aspect_3d(ax, hand_positions):
    """Set equal aspect ratio for 3D plot with proper axis limits using rotated coordinates"""
    # Apply coordinate rotation for Z-up visualization
    x_points = hand_positions[:, 0]   # x stays x (right)
    y_points = -hand_positions[:, 2]  # -z becomes y (forward)
    z_points = hand_positions[:, 1]   # y becomes z (up)

    # Get rotated data ranges
    x_range = [x_points.min(), x_points.max()]
    y_range = [y_points.min(), y_points.max()]
    z_range = [z_points.min(), z_points.max()]

    # Calculate centers and max range
    x_center = (x_range[0] + x_range[1]) / 2
    y_center = (y_range[0] + y_range[1]) / 2
    z_center = (z_range[0] + z_range[1]) / 2

    # Find the maximum range among all axes
    max_range = max(
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0]
    ) / 2

    # Add some padding (10%)
    max_range *= 1.1

    # Set equal limits centered on data
    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(y_center - max_range, y_center + max_range)
    ax.set_zlim(z_center - max_range, z_center + max_range)

def visualize_kinematic_endpoints(hand_positions, config_file):
    """Create 3D visualization of kinematic endpoint positions with equal aspect"""
    print("Creating kinematic endpoints visualization...")

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Rotate coordinates for Z-up visualization (same as original generate_svm_boundary.py)
    x_points = hand_positions[:, 0]   # x stays x (right)
    y_points = -hand_positions[:, 2]  # -z becomes y (forward)
    z_points = hand_positions[:, 1]   # y becomes z (up)

    # Create RGB color gradient for endpoints
    x_norm = (x_points - x_points.min()) / (x_points.max() - x_points.min()) if x_points.max() > x_points.min() else np.zeros_like(x_points)
    y_norm = (y_points - y_points.min()) / (y_points.max() - y_points.min()) if y_points.max() > y_points.min() else np.zeros_like(y_points)
    z_norm = (z_points - z_points.min()) / (z_points.max() - z_points.min()) if z_points.max() > z_points.min() else np.zeros_like(z_points)

    colors = np.column_stack([x_norm, y_norm, z_norm])

    # Plot endpoint positions
    scatter = ax.scatter(x_points, y_points, z_points, c=colors,
                        s=20, alpha=0.6, label='Hand Positions')

    # Add shoulder position with coordinate rotation (same as original generate_svm_boundary.py)
    # Original shoulder position from model: [0.0, 1.25, 0.15]
    shoulder_x_orig, shoulder_y_orig, shoulder_z_orig = 0.0, 1.25, 0.15

    # Rotate shoulder coordinates for Z-up visualization
    shoulder_x_rot = shoulder_x_orig      # x stays x (right)
    shoulder_y_rot = -shoulder_z_orig     # -z becomes y (forward)
    shoulder_z_rot = shoulder_y_orig      # y becomes z (up)

    ax.scatter(shoulder_x_rot, shoulder_y_rot, shoulder_z_rot,
              c='red', s=200, alpha=0.9, label='Shoulder Center',
              edgecolors='darkred', linewidths=2)

    # Set labels and title (rotated coordinate system)
    ax.set_xlabel('X Position (m) [Right]')
    ax.set_ylabel('Y Position (m) [Forward]')
    ax.set_zlabel('Z Position (m) [Up]')

    ax.set_title(f'Kinematic Hand Reachability\n{len(hand_positions):,} sampled positions')

    # Set equal aspect ratio
    set_equal_aspect_3d(ax, hand_positions)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig, ax

def visualize_kinematic_trajectories(hand_positions, config_file):
    """Create 3D visualization showing kinematic samples as trajectory-style"""
    print("Creating kinematic trajectory-style visualization...")

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Rotate coordinates for Z-up visualization (same as original generate_svm_boundary.py)
    x_points = hand_positions[:, 0]   # x stays x (right)
    y_points = -hand_positions[:, 2]  # -z becomes y (forward)
    z_points = hand_positions[:, 1]   # y becomes z (up)

    # Plot trajectory points with temporal color progression
    scatter = ax.scatter(x_points, y_points, z_points, c=range(len(x_points)), cmap='viridis',
                        s=15, alpha=0.7, label='Sample Progression')

    # Add shoulder position with coordinate rotation
    # Original shoulder position from model: [0.0, 1.25, 0.15]
    shoulder_x_orig, shoulder_y_orig, shoulder_z_orig = 0.0, 1.25, 0.15

    # Rotate shoulder coordinates for Z-up visualization
    shoulder_x_rot = shoulder_x_orig      # x stays x (right)
    shoulder_y_rot = -shoulder_z_orig     # -z becomes y (forward)
    shoulder_z_rot = shoulder_y_orig      # y becomes z (up)

    ax.scatter(shoulder_x_rot, shoulder_y_rot, shoulder_z_rot,
              c='red', s=200, alpha=0.9, label='Shoulder Center',
              edgecolors='darkred', linewidths=2)

    # Add colorbar
    try:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Sample Order', rotation=270, labelpad=15)
    except:
        pass

    # Set labels and title (rotated coordinate system)
    ax.set_xlabel('X Position (m) [Right]')
    ax.set_ylabel('Y Position (m) [Forward]')
    ax.set_zlabel('Z Position (m) [Up]')

    ax.set_title(f'Kinematic Reachability Sampling\n{len(hand_positions):,} samples')

    # Set equal aspect ratio
    set_equal_aspect_3d(ax, hand_positions)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig, ax

def create_kinematic_gif(hand_positions, config_file, gif_path, vis_mode='endpoints'):
    """Create a rotating GIF of kinematic reachability visualization"""
    print(f"Creating kinematic rotating GIF: {gif_path}")

    try:
        import matplotlib.animation as animation
    except ImportError:
        print("Error: Need matplotlib animation for GIF creation")
        return False

    # Create the appropriate visualization
    if vis_mode == 'endpoints':
        fig, ax = visualize_kinematic_endpoints(hand_positions, config_file)
    else:  # trajectories
        fig, ax = visualize_kinematic_trajectories(hand_positions, config_file)

    # Remove legend for cleaner GIF
    ax.legend().set_visible(False)

    # Function to update the view angle
    def update_view(frame):
        ax.view_init(elev=20, azim=frame * 3)  # Rotate 3 degrees per frame
        return []

    # Create animation (120 frames = 360 degrees rotation)
    frames = 120
    print(f"  Generating {frames} frames...")

    anim = animation.FuncAnimation(fig, update_view, frames=frames, interval=100, blit=False)

    # Save as GIF
    print(f"  Saving GIF to {gif_path}...")
    try:
        anim.save(gif_path, writer='pillow', fps=10, dpi=80)
        print(f"✓ GIF saved successfully: {gif_path}")
        return True
    except Exception as e:
        print(f"Error saving GIF: {e}")
        return False
    finally:
        plt.close(fig)

def run_kinematic_reach_visualization(config_file="kinematic_reachability_config.json",
                                    vis_mode="endpoints",
                                    save_gif=None,
                                    gamma_percentile=100.0,
                                    resolution=50,
                                    show_interactive=False):
    """Run complete kinematic reachability sampling and visualization pipeline"""

    print("=" * 60)
    print("KINEMATIC REACHABILITY VISUALIZATION PIPELINE")
    print("=" * 60)

    # Step 1: Sample kinematic reachability
    print("Step 1: Sampling kinematic reachability...")
    try:
        hand_positions, joint_configs, dof_names = sample_kinematic_reachability(config_file)
        print(f"✓ Generated {len(hand_positions)} hand positions")
    except Exception as e:
        print(f"✗ Error in kinematic sampling: {e}")
        return False

    # Step 2: Save hand positions to temporary file
    print("Step 2: Preparing data for visualization...")
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, "kinematic_hand_positions.txt")
    save_hand_positions_as_text(hand_positions, temp_file)
    print(f"✓ Hand positions saved to: {temp_file}")

    # Step 3: Generate visualization
    print(f"Step 3: Creating {vis_mode} visualization...")

    try:
        if vis_mode == "endpoints":
            # Custom kinematic endpoint visualization with equal aspect
            fig, ax = visualize_kinematic_endpoints(hand_positions, config_file)
            print("✓ Created kinematic endpoints visualization")

        elif vis_mode == "trajectories":
            # Custom kinematic trajectory-style visualization with equal aspect
            fig, ax = visualize_kinematic_trajectories(hand_positions, config_file)
            print("✓ Created kinematic trajectory-style visualization")

        elif vis_mode == "reachable":
            # Full SVM boundary visualization (uses original function)
            print("  Training SVM boundary model...")
            svm_model, scaler, gamma, nu = train_svm_boundary(
                hand_positions, gamma=None, nu=0.05, gamma_percentile=gamma_percentile)

            fig, ax = visualize_svm_boundary(
                hand_positions, svm_model, scaler, config_file, 1.0,
                gamma, nu, resolution, gamma_percentile)

            # Apply equal aspect to SVM visualization too
            set_equal_aspect_3d(ax, hand_positions)
            print("✓ Created SVM reachability visualization with equal aspect")

        else:
            raise ValueError(f"Unknown vis_mode: {vis_mode}")

    except Exception as e:
        print(f"✗ Error creating visualization: {e}")
        return False

    # Step 4: Save GIF if requested
    if save_gif:
        print(f"Step 4: Creating rotating GIF: {save_gif}")
        try:
            if vis_mode == "reachable":
                success = create_rotating_gif(
                    hand_positions, svm_model, scaler, config_file, 1.0,
                    gamma, nu, resolution, gamma_percentile, save_gif, vis_mode)
            else:
                # For endpoints/trajectories, use custom kinematic GIF function
                success = create_kinematic_gif(hand_positions, config_file, save_gif, vis_mode)

            if success:
                print(f"✓ GIF saved successfully: {save_gif}")
            else:
                print("✗ Failed to create GIF")
        except Exception as e:
            print(f"✗ Error creating GIF: {e}")

    # Step 5: Show interactive visualization if requested
    if show_interactive:
        print("Step 5: Showing interactive visualization...")
        try:
            import matplotlib.pyplot as plt
            plt.show()
            print("✓ Interactive visualization displayed")
        except Exception as e:
            print(f"✗ Error showing interactive plot: {e}")
    else:
        # Clean up the figure
        import matplotlib.pyplot as plt
        plt.close(fig)

    # Clean up temporary file
    try:
        os.remove(temp_file)
        print(f"✓ Cleaned up temporary file")
    except:
        pass

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Kinematic Reachability Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--config', type=str,
                       default='kinematic_reachability_config.json',
                       help='Configuration file path')
    parser.add_argument('--vis-mode', type=str, default='endpoints',
                       choices=['endpoints', 'trajectories', 'reachable'],
                       help='Visualization mode')
    parser.add_argument('--save-gif', type=str,
                       help='Save rotating GIF to this file path')
    parser.add_argument('--gamma-percentile', type=float, default=100.0,
                       help='SVM gamma percentile (0-100, lower = more gap-filling)')
    parser.add_argument('--resolution', type=int, default=50,
                       help='Grid resolution for SVM visualization')
    parser.add_argument('--visualize', action='store_true',
                       help='Show interactive 3D visualization')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return 1

    if args.gamma_percentile < 0 or args.gamma_percentile > 100:
        print("Error: gamma-percentile must be between 0 and 100")
        return 1

    # Run the visualization pipeline
    success = run_kinematic_reach_visualization(
        config_file=args.config,
        vis_mode=args.vis_mode,
        save_gif=args.save_gif,
        gamma_percentile=args.gamma_percentile,
        resolution=args.resolution,
        show_interactive=args.visualize
    )

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())