#!/usr/bin/env python3
"""
Alpha Shape Hull Generation for Reachability Analysis

This script generates alpha shape hulls from hand trajectory data to approximate
the reachable workspace. Alpha shapes are a generalization of convex hulls that
can represent non-convex and complex boundaries.

Usage:
    python generate_alpha_hull.py trajectory_file.txt [options]

Arguments:
    trajectory_file: Path to the hand trajectory file
    --alpha: Alpha parameter (default: auto-compute optimal)
    --sample-ratio: Fraction of points to use (default: 1.0)
    --visualize: Show interactive 3D visualization
    --save-mesh: Save hull as mesh file (.ply format)

Dependencies:
    pip install alphashape numpy matplotlib scipy trimesh
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import os
import random

try:
    import alphashape
except ImportError:
    print("Error: alphashape package not found. Install with: pip install alphashape")
    exit(1)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Mesh saving will be disabled.")


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


def estimate_alpha_parameter(points):
    """Estimate a reasonable alpha parameter based on point cloud statistics"""
    from scipy.spatial.distance import pdist

    # Sample a subset of points to estimate distances (for performance)
    n_sample = min(1000, len(points))
    sample_indices = np.random.choice(len(points), n_sample, replace=False)
    sample_points = points[sample_indices]

    # Compute pairwise distances
    distances = pdist(sample_points)

    # Use percentiles of distances as alpha estimates
    alpha_estimates = [
        np.percentile(distances, 10),   # Tight fit
        np.percentile(distances, 25),   # Medium-tight fit
        np.percentile(distances, 50),   # Medium fit
        np.percentile(distances, 75),   # Loose fit
    ]

    return alpha_estimates


def compute_alpha_shape(points, alpha=None):
    """Compute 3D alpha shape from points"""
    print(f"Computing alpha shape from {len(points):,} points...")

    if alpha is None:
        # Estimate reasonable alpha parameters
        print("Estimating alpha parameter based on point cloud...")
        alpha_estimates = estimate_alpha_parameter(points)

        # Try different alpha values, starting from tighter fits
        for i, test_alpha in enumerate(alpha_estimates):
            try:
                print(f"Trying alpha = {test_alpha:.6f} (estimate {i+1}/4)")
                alpha_shape = alphashape.alphashape(points, test_alpha)

                # Check if we got a valid result
                if alpha_shape is not None and hasattr(alpha_shape, 'vertices'):
                    if len(alpha_shape.vertices) > 0:
                        print(f"Success with alpha = {test_alpha:.6f}")
                        return alpha_shape
                    else:
                        print(f"Empty result with alpha = {test_alpha:.6f}")
                else:
                    print(f"Invalid result with alpha = {test_alpha:.6f}")

            except Exception as e:
                print(f"Failed with alpha = {test_alpha:.6f}: {e}")
                continue

        # If all estimates failed, try a simple heuristic
        print("All estimates failed, trying simple heuristic...")
        try:
            # Use average nearest neighbor distance
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=2)  # k=2 to get nearest neighbor (excluding self)
            avg_nn_distance = np.mean(distances[:, 1])  # distances[:, 1] is nearest neighbor distance

            test_alpha = avg_nn_distance * 2  # Heuristic: 2x average nearest neighbor distance
            print(f"Trying heuristic alpha = {test_alpha:.6f}")
            alpha_shape = alphashape.alphashape(points, test_alpha)

            if alpha_shape is not None and hasattr(alpha_shape, 'vertices') and len(alpha_shape.vertices) > 0:
                print(f"Success with heuristic alpha = {test_alpha:.6f}")
                return alpha_shape

        except Exception as e:
            print(f"Heuristic method also failed: {e}")

        # Last resort: try the original auto method
        print("Trying alphashape auto-computation as last resort...")
        try:
            alpha_shape = alphashape.alphashape(points)
            return alpha_shape
        except Exception as e:
            print(f"Auto-computation failed: {e}")
            raise Exception("Could not compute alpha shape with any method. Try specifying alpha manually (e.g., --alpha 0.1)")

    else:
        print(f"Using specified alpha: {alpha}")
        alpha_shape = alphashape.alphashape(points, alpha)

        # Validate result
        if alpha_shape is None or not hasattr(alpha_shape, 'vertices') or len(alpha_shape.vertices) == 0:
            raise Exception(f"Alpha shape computation failed with alpha={alpha}. Try a different alpha value.")

        return alpha_shape


def visualize_alpha_shape(points, alpha_shape, file_path, sample_ratio, alpha_param):
    """Create 3D visualization of points and alpha shape hull"""

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Rotate coordinates for Z-up visualization (same as trajectory viewer)
    x_points = points[:, 0]   # x stays x (right)
    y_points = -points[:, 2]  # -z becomes y (forward)
    z_points = points[:, 1]   # y becomes z (up)

    # Plot original points
    scatter = ax.scatter(x_points, y_points, z_points,
                        c='blue', s=5, alpha=0.3, label='Trajectory Points')

    # Plot alpha shape hull if it exists and has faces
    if alpha_shape is not None and hasattr(alpha_shape, 'faces') and len(alpha_shape.faces) > 0:
        # Get vertices and faces of the alpha shape
        vertices = alpha_shape.vertices
        faces = alpha_shape.faces

        # Rotate vertices to match point rotation
        vertices_rotated = np.zeros_like(vertices)
        vertices_rotated[:, 0] = vertices[:, 0]   # x stays x
        vertices_rotated[:, 1] = -vertices[:, 2]  # -z becomes y
        vertices_rotated[:, 2] = vertices[:, 1]   # y becomes z

        # Create mesh for visualization
        face_vertices = vertices_rotated[faces]

        # Create 3D polygon collection
        poly3d = Poly3DCollection(face_vertices, alpha=0.3, facecolor='red',
                                 edgecolor='darkred', linewidths=0.5)
        ax.add_collection3d(poly3d)

        print(f"Alpha shape hull: {len(vertices)} vertices, {len(faces)} faces")
    else:
        print("Warning: Alpha shape computation resulted in empty or invalid hull")
        print("Try adjusting the alpha parameter or increasing sample size")

    # Set labels and title
    ax.set_xlabel('X Position (m) [Right]')
    ax.set_ylabel('Y Position (m) [Forward]')
    ax.set_zlabel('Z Position (m) [Up]')

    filename = os.path.basename(file_path)
    sample_info = f" (sampled {sample_ratio:.1%})" if sample_ratio < 1.0 else ""
    alpha_info = f"α={alpha_param}" if alpha_param else "α=auto"
    ax.set_title(f'Alpha Shape Reachability Hull\n{filename}{sample_info} - {alpha_info}\n{len(points):,} points')

    # Set equal aspect ratio
    max_range = np.array([x_points.max()-x_points.min(),
                         y_points.max()-y_points.min(),
                         z_points.max()-z_points.min()]).max() / 2.0
    mid_x = (x_points.max()+x_points.min()) * 0.5
    mid_y = (y_points.max()+y_points.min()) * 0.5
    mid_z = (z_points.max()+z_points.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Improve the viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig, ax


def save_alpha_shape_mesh(alpha_shape, output_path):
    """Save alpha shape as a mesh file"""
    if not TRIMESH_AVAILABLE:
        print("Warning: trimesh not available. Cannot save mesh file.")
        return False

    if alpha_shape is None or not hasattr(alpha_shape, 'faces') or len(alpha_shape.faces) == 0:
        print("Warning: No valid alpha shape to save")
        return False

    try:
        # Convert to trimesh object
        mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces)

        # Save as PLY file
        mesh.export(output_path)
        print(f"Alpha shape mesh saved to: {output_path}")

        # Print mesh statistics
        print(f"Mesh statistics:")
        print(f"  Vertices: {len(mesh.vertices):,}")
        print(f"  Faces: {len(mesh.faces):,}")
        print(f"  Volume: {mesh.volume:.6f} cubic units")
        print(f"  Surface area: {mesh.area:.6f} square units")

        return True

    except Exception as e:
        print(f"Error saving mesh: {e}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate Alpha Shape Hull for Reachability Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_alpha_hull.py trajectory.txt --visualize
    python generate_alpha_hull.py trajectory.txt --alpha 0.1 --save-mesh hull.ply
    python generate_alpha_hull.py trajectory.txt --sample-ratio 0.5 --alpha 0.05
        """
    )

    parser.add_argument('trajectory_file',
                       help='Path to the hand trajectory file')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Alpha parameter for alpha shape (default: auto-compute)')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                       help='Fraction of points to use (0.0-1.0, default: 1.0)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show interactive 3D visualization')
    parser.add_argument('--save-mesh', type=str,
                       help='Save alpha shape hull as mesh file (.ply)')

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
        print(f"Using {len(trajectory_points):,} sampled points for alpha shape")

    # Compute alpha shape
    try:
        alpha_shape = compute_alpha_shape(trajectory_points, args.alpha)
    except Exception as e:
        print(f"Error computing alpha shape: {e}")
        print("Try adjusting the alpha parameter or sample ratio")
        return 1

    # Save mesh if requested
    if args.save_mesh:
        success = save_alpha_shape_mesh(alpha_shape, args.save_mesh)
        if not success:
            print("Failed to save mesh file")

    # Show visualization if requested
    if args.visualize:
        print("Creating visualization...")
        fig, ax = visualize_alpha_shape(trajectory_points, alpha_shape,
                                       args.trajectory_file, args.sample_ratio, args.alpha)

        print("\nVisualization ready!")
        print("Instructions:")
        print("- Left mouse button: Rotate the view")
        print("- Right mouse button: Zoom in/out")
        print("- Middle mouse button: Pan the view")
        print("- Close the window to exit")

        plt.show()

    print("Alpha shape generation complete!")
    return 0


if __name__ == "__main__":
    exit(main())