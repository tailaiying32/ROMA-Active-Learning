#!/usr/bin/env python3
"""
OneClass SVM Reachability Boundary Generation

This script uses OneClass SVM with RBF kernel to generate smooth reachability
boundaries from hand trajectory data. OneClass SVM is perfect for this task
because it learns the boundary of reachable space from positive examples only.

Usage:
    python generate_svm_boundary.py trajectory_file.txt [options]

Arguments:
    trajectory_file: Path to the hand trajectory file
    --gamma: RBF kernel gamma parameter (default: auto-tune)
    --nu: Outlier fraction parameter (default: 0.05)
    --sample-ratio: Fraction of points to use (default: 1.0)
    --resolution: Grid resolution for boundary visualization (default: 50)
    --visualize: Show interactive 3D visualization
    --save-model: Save trained SVM model for later use
    --query: Test reachability of specific points

Dependencies:
    pip install scikit-learn numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import random
import pickle
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid


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


def load_endpoint_data(file_path):
    """Load endpoint positions from results file"""
    endpoint_points = []

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
                            endpoint_points.append((x, y, z))
                        except ValueError:
                            continue

    except Exception as e:
        print(f"Error reading endpoint file: {e}")
        return None

    return np.array(endpoint_points)


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



def auto_tune_svm_parameters_for_completion(points, gamma_values=None):
    """Auto-tune SVM to capture detailed geometry while including all points"""
    print("Auto-tuning SVM to capture detailed geometry...")

    if gamma_values is None:
        # Estimate reasonable gamma range based on data scale
        from sklearn.metrics.pairwise import pairwise_distances
        distances = pairwise_distances(points[:min(1000, len(points))])
        median_dist = np.median(distances[distances > 0])

        # Use much tighter gamma values to capture detailed bowl geometry
        gamma_values = [
            1 / (0.1 * median_dist ** 2),   # Extremely tight (maximum detail)
            1 / (0.25 * median_dist ** 2),  # Very tight
            1 / (0.5 * median_dist ** 2),   # Tight
            1 / (median_dist ** 2),         # Medium-tight
            1 / (2 * median_dist ** 2),     # Medium
            1 / (4 * median_dist ** 2),     # Loose
            1 / (8 * median_dist ** 2),     # Very loose (fallback)
        ]

    # Strategy: Find best gamma/nu combination using joint optimization
    nu_candidates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    def test_gamma_with_best_nu(gamma, points):
        """Test a gamma with multiple nu values, return best nu that works"""
        best_nu = None
        best_inclusion = 0

        for nu in nu_candidates:
            try:
                model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
                model.fit(points)
                predictions = model.predict(points)
                reachable_ratio = np.sum(predictions == 1) / len(points)

                if reachable_ratio >= 0.98 and reachable_ratio > best_inclusion:
                    best_nu = nu
                    best_inclusion = reachable_ratio
            except:
                continue

        return best_nu, best_inclusion

    print("Finding initial working gamma with optimized nu...")

    # Step 1: Find ANY working gamma with its best nu
    initial_gamma = None
    initial_nu = None
    for gamma in gamma_values:  # Try from tight to loose
        best_nu, best_inclusion = test_gamma_with_best_nu(gamma, points)

        print(f"gamma={gamma:.6f}: best_nu={best_nu}, inclusion={best_inclusion:.1%}")

        if best_nu is not None:
            initial_gamma = gamma
            initial_nu = best_nu
            print(f"✓ Found initial working solution: gamma={gamma:.6f}, nu={best_nu:.3f} ({best_inclusion:.1%})")
            break

    if initial_gamma is None:
        raise Exception("Could not find any working gamma/nu combination")

    # Step 2: Binary search for MAXIMUM gamma that still works
    print(f"\nBinary searching for maximum gamma starting from {initial_gamma:.6f}...")

    gamma_low = initial_gamma
    gamma_high = initial_gamma * 100  # Start with much higher upper bound

    # First, find upper bound by doubling until failure with joint gamma-nu optimization
    print("Finding upper bound by doubling...")
    test_gamma = initial_gamma
    best_nu_low = initial_nu
    while True:
        test_gamma *= 2  # Double each time

        best_nu, inclusion_ratio = test_gamma_with_best_nu(test_gamma, points)

        print(f"  gamma={test_gamma:.6f}: ", end="")

        if best_nu is not None:
            print(f"{inclusion_ratio:.1%} with nu={best_nu:.3f} ✓")
            gamma_low = test_gamma  # This works, try higher
            best_nu_low = best_nu
        else:
            print("No valid nu found ✗ (upper bound found)")
            gamma_high = test_gamma  # This fails, stop here
            break

    # Now binary search between gamma_low and gamma_high with joint optimization
    print(f"Binary searching between {gamma_low:.6f} and {gamma_high:.6f}...")

    best_gamma = gamma_low
    best_nu = best_nu_low

    for iteration in range(10):  # Max 10 iterations for precision
        gamma_mid = (gamma_low + gamma_high) / 2

        mid_nu, inclusion_ratio = test_gamma_with_best_nu(gamma_mid, points)

        print(f"  Iteration {iteration+1}: gamma={gamma_mid:.6f} → ", end="")

        if mid_nu is not None:
            print(f"{inclusion_ratio:.1%} with nu={mid_nu:.3f} ✓ (works)")
            gamma_low = gamma_mid  # This works, try higher
            best_gamma = gamma_mid
            best_nu = mid_nu
        else:
            print("No valid nu found ✗ (too tight)")
            gamma_high = gamma_mid  # This fails, try lower

        # Stop if range is small enough
        if (gamma_high - gamma_low) / gamma_low < 0.01:  # 1% precision
            break

    # Use the best working gamma-nu combination
    best_model = OneClassSVM(kernel='rbf', gamma=best_gamma, nu=best_nu)
    best_model.fit(points)

    print(f"\n✓ FINAL RESULT - Maximum gamma found with optimized nu:")
    print(f"  gamma={best_gamma:.6f}, nu={best_nu:.3f}")

    # Verify final result
    final_predictions = best_model.predict(points)
    final_ratio = np.sum(final_predictions == 1) / len(points)
    print(f"  Final inclusion: {final_ratio:.1%} ({np.sum(final_predictions == 1)}/{len(points)})")

    return best_model, {'gamma': best_gamma, 'nu': initial_nu}

    # Fallback: lower inclusion requirement to 95%
    print("\nNo solutions with 98% inclusion found. Trying 95% threshold...")
    for solution_data in [(gamma, nu) for gamma in gamma_values for nu in nu_values]:
        gamma, nu = solution_data
        try:
            model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
            model.fit(points)
            predictions = model.predict(points)
            reachable_ratio = np.sum(predictions == 1) / len(points)

            if reachable_ratio >= 0.95:
                print(f"✓ Fallback solution: gamma={gamma:.6f}, nu={nu:.3f}, inclusion={reachable_ratio:.1%}")
                return model, {'gamma': gamma, 'nu': nu}
        except:
            continue

    # Fallback: use very inclusive parameters
    print("Warning: Could not find parameters that include enough trajectory points")
    print("Using most inclusive parameters as fallback...")

    gamma = gamma_values[-1]  # Smallest (most inclusive)
    nu = 0.001               # Smallest nu

    try:
        model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        model.fit(points)
        return model, {'gamma': gamma, 'nu': nu}
    except Exception as e:
        raise Exception(f"Could not train SVM even with most inclusive parameters: {e}")


def auto_tune_svm_with_percentile(points, gamma_percentile=100.0, gamma_values=None):
    """Auto-tune SVM using binary search with gamma percentile selection"""
    if gamma_percentile == 100.0:
        print("Auto-tuning SVM for tightest fit...")
    else:
        print(f"Auto-tuning SVM with gamma percentile {gamma_percentile:.1f}% for gap-filling...")

    if gamma_values is None:
        # Estimate reasonable gamma range based on data scale
        from sklearn.metrics.pairwise import pairwise_distances
        distances = pairwise_distances(points[:min(1000, len(points))])
        median_dist = np.median(distances[distances > 0])

        # Use starting gamma values for initial search
        gamma_values = [
            1 / (0.1 * median_dist ** 2),   # Extremely tight (maximum detail)
            1 / (0.25 * median_dist ** 2),  # Very tight
            1 / (0.5 * median_dist ** 2),   # Tight
            1 / (median_dist ** 2),         # Medium-tight
            1 / (2 * median_dist ** 2),     # Medium
            1 / (4 * median_dist ** 2),     # Loose
            1 / (8 * median_dist ** 2),     # Very loose (fallback)
        ]

    # Nu candidates for joint optimization
    nu_candidates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    def test_gamma_with_best_nu(gamma, points):
        """Test a gamma with multiple nu values, return best nu that works"""
        best_nu = None
        best_inclusion = 0

        for nu in nu_candidates:
            try:
                model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
                model.fit(points)
                predictions = model.predict(points)
                reachable_ratio = np.sum(predictions == 1) / len(points)
                if reachable_ratio >= 0.98 and reachable_ratio > best_inclusion:
                    best_nu = nu
                    best_inclusion = reachable_ratio
            except:
                continue
        return best_nu, best_inclusion

    # STEP 1: Find initial working gamma from candidates
    print("Finding initial working gamma from candidates...")
    best_gamma_initial = None
    initial_nu = None

    for gamma in sorted(gamma_values, reverse=True):  # Start from tightest
        best_nu, inclusion_ratio = test_gamma_with_best_nu(gamma, points)
        if best_nu is not None:
            best_gamma_initial = gamma
            initial_nu = best_nu
            print(f"  Initial: gamma={gamma:.6f}, nu={best_nu:.3f}, inclusion={inclusion_ratio:.1%}")
            break

    if best_gamma_initial is None:
        raise Exception("Could not find any working gamma-nu combination")

    # STEP 2: Binary search to find upper bound (maximum gamma)
    print("Finding upper bound by doubling...")
    gamma_low = best_gamma_initial
    gamma_high = None
    best_nu_low = initial_nu

    test_gamma = gamma_low * 2
    while gamma_high is None:
        print(f"  Testing gamma={test_gamma:.6f}...", end="")

        best_nu, inclusion_ratio = test_gamma_with_best_nu(test_gamma, points)

        if best_nu is not None:
            print(f" ✓ (nu={best_nu:.3f}, {inclusion_ratio:.1%})")
            gamma_low = test_gamma  # This works, try higher
            best_nu_low = best_nu
            test_gamma *= 2
        else:
            print(" ✗ (upper bound found)")
            gamma_high = test_gamma  # This fails, stop here
            break

    # STEP 3: Binary search refinement between gamma_low and gamma_high
    print(f"Binary searching between {gamma_low:.6f} and {gamma_high:.6f}...")

    for iteration in range(10):  # Max 10 iterations for precision
        gamma_mid = (gamma_low + gamma_high) / 2

        mid_nu, inclusion_ratio = test_gamma_with_best_nu(gamma_mid, points)

        print(f"  Iteration {iteration+1}: gamma={gamma_mid:.6f} → ", end="")

        if mid_nu is not None:
            print(f"{inclusion_ratio:.1%} with nu={mid_nu:.3f} ✓ (works)")
            gamma_low = gamma_mid  # This works, try higher
            best_nu_low = mid_nu
        else:
            print("No valid nu found ✗ (too tight)")
            gamma_high = gamma_mid  # This fails, try lower

        # Stop if range is small enough
        if (gamma_high - gamma_low) / gamma_low < 0.01:  # 1% precision
            break

    # STEP 4: Calculate target gamma as percentage of maximum found
    max_gamma = gamma_low
    target_gamma = max_gamma * (gamma_percentile / 100.0)

    print(f"\nUsing gamma: {target_gamma:.6f} ({gamma_percentile:.1f}% of max {max_gamma:.6f})")

    # Find best nu for this target gamma
    best_nu, inclusion_ratio = test_gamma_with_best_nu(target_gamma, points)

    if best_nu is None:
        raise Exception(f"Could not find valid nu for gamma={target_gamma:.6f}")

    print(f"✓ Found valid nu={best_nu:.3f} with {inclusion_ratio:.1%} inclusion")

    # Train final model
    model = OneClassSVM(kernel='rbf', gamma=target_gamma, nu=best_nu)
    model.fit(points)

    return model, {
        'gamma': target_gamma,
        'nu': best_nu
    }


def train_svm_boundary(points, gamma=None, nu=0.05, gamma_percentile=100.0):
    """Train OneClass SVM to learn reachability boundary"""
    print(f"Training OneClass SVM on {len(points):,} points...")

    # Standardize the data for better SVM performance
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    if gamma is None:
        # Auto-tune with gamma percentile control
        svm_model, best_params = auto_tune_svm_with_percentile(points_scaled, gamma_percentile)
        gamma = best_params['gamma']
        nu = best_params['nu']
    else:
        # Use specified parameters
        print(f"Using specified parameters: gamma={gamma:.6f}, nu={nu:.3f}")
        svm_model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        svm_model.fit(points_scaled)

    # Print model statistics
    n_support = len(svm_model.support_vectors_)
    support_ratio = n_support / len(points_scaled)
    print(f"Model trained successfully:")
    print(f"  Support vectors: {n_support}/{len(points_scaled)} ({support_ratio:.1%})")
    print(f"  Final gamma: {gamma:.6f}")
    print(f"  Final nu: {nu:.3f}")

    return svm_model, scaler, gamma, nu


def create_boundary_grid(points, resolution=50):
    """Create 3D grid for boundary visualization"""
    # Add some padding around the data
    padding = 0.1
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range
    z_min -= padding * z_range
    z_max += padding * z_range

    # Create 3D grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)

    X, Y, Z = np.meshgrid(x, y, z)
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    return grid_points, X, Y, Z


def visualize_svm_boundary(points, svm_model, scaler, file_path, sample_ratio,
                          gamma, nu, resolution=50, gamma_percentile=100.0):
    """Create simple 3D visualization of reachable vs unreachable regions"""

    print("Creating simple 3D visualization...")

    # Create boundary grid
    grid_points, X, Y, Z = create_boundary_grid(points, resolution)

    # Transform grid points and predict
    grid_scaled = scaler.transform(grid_points)
    predictions = svm_model.predict(grid_scaled)

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Rotate coordinates for Z-up visualization (same as trajectory viewer)
    x_points = points[:, 0]   # x stays x (right)
    y_points = -points[:, 2]  # -z becomes y (forward)
    z_points = points[:, 1]   # y becomes z (up)

    # Plot original trajectory points using simple plot
    ax.plot(x_points, y_points, z_points, 'b.', markersize=4, alpha=0.8, label='Trajectory Points')

    # Add shoulder position as a big red sphere
    # Shoulder center from run_optimization.py: [0.0, 1.25, 0.15]
    shoulder_x = 0.0
    shoulder_y = 1.25
    shoulder_z = 0.15

    # Rotate shoulder coordinates for Z-up visualization (same as trajectory points)
    shoulder_x_rot = shoulder_x      # x stays x (right)
    shoulder_y_rot = -shoulder_z     # -z becomes y (forward)
    shoulder_z_rot = shoulder_y      # y becomes z (up)

    ax.scatter(shoulder_x_rot, shoulder_y_rot, shoulder_z_rot,
              c='red', s=500, alpha=0.9, label='Shoulder Center',
              edgecolors='darkred', linewidths=2)

    # Subsample grid for visualization (reduced for denser sphere)
    subsample_factor = 2
    grid_sub = grid_points[::subsample_factor]
    pred_sub = predictions[::subsample_factor]

    # Separate reachable vs unreachable points
    reachable_mask = pred_sub == 1
    unreachable_mask = pred_sub == -1

    print(f"Grid classification: {np.sum(reachable_mask)} reachable, {np.sum(unreachable_mask)} unreachable")

    if np.any(reachable_mask):
        reachable_points = grid_sub[reachable_mask]
        # Rotate for visualization
        x_reach = reachable_points[:, 0]
        y_reach = -reachable_points[:, 2]  # -z becomes y
        z_reach = reachable_points[:, 1]   # y becomes z

        # Create RGB color gradient using all 3 coordinates normalized to [0,1]
        # X -> Red channel, Y -> Green channel, Z -> Blue channel
        x_norm = (x_reach - x_reach.min()) / (x_reach.max() - x_reach.min()) if x_reach.max() > x_reach.min() else np.zeros_like(x_reach)
        y_norm = (y_reach - y_reach.min()) / (y_reach.max() - y_reach.min()) if y_reach.max() > y_reach.min() else np.zeros_like(y_reach)
        z_norm = (z_reach - z_reach.min()) / (z_reach.max() - z_reach.min()) if z_reach.max() > z_reach.min() else np.zeros_like(z_reach)

        # Combine into RGB colors
        colors = np.column_stack([x_norm, y_norm, z_norm])

        scatter = ax.scatter(x_reach, y_reach, z_reach, c=colors,
                           s=25, alpha=0.8, label='Reachable Region')

        # Add text explanation of color mapping
        color_info = "Color: Red=X, Green=Y, Blue=Z (normalized to cloud range)"
        ax.text2D(0.02, 0.02, color_info, transform=ax.transAxes, fontsize=8,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set labels and title
    ax.set_xlabel('X Position (m) [Right]')
    ax.set_ylabel('Y Position (m) [Forward]')
    ax.set_zlabel('Z Position (m) [Up]')

    filename = os.path.basename(file_path)
    sample_info = f" (sampled {sample_ratio:.1%})" if sample_ratio < 1.0 else ""
    param_info = f"γ={gamma:.4f}, ν={nu:.3f}"
    percentile_info = f" - {gamma_percentile:.0f}% Percentile" if gamma_percentile < 100.0 else ""
    ax.set_title(f'OneClass SVM Reachability Classification{percentile_info}\n{filename}{sample_info} - {param_info}\n{len(points):,} points')

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


def visualize_endpoints_only(points, file_path, sample_ratio):
    """Create 3D visualization showing only endpoint positions"""
    print("Creating endpoints-only visualization...")

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Rotate coordinates for Z-up visualization
    x_points = points[:, 0]   # x stays x (right)
    y_points = -points[:, 2]  # -z becomes y (forward)
    z_points = points[:, 1]   # y becomes z (up)

    # Add shoulder position as a big red sphere
    shoulder_x_rot = 0.0      # x stays x (right)
    shoulder_y_rot = -0.15    # -z becomes y (forward)
    shoulder_z_rot = 1.25     # y becomes z (up)

    ax.scatter(shoulder_x_rot, shoulder_y_rot, shoulder_z_rot,
              c='red', s=500, alpha=0.9, label='Shoulder Center',
              edgecolors='darkred', linewidths=2)

    # Plot endpoint positions as big spheres for visibility
    # Create RGB color gradient for endpoints
    x_norm = (x_points - x_points.min()) / (x_points.max() - x_points.min()) if x_points.max() > x_points.min() else np.zeros_like(x_points)
    y_norm = (y_points - y_points.min()) / (y_points.max() - y_points.min()) if y_points.max() > y_points.min() else np.zeros_like(y_points)
    z_norm = (z_points - z_points.min()) / (z_points.max() - z_points.min()) if z_points.max() > z_points.min() else np.zeros_like(z_points)

    colors = np.column_stack([x_norm, y_norm, z_norm])

    ax.scatter(x_points, y_points, z_points, c=colors,
              s=100, alpha=0.8, label='Endpoint Positions',
              edgecolors='black', linewidths=0.5)

    # Set labels and title
    ax.set_xlabel('X Position (m) [Right]')
    ax.set_ylabel('Y Position (m) [Forward]')
    ax.set_zlabel('Z Position (m) [Up]')

    filename = os.path.basename(file_path)
    sample_info = f" (sampled {sample_ratio:.1%})" if sample_ratio < 1.0 else ""
    ax.set_title(f'Endpoint Positions Only\n{filename}{sample_info}\n{len(points):,} endpoints')

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

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig, ax


def visualize_trajectories_only(points, file_path, sample_ratio):
    """Create 3D visualization showing only trajectory paths"""
    print("Creating trajectories-only visualization...")

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Rotate coordinates for Z-up visualization
    x_points = points[:, 0]   # x stays x (right)
    y_points = -points[:, 2]  # -z becomes y (forward)
    z_points = points[:, 1]   # y becomes z (up)

    # Add shoulder position as a big red sphere
    shoulder_x_rot = 0.0      # x stays x (right)
    shoulder_y_rot = -0.15    # -z becomes y (forward)
    shoulder_z_rot = 1.25     # y becomes z (up)

    ax.scatter(shoulder_x_rot, shoulder_y_rot, shoulder_z_rot,
              c='red', s=500, alpha=0.9, label='Shoulder Center',
              edgecolors='darkred', linewidths=2)

    # Plot trajectory points as small dots with temporal color progression
    scatter = ax.scatter(x_points, y_points, z_points, c=range(len(x_points)), cmap='viridis',
                        s=15, alpha=0.7, edgecolors='none', label='Trajectory Points')

    # Add colorbar to show temporal progression
    try:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Trajectory Progression', rotation=270, labelpad=15)
    except:
        pass

    # Set labels and title
    ax.set_xlabel('X Position (m) [Right]')
    ax.set_ylabel('Y Position (m) [Forward]')
    ax.set_zlabel('Z Position (m) [Up]')

    filename = os.path.basename(file_path)
    sample_info = f" (sampled {sample_ratio:.1%})" if sample_ratio < 1.0 else ""
    ax.set_title(f'Trajectory Paths Only\n{filename}{sample_info}\n{len(points):,} points')

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

    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig, ax


def create_rotating_gif(points, svm_model, scaler, file_path, sample_ratio,
                       gamma, nu, resolution, gamma_percentile, gif_path, vis_mode='reachable'):
    """Create a rotating GIF of the 3D visualization"""
    print(f"Creating rotating GIF: {gif_path}")

    try:
        import matplotlib.animation as animation
        from PIL import Image
        import io
    except ImportError:
        print("Error: Need pillow for GIF creation. Install with: pip install pillow")
        return False

    # Create the appropriate visualization based on mode
    if vis_mode == 'endpoints':
        fig, ax = visualize_endpoints_only(points, file_path, sample_ratio)
    elif vis_mode == 'trajectories':
        fig, ax = visualize_trajectories_only(points, file_path, sample_ratio)
    else:  # 'reachable'
        fig, ax = visualize_svm_boundary(points, svm_model, scaler, file_path, sample_ratio,
                                       gamma, nu, resolution, gamma_percentile)

    # Remove the interactive elements for cleaner GIF
    ax.legend().set_visible(False)  # Hide legend for cleaner look

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


def save_svm_model(svm_model, scaler, output_path, metadata=None):
    """Save SVM model and scaler for later use"""
    try:
        model_data = {
            'svm_model': svm_model,
            'scaler': scaler,
            'metadata': metadata or {}
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"SVM model saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def load_svm_model(model_path):
    """Load saved SVM model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        return model_data['svm_model'], model_data['scaler'], model_data.get('metadata', {})

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def query_reachability(svm_model, scaler, query_points):
    """Test reachability of specific points"""
    if len(query_points) == 0:
        return

    print(f"\nTesting reachability of {len(query_points)} query points:")

    # Transform query points
    query_scaled = scaler.transform(query_points)

    # Get predictions and decision values
    predictions = svm_model.predict(query_scaled)
    decision_values = svm_model.decision_function(query_scaled)

    print("Results:")
    print("Point (x, y, z) -> Reachable | Decision Value")
    print("-" * 50)

    for i, (point, pred, decision) in enumerate(zip(query_points, predictions, decision_values)):
        reachable = "Yes" if pred == 1 else "No"
        print(f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) -> {reachable:>3} | {decision:.4f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='OneClass SVM Reachability Boundary Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_svm_boundary.py trajectory.txt --visualize
    python generate_svm_boundary.py trajectory.txt --gamma 0.1 --nu 0.05 --visualize
    python generate_svm_boundary.py trajectory.txt --sample-ratio 0.5 --save-model model.pkl
        """
    )

    parser.add_argument('trajectory_file',
                       help='Path to the hand trajectory file')
    parser.add_argument('--gamma', type=float, default=None,
                       help='RBF kernel gamma parameter (default: auto-tune)')
    parser.add_argument('--nu', type=float, default=0.05,
                       help='Outlier fraction parameter (default: 0.05)')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                       help='Fraction of points to use (0.0-1.0, default: 1.0)')
    parser.add_argument('--resolution', type=int, default=60,
                       help='Grid resolution for visualization (default: 50)')
    parser.add_argument('--gamma-percentile', type=float, default=100.0,
                       help='Gamma percentile to use (0-100, default: 100.0=tightest fit, 50.0=medium gap-filling, 25.0=more gap-filling)')
    parser.add_argument('--vis-mode', type=str, default='reachable', choices=['endpoints', 'trajectories', 'reachable'],
                       help='Visualization mode: endpoints (final positions only), trajectories (full paths), reachable (SVM boundary)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show interactive 3D visualization')
    parser.add_argument('--save-gif', type=str,
                       help='Save rotating 3D visualization as GIF file')
    parser.add_argument('--save-model', type=str,
                       help='Save trained SVM model to file (.pkl)')
    parser.add_argument('--query', type=str,
                       help='Query points file (format: x y z per line)')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.trajectory_file):
        print(f"Error: Trajectory file not found: {args.trajectory_file}")
        return 1

    if args.sample_ratio <= 0 or args.sample_ratio > 1.0:
        print(f"Error: Sample ratio must be between 0.0 and 1.0, got: {args.sample_ratio}")
        return 1

    print(f"Loading data from: {args.trajectory_file}")

    # Load data based on visualization mode
    if args.vis_mode == 'endpoints':
        # For endpoints mode, expect a results file (condition_x_results.txt)
        trajectory_points = load_endpoint_data(args.trajectory_file)
        data_type = "endpoint positions"
    else:
        # For trajectories and reachable modes, expect trajectory file
        trajectory_points = load_trajectory_data(args.trajectory_file)
        data_type = "trajectory points"

    if trajectory_points is None or len(trajectory_points) == 0:
        print(f"Error: No valid {data_type} found in the file")
        return 1

    print(f"Loaded {len(trajectory_points):,} {data_type}")

    # Sample points if requested
    if args.sample_ratio < 1.0:
        print(f"Sampling {args.sample_ratio:.1%} of points...")
        trajectory_points = sample_points(trajectory_points, args.sample_ratio)
        print(f"Using {len(trajectory_points):,} sampled points for SVM training")

    # Train SVM model (only needed for reachable mode)
    if args.vis_mode == 'reachable':
        try:
            svm_model, scaler, gamma, nu = train_svm_boundary(
                trajectory_points, args.gamma, args.nu, args.gamma_percentile)
        except Exception as e:
            print(f"Error training SVM: {e}")
            return 1
    else:
        svm_model, scaler, gamma, nu = None, None, None, None

    # Save model if requested
    if args.save_model:
        metadata = {
            'gamma': gamma,
            'nu': nu,
            'n_points': len(trajectory_points),
            'source_file': args.trajectory_file
        }
        save_svm_model(svm_model, scaler, args.save_model, metadata)

    # Process query points if provided
    if args.query:
        if os.path.exists(args.query):
            query_points = load_trajectory_data(args.query)
            if query_points is not None:
                query_reachability(svm_model, scaler, query_points)
        else:
            print(f"Warning: Query file not found: {args.query}")

    # Show visualization if requested
    if args.visualize:
        print(f"Creating {args.vis_mode} visualization...")
        try:
            if args.vis_mode == 'endpoints':
                fig, ax = visualize_endpoints_only(
                    trajectory_points, args.trajectory_file, args.sample_ratio)
            elif args.vis_mode == 'trajectories':
                fig, ax = visualize_trajectories_only(
                    trajectory_points, args.trajectory_file, args.sample_ratio)
            else:  # 'reachable'
                fig, ax = visualize_svm_boundary(
                    trajectory_points, svm_model, scaler,
                    args.trajectory_file, args.sample_ratio,
                    gamma, nu, args.resolution, args.gamma_percentile)

            print("\nVisualization ready!")
            print("Instructions:")
            print("- Left mouse button: Rotate the view")
            print("- Right mouse button: Zoom in/out")
            print("- Middle mouse button: Pan the view")
            print("- Close the window to exit")

            plt.show()

        except Exception as e:
            print(f"Error creating visualization: {e}")

    # Create rotating GIF if requested
    if args.save_gif:
        try:
            success = create_rotating_gif(
                trajectory_points, svm_model, scaler,
                args.trajectory_file, args.sample_ratio,
                gamma, nu, args.resolution, args.gamma_percentile,
                args.save_gif, args.vis_mode)
            if not success:
                print("Failed to create GIF")
        except Exception as e:
            print(f"Error creating GIF: {e}")

    print("SVM boundary generation complete!")
    return 0


if __name__ == "__main__":
    exit(main())