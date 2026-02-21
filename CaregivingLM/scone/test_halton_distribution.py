#!/usr/bin/env python3
"""
Test script to verify Halton sample distribution quality
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_samples(filename):
    """Load samples from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data['samples']), data['sphere_center'], data['sphere_radius']

def compute_distribution_metrics(samples, center, radius):
    """Compute various metrics to assess distribution quality."""
    center = np.array(center)
    
    # Compute distances from center
    distances = np.linalg.norm(samples - center, axis=1)
    
    # Theoretical maximum distance should be radius
    max_distance = np.max(distances)
    
    # Compute radial distribution (should follow r^2 for volume uniformity)
    # For uniform distribution in sphere, P(r < R) = (R/radius)^3
    sorted_distances = np.sort(distances)
    n_samples = len(samples)
    
    # Expected cumulative distribution for uniform sphere
    theoretical_cdf = (sorted_distances / radius) ** 3
    empirical_cdf = np.arange(1, n_samples + 1) / n_samples
    
    # Kolmogorov-Smirnov statistic
    ks_statistic = np.max(np.abs(theoretical_cdf - empirical_cdf))
    
    # Compute nearest neighbor distances (should be well-separated)
    from scipy.spatial.distance import pdist
    pairwise_distances = pdist(samples)
    min_distance = np.min(pairwise_distances)
    mean_nn_distance = np.mean(pairwise_distances)
    
    return {
        'max_distance': max_distance,
        'radius': radius,
        'ks_statistic': ks_statistic,
        'min_distance': min_distance,
        'mean_distance': mean_nn_distance,
        'n_samples': n_samples
    }

def plot_samples_3d(samples, center, radius, title="Halton Samples in Sphere"):
    """Create 3D plot of samples."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot samples
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], 
               c=range(len(samples)), cmap='viridis', s=50, alpha=0.7)
    
    # Plot sphere center
    ax.scatter(*center, color='red', s=100, marker='*', label='Center')
    
    # Create sphere wireframe for reference
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.2, color='gray')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    return fig

def plot_radial_distribution(samples, center, radius):
    """Plot radial distribution comparison."""
    center = np.array(center)
    distances = np.linalg.norm(samples - center, axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of distances
    ax1.hist(distances, bins=20, alpha=0.7, density=True, label='Empirical')
    
    # Theoretical distribution for uniform sphere: 3r^2/R^3
    r_theory = np.linspace(0, radius, 100)
    theoretical_density = 3 * r_theory**2 / radius**3
    ax1.plot(r_theory, theoretical_density, 'r-', linewidth=2, label='Theoretical')
    
    ax1.set_xlabel('Distance from center')
    ax1.set_ylabel('Density')
    ax1.set_title('Radial Distance Distribution')
    ax1.legend()
    
    # Q-Q plot
    sorted_distances = np.sort(distances)
    n = len(sorted_distances)
    empirical_quantiles = np.arange(1, n + 1) / n
    theoretical_quantiles = (sorted_distances / radius) ** 3
    
    ax2.plot([0, 1], [0, 1], 'r--', label='Perfect match')
    ax2.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.7)
    ax2.set_xlabel('Theoretical quantiles')
    ax2.set_ylabel('Empirical quantiles')
    ax2.set_title('Q-Q Plot (Radial Distribution)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load the generated samples
    samples_file = "halton_samples_100.json"
    
    print(f"Loading samples from {samples_file}")
    samples, center, radius = load_samples(samples_file)
    
    print(f"Loaded {len(samples)} samples")
    print(f"Sphere center: {center}")
    print(f"Sphere radius: {radius}")
    
    # Compute distribution metrics
    metrics = compute_distribution_metrics(samples, center, radius)
    
    print("\n=== Distribution Quality Metrics ===")
    print(f"Maximum distance from center: {metrics['max_distance']:.4f} (should be ≤ {metrics['radius']:.4f})")
    print(f"Kolmogorov-Smirnov statistic: {metrics['ks_statistic']:.4f} (lower is better, < 0.1 is good)")
    print(f"Minimum pairwise distance: {metrics['min_distance']:.4f}")
    print(f"Mean pairwise distance: {metrics['mean_distance']:.4f}")
    
    # Check if samples are within sphere
    within_sphere = metrics['max_distance'] <= radius
    print(f"All samples within sphere: {within_sphere}")
    
    # Quality assessment
    if metrics['ks_statistic'] < 0.1:
        print("✓ Good radial distribution (KS < 0.1)")
    else:
        print("⚠ Radial distribution could be improved (KS ≥ 0.1)")
    
    # Create plots
    print("\nCreating distribution plots...")
    
    # 3D scatter plot
    fig1 = plot_samples_3d(samples, center, radius)
    fig1.savefig('halton_samples_3d.png', dpi=300, bbox_inches='tight')
    print("Saved 3D visualization to halton_samples_3d.png")
    
    # Radial distribution plot
    fig2 = plot_radial_distribution(samples, center, radius)
    fig2.savefig('halton_radial_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved radial distribution plot to halton_radial_distribution.png")
    
    plt.show()