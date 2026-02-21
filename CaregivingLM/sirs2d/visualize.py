"""
Visualization utilities for SIRS 2D system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from . import config


def plot_box(ax, box):
    """
    Plot box outline as dashed rectangle.

    Args:
        ax: Matplotlib axes
        box: Box dictionary with 'q1_range' and 'q2_range'
    """
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    width = q1_max - q1_min
    height = q2_max - q2_min

    rect = Rectangle(
        (q1_min, q2_min), width, height,
        fill=False,
        edgecolor=config.BOX_COLOR,
        linestyle=config.BOX_LINESTYLE,
        linewidth=config.BOX_LINEWIDTH,
        label='Box limits'
    )
    ax.add_patch(rect)


def plot_contours(ax, X, Y, H):
    """
    Plot h=0 contour line (feasibility boundary).

    Args:
        ax: Matplotlib axes
        X, Y: Meshgrid coordinates
        H: h values at each grid point
    """
    contour = ax.contour(
        X, Y, H,
        levels=[0.0],
        colors=config.CONTOUR_COLOR,
        linewidths=config.CONTOUR_LINEWIDTH,
        label='Feasible boundary (h=0)'
    )


def plot_bumps(ax, bumps):
    """
    Plot bump centers and 1-sigma ellipses.

    Args:
        ax: Matplotlib axes
        bumps: List of bump dictionaries
    """
    for i, bump in enumerate(bumps):
        mu = bump['mu']
        ls = bump['ls']
        alpha = bump['alpha']

        # Plot center
        ax.scatter(
            mu[0], mu[1],
            marker=config.BUMP_CENTER_MARKER,
            c=config.BUMP_CENTER_COLOR,
            s=config.BUMP_CENTER_SIZE,
            zorder=10,
            label='Bump center' if i == 0 else None
        )

        # Plot 1-sigma ellipse (with rotation if available)
        # Get rotation angle in degrees (Matplotlib expects degrees)
        theta = bump.get('theta', 0.0)  # Radians
        angle_deg = np.degrees(theta)

        ellipse = Ellipse(
            xy=mu,
            width=2 * ls[0],  # 1-sigma corresponds to radius = ls
            height=2 * ls[1],
            angle=angle_deg,  # Rotation angle in degrees
            fill=True,
            facecolor=config.BUMP_ELLIPSE_COLOR,
            alpha=config.BUMP_ELLIPSE_ALPHA,
            edgecolor=config.BUMP_ELLIPSE_COLOR,
            linewidth=config.BUMP_ELLIPSE_LINEWIDTH,
            label='1σ ellipse' if i == 0 else None
        )
        ax.add_patch(ellipse)


def plot_feasible_heat(ax, X, Y, M):
    """
    Plot heatmap of feasible region.

    Args:
        ax: Matplotlib axes
        X, Y: Meshgrid coordinates
        M: Boolean mask (True = feasible)
    """
    # Convert boolean mask to float for plotting
    # Feasible = 1.0 (green), Infeasible = 0.0 (white/gray)
    M_float = M.astype(float)

    im = ax.pcolormesh(
        X, Y, M_float,
        cmap=config.COLORMAP,
        shading='auto',
        alpha=0.6,
        vmin=0.0,
        vmax=1.0
    )

    return im


def compose_panel(box, bumps, X, Y, H, M, title=None, ax=None, show_bumps=True, show_box=True):
    """
    Compose a complete visualization panel.

    Args:
        box: Box dictionary
        bumps: List of bump dictionaries
        X, Y: Meshgrid coordinates
        H: h values
        M: Feasible mask
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        show_bumps: Whether to show bump centers and ellipses
        show_box: Whether to show box outline

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot feasible region heatmap
    plot_feasible_heat(ax, X, Y, M)

    # Plot h=0 contour
    plot_contours(ax, X, Y, H)

    # Plot box outline
    if show_box:
        plot_box(ax, box)

    # Plot bumps
    if show_bumps:
        plot_bumps(ax, bumps)

    # Labels and title
    ax.set_xlabel('Joint 1 (degrees)', fontsize=12)
    ax.set_ylabel('Joint 2 (degrees)', fontsize=12)

    if title is None:
        frac = np.mean(M)
        title = f'K={len(bumps)} bumps, feasible fraction={frac:.2%}'

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc='best', fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Equal aspect ratio for joint space
    ax.set_aspect('equal', adjustable='box')

    return ax


def plot_sample_points(ax, feasible_points, infeasible_points, n_points=50):
    """
    Scatter plot of sample feasible/infeasible points.

    Args:
        ax: Matplotlib axes
        feasible_points: Array of shape (N, 2) with feasible points
        infeasible_points: Array of shape (M, 2) with infeasible points
        n_points: Maximum number of points to plot per category
    """
    if len(feasible_points) > 0:
        sample_idx = np.random.choice(
            len(feasible_points),
            min(n_points, len(feasible_points)),
            replace=False
        )
        ax.scatter(
            feasible_points[sample_idx, 0],
            feasible_points[sample_idx, 1],
            c='green', marker='o', s=20, alpha=0.6,
            label=f'Feasible samples ({len(feasible_points)})',
            zorder=5
        )

    if len(infeasible_points) > 0:
        sample_idx = np.random.choice(
            len(infeasible_points),
            min(n_points, len(infeasible_points)),
            replace=False
        )
        ax.scatter(
            infeasible_points[sample_idx, 0],
            infeasible_points[sample_idx, 1],
            c='red', marker='x', s=20, alpha=0.6,
            label=f'Infeasible samples ({len(infeasible_points)})',
            zorder=5
        )


def save_figure(filename, dpi=150, bbox_inches='tight'):
    """
    Save current figure to file.

    Args:
        filename: Output filename (with path)
        dpi: Resolution
        bbox_inches: Bounding box setting
    """
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure to: {filename}")
