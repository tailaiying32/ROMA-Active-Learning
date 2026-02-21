"""
Generate a combined GIF showing multiple seeds side-by-side.

Generates frames for all specified seeds, then combines them into a grid GIF
showing only every N iterations.

Usage:
    python generate_combined_gif.py
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.config import DEVICE

from run_toy_2d_demo import (
    Toy2DChecker,
    Toy2DDecoder,
    Toy2DOracle,
    Toy2DParticlePosterior,
    Toy2DSVGD,
    Toy2DBALD,
    compute_boundary_data,
    create_random_blob_gt,
)


def compute_iou(data):
    """Compute IoU between GT and predicted feasibility regions."""
    gt_mask = data['Z_gt'] > 0
    pred_mask = data['Z_pred'] > 0
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def plot_frame_for_seed(data, iteration, ax, seed, show_legend=True):
    """Plot a single seed's frame onto a given axis."""
    X, Y = data['X'], data['Y']
    Z_gt, Z_pred = data['Z_gt'], data['Z_pred']

    # Colors
    GT_FILL = '#87CEEB'
    GT_BORDER = '#00CED1'
    PRED_FILL = '#FFB366'
    PRED_BORDER = '#FF6600'

    # GT region
    gt_feasible = Z_gt > 0
    ax.contourf(X, Y, gt_feasible.astype(float), levels=[0.5, 1.5],
               colors=[GT_FILL], alpha=0.5, zorder=1)
    try:
        ax.contour(X, Y, Z_gt, levels=[0], colors=[GT_BORDER],
                  linewidths=2, linestyles='solid', zorder=2)
    except Exception:
        pass

    # Predicted region
    pred_feasible = Z_pred > 0
    ax.contourf(X, Y, pred_feasible.astype(float), levels=[0.5, 1.5],
               colors=[PRED_FILL], alpha=0.4, zorder=3)
    try:
        ax.contour(X, Y, Z_pred, levels=[0], colors=[PRED_BORDER],
                  linewidths=2, linestyles='dashed', zorder=4)
    except Exception:
        pass

    # Clean formatting
    ax.set_title(f'Seed {seed}', fontsize=10, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('auto')


def run_seed_and_capture_frames(seed, budget, particles=103, n_blobs=10, gt_blobs=7):
    """
    Run active learning for a seed and capture boundary data at each iteration.

    Returns:
        frames_data: list of boundary data dicts for each iteration (0 to budget inclusive)
    """
    gt_seed = seed
    learning_seed = seed + 1000
    torch.manual_seed(learning_seed)
    np.random.seed(learning_seed)

    # Create GT
    gt_checker = create_random_blob_gt(gt_seed, n_blobs=gt_blobs, device=DEVICE)
    oracle = Toy2DOracle(gt_checker)

    # Create decoder and posterior
    decoder = Toy2DDecoder(n_blobs=n_blobs, device=DEVICE)

    prior_mean = torch.zeros(decoder.latent_dim, device=DEVICE)
    size_start = n_blobs * 2
    weight_start = n_blobs * 3
    n_positive = (n_blobs * 2) // 3

    prior_mean[size_start:weight_start] = 0.3
    prior_mean[weight_start:weight_start + n_positive] = 0.95
    prior_mean[weight_start + n_positive:] = -0.20

    prior_std = torch.ones(decoder.latent_dim, device=DEVICE) * 1.0

    posterior = Toy2DParticlePosterior(
        decoder.latent_dim, particles, prior_mean, prior_std, DEVICE
    )

    # Create SVGD and BALD
    svgd = Toy2DSVGD(decoder, posterior, prior_mean, prior_std,
                     step_size=0.05, n_iters=50, device=DEVICE)
    bald = Toy2DBALD(decoder, posterior, tau=0.3, device=DEVICE)

    bounds = torch.tensor([[-1.2, 1.2], [-1.2, 1.2]], device=DEVICE)

    frames_data = []

    for i in range(budget):
        # Capture frame before the query
        data = compute_boundary_data(gt_checker, decoder, posterior, bounds)
        frames_data.append(data)

        # Do the active learning step
        test_point, score = bald.select_test(bounds)
        outcome = oracle.query(test_point)
        svgd.add_observation(test_point, outcome)
        svgd.update()

        iou = compute_iou(data)
        print(f"  Seed {seed} iter {i}: IoU={iou:.4f}")

    # Final frame
    data = compute_boundary_data(gt_checker, decoder, posterior, bounds)
    frames_data.append(data)
    iou = compute_iou(data)
    print(f"  Seed {seed} iter {budget}: IoU={iou:.4f}")

    return frames_data


def create_combined_frame(all_seeds_data, iteration, seeds, save_path, dpi=200):
    """
    Create a combined frame showing all seeds in a 2x3 grid.

    Args:
        all_seeds_data: dict mapping seed -> list of frame data
        iteration: which iteration to show
        seeds: list of seeds in order
        save_path: path to save the combined frame
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, seed in enumerate(seeds):
        data = all_seeds_data[seed][iteration]
        plot_frame_for_seed(data, iteration, axes[idx], seed)

    # Add overall title
    fig.suptitle(f'Iteration {iteration}', fontsize=16, fontweight='bold', y=0.98)

    # Add a single legend for the whole figure
    GT_FILL = '#87CEEB'
    GT_BORDER = '#00CED1'
    PRED_FILL = '#FFB366'
    PRED_BORDER = '#FF6600'

    legend_elements = [
        mpatches.Patch(facecolor=GT_FILL, edgecolor=GT_BORDER,
                      linewidth=2, alpha=0.5, label='Ground Truth'),
        mpatches.Patch(facecolor=PRED_FILL, edgecolor=PRED_BORDER,
                      linewidth=2, linestyle='--', alpha=0.4, label='Predicted'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=11, framealpha=0.9, edgecolor='gray',
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def create_gif(frame_paths, gif_path, duration_ms=500):
    """Create a GIF from frame images."""
    frames = [Image.open(p) for p in frame_paths]

    if not frames:
        print("No frames found!")
        return

    # Hold the last frame longer
    durations = [duration_ms] * len(frames)
    durations[-1] = duration_ms * 4

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"\nSaved GIF: {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate combined GIF for multiple seeds")
    parser.add_argument("--seeds", type=int, nargs='+', default=[36, 56, 61, 62, 84, 153],
                       help="Seeds to include")
    parser.add_argument("--budget", type=int, default=32, help="Number of iterations")
    parser.add_argument("--frame_step", type=int, default=4,
                       help="Show frame every N iterations")
    parser.add_argument("--particles", type=int, default=103)
    parser.add_argument("--n_blobs", type=int, default=10)
    parser.add_argument("--gt_blobs", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--output_dir", type=str,
                       default="active_learning/images/toy_2d/combined_6seeds_32iter")
    parser.add_argument("--gif_duration", type=int, default=500)
    args = parser.parse_args()

    seeds = args.seeds
    budget = args.budget
    frame_step = args.frame_step

    print(f"Generating combined GIF for seeds: {seeds}")
    print(f"Budget: {budget}, Frame step: {frame_step}")
    print(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Run each seed and capture all frames
    all_seeds_data = {}
    for seed in seeds:
        print(f"\nRunning seed {seed}...")
        all_seeds_data[seed] = run_seed_and_capture_frames(
            seed, budget, args.particles, args.n_blobs, args.gt_blobs
        )

    # Determine which iterations to show
    iterations_to_show = list(range(0, budget + 1, frame_step))
    if budget not in iterations_to_show:
        iterations_to_show.append(budget)

    print(f"\nCreating combined frames for iterations: {iterations_to_show}")

    # Create combined frames
    frame_paths = []
    for iteration in iterations_to_show:
        save_path = os.path.join(args.output_dir, f"combined_iter_{iteration:02d}.png")
        create_combined_frame(all_seeds_data, iteration, seeds, save_path, args.dpi)
        frame_paths.append(save_path)
        print(f"  Saved: {save_path}")

    # Create GIF
    gif_path = os.path.join(args.output_dir, "combined_animation.gif")
    create_gif(frame_paths, gif_path, args.gif_duration)

    print(f"\nDone! Combined GIF saved to {gif_path}")


if __name__ == "__main__":
    main()
