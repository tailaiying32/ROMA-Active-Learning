"""
2D Toy Animation: BALD + Projected SVGD on a non-convex feasibility region.

Generates individual frames for all iterations and combines them into a GIF.
No test markers (query points) are shown — only the feasibility boundaries.

Usage:
    python run_toy_2d_animation.py --particles 103 --shape random --seed 8
    python run_toy_2d_animation.py --particles 103 --shape random --seed 8 --output_dir active_learning/images/toy_2d/seed_8
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

# Import everything from the original demo
from run_toy_2d_demo import (
    Toy2DChecker,
    Toy2DDecoder,
    Toy2DOracle,
    Toy2DParticlePosterior,
    Toy2DSVGD,
    Toy2DBALD,
    compute_boundary_data,
    create_random_blob_gt,
    create_crescent_gt,
    create_pac_man_gt,
    create_irregular_blob_gt,
    create_two_islands_gt,
    create_star_gt,
    create_annulus_gt,
    create_snake_gt,
)


def compute_iou(data):
    """
    Compute IoU (Intersection over Union) between GT and predicted feasibility regions.

    Args:
        data: dict with 'Z_gt' and 'Z_pred' arrays

    Returns:
        iou: float, IoU score between 0 and 1
    """
    gt_mask = data['Z_gt'] > 0
    pred_mask = data['Z_pred'] > 0
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def plot_animation_frame(data, iteration, save_path, dpi=300):
    """Plot a single animation frame — boundaries only, no test markers."""
    fig, ax = plt.subplots(figsize=(5, 5))

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
                  linewidths=3, linestyles='solid', zorder=2)
    except Exception:
        pass

    # Predicted region
    pred_feasible = Z_pred > 0
    ax.contourf(X, Y, pred_feasible.astype(float), levels=[0.5, 1.5],
               colors=[PRED_FILL], alpha=0.4, zorder=3)
    try:
        ax.contour(X, Y, Z_pred, levels=[0], colors=[PRED_BORDER],
                  linewidths=3, linestyles='dashed', zorder=4)
    except Exception:
        pass

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=GT_FILL, edgecolor=GT_BORDER,
                      linewidth=2, alpha=0.5, label='Ground Truth'),
        mpatches.Patch(facecolor=PRED_FILL, edgecolor=PRED_BORDER,
                      linewidth=2, linestyle='--', alpha=0.4, label='Predicted'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.8, edgecolor='gray')

    # Clean formatting
    ax.set_title(f'Iteration {iteration}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('auto')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def run_active_learning(args, output_dir=None, compute_only_iou=False):
    """
    Run the active learning loop and optionally save animation frames.

    Args:
        args: parsed arguments
        output_dir: directory to save frames (None = don't save)
        compute_only_iou: if True, only compute final IoU without saving frames

    Returns:
        final_iou: IoU at the final iteration
    """
    # Set seeds
    gt_seed = args.seed
    learning_seed = args.seed + 1000
    torch.manual_seed(learning_seed)
    np.random.seed(learning_seed)

    # 1. Create GT
    if args.shape == "random":
        gt_checker = create_random_blob_gt(gt_seed, n_blobs=args.gt_blobs, device=DEVICE)
    else:
        shape_creators = {
            "crescent": create_crescent_gt,
            "pacman": create_pac_man_gt,
            "blob": create_irregular_blob_gt,
            "two_islands": create_two_islands_gt,
            "star": create_star_gt,
            "annulus": create_annulus_gt,
            "snake": create_snake_gt,
        }
        gt_checker = shape_creators[args.shape](DEVICE)

    oracle = Toy2DOracle(gt_checker)

    # 2. Create decoder and posterior
    n_blobs = args.n_blobs
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
        decoder.latent_dim, args.particles, prior_mean, prior_std, DEVICE
    )

    # 3. Create SVGD and BALD
    svgd = Toy2DSVGD(decoder, posterior, prior_mean, prior_std,
                     step_size=0.05, n_iters=50, device=DEVICE)
    bald = Toy2DBALD(decoder, posterior, tau=0.3, device=DEVICE)

    bounds = torch.tensor([[-1.2, 1.2], [-1.2, 1.2]], device=DEVICE)

    # 4. Run active learning loop
    query_history = []
    final_iou = 0.0

    if not compute_only_iou and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i in range(args.budget):
        test_point, score = bald.select_test(bounds)
        outcome = oracle.query(test_point)

        # Capture frame for every iteration
        if not compute_only_iou and output_dir:
            data = compute_boundary_data(gt_checker, decoder, posterior, bounds)
            iou = compute_iou(data)

            save_path = os.path.join(output_dir, f"iteration_{i:02d}.png")
            plot_animation_frame(data, i, save_path, args.dpi)

            status = "feasible" if outcome > 0.5 else "infeasible"
            print(f"  [{i+1:2d}] BALD={score:.4f}, {status}, IoU={iou:.4f}  -> {save_path}")

        # Update posterior
        query_history.append((test_point.clone(), outcome))
        svgd.add_observation(test_point, outcome)
        svgd.update()

    # Final iteration (iteration 20): show final posterior
    data = compute_boundary_data(gt_checker, decoder, posterior, bounds)
    final_iou = compute_iou(data)

    if not compute_only_iou and output_dir:
        save_path = os.path.join(output_dir, f"iteration_{args.budget:02d}.png")
        plot_animation_frame(data, args.budget, save_path, args.dpi)
        print(f"  [final] IoU={final_iou:.4f}  -> {save_path}")

    return final_iou


def create_gif(output_dir, budget=20, duration_ms=500):
    """Create a GIF from the individual frames."""
    frames = []
    for i in range(budget + 1):
        frame_path = os.path.join(output_dir, f"iteration_{i:02d}.png")
        if os.path.exists(frame_path):
            frames.append(Image.open(frame_path))

    if not frames:
        print("No frames found!")
        return

    gif_path = os.path.join(output_dir, "animation.gif")

    # Hold the last frame longer
    durations = [duration_ms] * len(frames)
    durations[-1] = duration_ms * 4  # Hold final frame 4x longer

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"\nSaved GIF: {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="2D Toy Animation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--particles", type=int, default=50)
    parser.add_argument("--n_blobs", type=int, default=10)
    parser.add_argument("--gt_blobs", type=int, default=7)
    parser.add_argument("--shape", type=str, default="random",
                       choices=["crescent", "pacman", "blob", "two_islands",
                                "star", "annulus", "snake", "random"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for frames and GIF")
    parser.add_argument("--gif_duration", type=int, default=500,
                       help="Duration per frame in ms for GIF")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"active_learning/images/toy_2d/seed_{args.seed}"

    print(f"2D Toy Animation: shape={args.shape}, seed={args.seed}, "
          f"budget={args.budget}, particles={args.particles}, n_blobs={args.n_blobs}")
    print(f"Output: {args.output_dir}")

    # Run active learning and save frames
    final_iou = run_active_learning(args, output_dir=args.output_dir)
    print(f"\nFinal IoU: {final_iou:.4f}")

    # Create GIF
    create_gif(args.output_dir, args.budget, args.gif_duration)

    print(f"\nDone! Frames and GIF saved to {args.output_dir}")


if __name__ == "__main__":
    main()
