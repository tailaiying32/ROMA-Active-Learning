"""Quick seed search: finds seeds with high final IoU and irregular shapes."""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from run_toy_2d_animation import run_active_learning, compute_iou
from run_toy_2d_demo import create_random_blob_gt, compute_boundary_data, Toy2DDecoder, Toy2DParticlePosterior
from active_learning.src.config import DEVICE

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)


def compute_shape_complexity(gt_checker, bounds, resolution=200):
    """
    Compute a complexity score for the GT shape.
    Higher scores indicate more irregular/non-convex shapes.

    Metrics:
    - Perimeter to area ratio (higher = more irregular)
    - Non-convexity (area of convex hull - area of shape)
    """
    x = torch.linspace(bounds[0, 0], bounds[0, 1], resolution, device=gt_checker.device)
    y = torch.linspace(bounds[1, 0], bounds[1, 1], resolution, device=gt_checker.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    with torch.no_grad():
        Z_gt = gt_checker.logit_value(points).view(resolution, resolution).cpu().numpy()

    gt_mask = Z_gt > 0

    # Area (number of feasible pixels)
    area = gt_mask.sum()
    if area < 100:  # Too small
        return 0.0

    # Perimeter (count boundary pixels using gradient)
    from scipy import ndimage
    # Use morphological gradient to find boundary
    dilated = ndimage.binary_dilation(gt_mask)
    eroded = ndimage.binary_erosion(gt_mask)
    perimeter = (dilated & ~eroded).sum()

    # Perimeter-to-area ratio (circularity inverse)
    # A circle has the minimum perimeter for a given area
    # More irregular shapes have higher ratios
    circularity_inv = perimeter / np.sqrt(area) if area > 0 else 0

    # Compute convex hull area to measure non-convexity
    try:
        from scipy.spatial import ConvexHull
        yx = np.array(np.where(gt_mask)).T  # Get coordinates of feasible points
        if len(yx) > 10:
            hull = ConvexHull(yx)
            convex_area = hull.volume  # In 2D, volume is area
            non_convexity = (convex_area - area) / convex_area if convex_area > 0 else 0
        else:
            non_convexity = 0
    except:
        non_convexity = 0

    # Combined complexity score
    # Weight both metrics
    complexity = circularity_inv * 0.5 + non_convexity * 100

    return complexity, circularity_inv, non_convexity, area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--particles", type=int, default=103)
    parser.add_argument("--iou_threshold", type=float, default=0.80)
    parser.add_argument("--complexity_threshold", type=float, default=15.0)
    parser.add_argument("--quick", action="store_true", help="Only check shape complexity, skip IoU")
    args_search = parser.parse_args()

    bounds = torch.tensor([[-1.2, 1.2], [-1.2, 1.2]], device=DEVICE)

    # Create a namespace matching what run_active_learning expects
    class Args:
        budget = 20
        particles = args_search.particles
        n_blobs = 10
        gt_blobs = 7
        shape = "random"
        dpi = 300
        seed = 0

    run_args = Args()

    results = []
    for seed in range(args_search.start, args_search.end):
        run_args.seed = seed

        # First check shape complexity (fast)
        gt_checker = create_random_blob_gt(seed, n_blobs=7, device=DEVICE)
        complexity_info = compute_shape_complexity(gt_checker, bounds)
        complexity, circ_inv, non_convex, area = complexity_info

        if args_search.quick:
            marker = " <-- COMPLEX" if complexity >= args_search.complexity_threshold else ""
            print(f"Seed {seed:4d}: complexity={complexity:.2f} (circ_inv={circ_inv:.2f}, non_convex={non_convex:.3f}, area={area}){marker}", flush=True)
            results.append((seed, 0, complexity))
            continue

        # Skip simple shapes
        if complexity < args_search.complexity_threshold:
            print(f"Seed {seed:4d}: complexity={complexity:.2f} - SKIP (too simple)", flush=True)
            continue

        # Run full AL to get IoU
        try:
            iou = run_active_learning(run_args, output_dir=None, compute_only_iou=True)
            results.append((seed, iou, complexity))

            good_iou = iou >= args_search.iou_threshold
            marker = " <-- GOOD!" if good_iou else ""
            print(f"Seed {seed:4d}: IoU={iou:.4f}, complexity={complexity:.2f}{marker}", flush=True)
        except Exception as e:
            print(f"Seed {seed:4d}: ERROR - {e}", flush=True)

    # Summary
    if not args_search.quick:
        # Filter for good IoU and sort by complexity
        good_results = [(s, i, c) for s, i, c in results if i >= args_search.iou_threshold]
        good_results.sort(key=lambda x: -x[2])  # Sort by complexity descending

        print(f"\n=== Best seeds (IoU >= {args_search.iou_threshold}, sorted by complexity) ===", flush=True)
        for seed, iou, complexity in good_results[:10]:
            print(f"  Seed {seed:4d}: IoU={iou:.4f}, complexity={complexity:.2f}", flush=True)
    else:
        results.sort(key=lambda x: -x[2])
        print(f"\n=== Most complex shapes ===", flush=True)
        for seed, _, complexity in results[:20]:
            print(f"  Seed {seed:4d}: complexity={complexity:.2f}", flush=True)


if __name__ == "__main__":
    main()
