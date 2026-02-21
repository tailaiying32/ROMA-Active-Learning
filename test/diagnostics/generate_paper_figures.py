"""
Generate publication-quality figures showing feasibility boundary convergence.

Creates high-resolution plots at key iterations (0, 5, 10, 20) showing:
- Ground truth feasible region (filled)
- Predicted boundary (line) converging over iterations
- Query points overlaid

Usage:
    python generate_paper_figures.py --pair 0 --budget 20
    python generate_paper_figures.py --pair "shoulder_flexion_r,shoulder_abduction_r" --budget 20
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.factory import build_learner
from active_learning.src.utils import load_decoder_model
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from infer_params.training.level_set_torch import create_evaluation_grid

# Output directory
PAPER_FIG_DIR = "active_learning/images/paper_figures"
os.makedirs(PAPER_FIG_DIR, exist_ok=True)


def compute_boundary_data(learner, pair_idx, resolution=100):
    """
    Compute GT and predicted feasibility for a single joint pair at high resolution.

    Returns dict with X, Y meshgrid and Z_gt, Z_pred feasibility values.
    """
    config = learner.config
    joint_names = config['prior']['joint_names']
    pairs = config['prior']['pair_names']
    bounds = learner.bounds

    pair = pairs[pair_idx]
    j1, j2 = pair[0], pair[1]
    idx1 = joint_names.index(j1)
    idx2 = joint_names.index(j2)

    # Get bounds for this pair
    b1 = bounds[idx1]
    b2 = bounds[idx2]

    # Create high-res grid
    x = torch.linspace(b1[0], b1[1], resolution, device=DEVICE)
    y = torch.linspace(b2[0], b2[1], resolution, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Fix other dimensions to center
    centers = (bounds[:, 0] + bounds[:, 1]) / 2

    # Create grid points
    grid = centers.unsqueeze(0).unsqueeze(0).repeat(resolution, resolution, 1)
    grid[:, :, idx1] = X
    grid[:, :, idx2] = Y
    flat_points = grid.reshape(-1, len(joint_names))

    # Evaluate GT
    with torch.no_grad():
        gt_h = learner.oracle.ground_truth_checker.logit_value(flat_points)
        Z_gt = gt_h.reshape(resolution, resolution).cpu().numpy()

        # Evaluate prediction (posterior mean)
        mean_z = learner.get_posterior().mean
        pred_h = LatentFeasibilityChecker.batched_logit_values(
            learner.decoder, mean_z.unsqueeze(0), flat_points
        )
        Z_pred = pred_h.squeeze(0).reshape(resolution, resolution).cpu().numpy()

    return {
        'X': X.cpu().numpy(),
        'Y': Y.cpu().numpy(),
        'Z_gt': Z_gt,
        'Z_pred': Z_pred,
        'j1': j1,
        'j2': j2,
        'idx1': idx1,
        'idx2': idx2,
        'extent': [b1[0].item(), b1[1].item(), b2[0].item(), b2[1].item()]
    }


def plot_boundary_figure(data, query_history, iteration, save_path,
                         show_queries=True, figsize=(5, 5), dpi=300):
    """
    Create a publication-quality boundary convergence figure.

    Style matches reference image:
    - GT: light blue semi-transparent fill with solid cyan border
    - Predicted: orange semi-transparent fill with dotted orange border

    Args:
        data: Dict from compute_boundary_data
        query_history: List of (point, outcome) tuples
        iteration: Current iteration number
        save_path: Where to save the figure
        show_queries: Whether to show query points
        figsize: Figure size in inches
        dpi: Resolution for saving
    """
    fig, ax = plt.subplots(figsize=figsize)

    X, Y = data['X'], data['Y']
    Z_gt, Z_pred = data['Z_gt'], data['Z_pred']

    # Colors matching reference image
    GT_FILL = '#87CEEB'      # Light blue fill
    GT_BORDER = '#00CED1'    # Cyan/turquoise border
    PRED_FILL = '#FFB366'    # Light orange fill
    PRED_BORDER = '#FF6600'  # Orange border

    # 1. Fill GT feasible region (light blue, semi-transparent) - draw first (bottom)
    gt_feasible = Z_gt > 0
    ax.contourf(X, Y, gt_feasible.astype(float), levels=[0.5, 1.5],
                colors=[GT_FILL], alpha=0.5, zorder=1)

    # 2. GT boundary (solid cyan line)
    try:
        ax.contour(X, Y, Z_gt, levels=[0], colors=[GT_BORDER],
                  linewidths=3, linestyles='solid', zorder=2)
    except:
        pass

    # 3. Fill Predicted feasible region (orange, semi-transparent) - on top of GT
    pred_feasible = Z_pred > 0
    ax.contourf(X, Y, pred_feasible.astype(float), levels=[0.5, 1.5],
                colors=[PRED_FILL], alpha=0.4, zorder=3)

    # 4. Predicted boundary (dashed orange line) - on top of everything
    try:
        ax.contour(X, Y, Z_pred, levels=[0], colors=[PRED_BORDER],
                  linewidths=3, linestyles='dashed', zorder=4)
    except:
        pass

    # 5. Query points
    if show_queries and query_history:
        idx1, idx2 = data['idx1'], data['idx2']
        for qpt, outcome in query_history:
            if outcome > 0.5:
                ax.plot(qpt[idx1], qpt[idx2], 'o', color='#27ae60',
                       markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                       zorder=10)
            else:
                ax.plot(qpt[idx1], qpt[idx2], 'X', color='#e74c3c',
                       markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                       zorder=10)

    # 6. Clean, minimal formatting - no axes clutter
    ax.set_title(f'Iteration {iteration}', fontsize=14, fontweight='bold', pad=10)

    # Remove all axes clutter, make square
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect('auto')  # Fill square space

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_figure(all_data, save_path, figsize=(12, 4), dpi=300):
    """
    Create a combined 1x4 figure showing all iterations.
    Style matches reference image.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    iterations = sorted(all_data.keys())

    # Colors matching reference image
    GT_FILL = '#87CEEB'      # Light blue fill
    GT_BORDER = '#00CED1'    # Cyan/turquoise border
    PRED_FILL = '#FFB366'    # Light orange fill
    PRED_BORDER = '#FF6600'  # Orange border

    for ax_idx, iteration in enumerate(iterations):
        ax = axes[ax_idx]
        data, query_history = all_data[iteration]

        X, Y = data['X'], data['Y']
        Z_gt, Z_pred = data['Z_gt'], data['Z_pred']

        # Fill GT feasible region (light blue) - draw first (bottom)
        gt_feasible = Z_gt > 0
        ax.contourf(X, Y, gt_feasible.astype(float), levels=[0.5, 1.5],
                   colors=[GT_FILL], alpha=0.5, zorder=1)

        # GT boundary (solid cyan)
        try:
            ax.contour(X, Y, Z_gt, levels=[0], colors=[GT_BORDER],
                      linewidths=2.5, linestyles='solid', zorder=2)
        except:
            pass

        # Fill Predicted feasible region (orange) - on top of GT
        pred_feasible = Z_pred > 0
        ax.contourf(X, Y, pred_feasible.astype(float), levels=[0.5, 1.5],
                   colors=[PRED_FILL], alpha=0.4, zorder=3)

        # Predicted boundary (dashed orange) - on top of everything
        try:
            ax.contour(X, Y, Z_pred, levels=[0], colors=[PRED_BORDER],
                      linewidths=2.5, linestyles='dashed', zorder=4)
        except:
            pass

        # Query points
        if query_history:
            idx1, idx2 = data['idx1'], data['idx2']
            for qpt, outcome in query_history:
                if outcome > 0.5:
                    ax.plot(qpt[idx1], qpt[idx2], 'o', color='#27ae60',
                           markersize=6, markeredgecolor='white', markeredgewidth=1)
                else:
                    ax.plot(qpt[idx1], qpt[idx2], 'X', color='#e74c3c',
                           markersize=6, markeredgecolor='white', markeredgewidth=1)

        ax.set_title(f'Iteration {iteration}', fontsize=12, fontweight='bold', pad=8)

        # Remove all axes clutter, make square
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_aspect('auto')  # Fill square space

    # Single legend for entire figure (matching reference style)
    legend_elements = [
        mpatches.Patch(facecolor=GT_FILL, edgecolor=GT_BORDER,
                      linewidth=2, alpha=0.5, label='Actual User Reachability'),
        mpatches.Patch(facecolor=PRED_FILL, edgecolor=PRED_BORDER,
                      linewidth=2, linestyle='--', alpha=0.4, label='Estimated Reachability'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
                  markersize=8, label='Feasible Query'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#e74c3c',
                  markersize=8, label='Infeasible Query'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined figure: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures for boundary convergence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--pair", type=str, default="0",
                       help="Joint pair index (0-5) or names like 'shoulder_flexion_r,shoulder_abduction_r'")
    parser.add_argument("--resolution", type=int, default=150,
                       help="Grid resolution for boundary computation")
    parser.add_argument("--iterations", type=str, default="0,5,10,20",
                       help="Comma-separated iteration numbers to plot")
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Parse iterations
    plot_iterations = [int(x) for x in args.iterations.split(',')]
    max_iter = max(plot_iterations)
    if args.budget < max_iter:
        args.budget = max_iter
        print(f"Adjusted budget to {args.budget} to cover all requested iterations")

    # Load config
    config = load_config(os.path.join(os.path.dirname(__file__), '../../configs/latent.yaml'))
    config['stopping']['budget'] = args.budget

    # Parse pair argument
    pairs = config['prior']['pair_names']
    joint_names = config['prior']['joint_names']

    if args.pair.isdigit():
        pair_idx = int(args.pair)
    else:
        # Parse as "joint1,joint2"
        j1, j2 = args.pair.split(',')
        pair_idx = None
        for i, p in enumerate(pairs):
            if (p[0] == j1 and p[1] == j2) or (p[0] == j2 and p[1] == j1):
                pair_idx = i
                break
        if pair_idx is None:
            print(f"Could not find pair {args.pair}. Available pairs:")
            for i, p in enumerate(pairs):
                print(f"  {i}: {p[0]}, {p[1]}")
            return

    print(f"Using pair {pair_idx}: {pairs[pair_idx]}")
    print(f"Will plot iterations: {plot_iterations}")

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading model...")
    decoder, embeddings, _ = load_decoder_model(args.model, DEVICE)

    # Select GT user (same logic as diagnostic script)
    prior_gen = LatentPriorGenerator(config, decoder)

    from infer_params.training.level_set_torch import evaluate_level_set_batched
    joint_limits = prior_gen.anatomical_limits
    grid_lowers = [joint_limits[j][0] for j in prior_gen.joint_names]
    grid_uppers = [joint_limits[j][1] for j in prior_gen.joint_names]
    check_grid = create_evaluation_grid(
        torch.tensor(grid_lowers, device=DEVICE),
        torch.tensor(grid_uppers, device=DEVICE),
        resolution=8, device=DEVICE
    )

    print("Selecting ground truth user...")
    gt_z = None
    for attempt in range(50):
        idx = np.random.randint(0, len(embeddings))
        z = embeddings[idx]

        with torch.no_grad():
            decoded = decoder.decode_from_embedding(z.unsqueeze(0))
            lower, upper, weights, pres, blob = decoded
            logits = evaluate_level_set_batched(check_grid, lower, upper, weights, pres, blob)
            feasible_frac = (logits > 0).float().mean().item()

        if feasible_frac > 0.33:
            gt_z = z.clone()
            print(f"Selected user {idx} (feasible volume: {feasible_frac:.1%})")
            break

    if gt_z is None:
        gt_z = embeddings[0].clone()
        print("Warning: Using first embedding as GT")

    # Setup learner
    prior = prior_gen.get_prior(gt_z, embeddings=embeddings)
    posterior = LatentUserDistribution(
        latent_dim=prior.latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )

    bounds = get_bounds_from_config(config, DEVICE)
    oracle = LatentOracle(decoder, gt_z, bounds.shape[0])

    learner = build_learner(
        decoder=decoder, prior=prior, posterior=posterior,
        oracle=oracle, bounds=bounds, config=config, embeddings=embeddings
    )

    # Run and collect data at key iterations
    all_data = {}
    query_history = []

    print("\nRunning active learning loop...")
    for i in range(args.budget + 1):
        if i in plot_iterations:
            print(f"  Capturing iteration {i}...")
            data = compute_boundary_data(learner, pair_idx, resolution=args.resolution)
            all_data[i] = (data, list(query_history))

            # Save individual figure
            save_path = os.path.join(PAPER_FIG_DIR, f"boundary_iter_{i:02d}.png")
            plot_boundary_figure(data, query_history, i, save_path, dpi=args.dpi)

        if i < args.budget:
            learner.step(verbose=False)
            if learner.results:
                last = learner.results[-1]
                pt = last.test_point.cpu().numpy() if torch.is_tensor(last.test_point) else last.test_point
                query_history.append((pt, last.outcome))

            if (i + 1) % 5 == 0:
                print(f"  Completed iteration {i + 1}/{args.budget}")

    # Save combined figure
    combined_path = os.path.join(PAPER_FIG_DIR, "boundary_convergence_combined.png")
    plot_combined_figure(all_data, combined_path, dpi=args.dpi)

    print(f"\nDone! Figures saved to {PAPER_FIG_DIR}")


if __name__ == "__main__":
    main()
