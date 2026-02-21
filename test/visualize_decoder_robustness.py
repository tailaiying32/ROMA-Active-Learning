"""
Decoder Robustness Visualization Script.

This script evaluates the robustness of the LevelSetDecoder by visualizing:
1. Latent space interpolation (distant vs close pairs)
2. Joint limit smoothness across interpolation paths
3. Dense-grid IoU metrics to assess categorical stability

Usage:
  mamba activate active_learning
  python active_learning/test/visualize_decoder_robustness.py --model models/best_model.pt --dataset models/training_data.npz
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import DEVICE
from infer_params.training.model import LevelSetDecoder
from infer_params.training.level_set_torch import evaluate_level_set_batched, create_evaluation_grid
from active_learning.test.visualization_utils import plot_latent_comparison, decode_to_metadata
from active_learning.src.config import load_config

def load_decoder_and_embeddings(checkpoint_path: str, device: str = DEVICE):
    """Load the LevelSetDecoder model and trained embeddings."""
    print(f"Loading decoder model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    train_config = checkpoint['config']
    model_cfg = train_config['model']
    embeddings = checkpoint['embeddings']
    num_samples, latent_dim = embeddings.shape

    model = LevelSetDecoder(
        num_samples=num_samples,
        latent_dim=latent_dim,
        hidden_dim=model_cfg['hidden_dim'],
        num_blocks=model_cfg['num_blocks'],
        num_slots=model_cfg.get('num_slots', 18),
        params_per_slot=model_cfg.get('params_per_slot', 6),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("  Model loaded.")
    return model, embeddings.to(device)

def get_interesting_pairs(embeddings: torch.Tensor, n_pairs: int = 1, shuffle: bool = False) -> Dict[str, List[Tuple[int, int]]]:
    """Find pairs of embeddings that are distant and close."""
    n_samples = embeddings.shape[0]
    print(f"Computing distance matrix for {n_samples} embeddings...")
    dist_matrix = torch.cdist(embeddings, embeddings)
    
    # --- Distant Pairs ---
    print("Finding distant pairs...")
    # Mask diagonal (self-distance is 0, we want max, so 0 is fine)
    dist_matrix.fill_diagonal_(0.0)
    
    # Use topk instead of sorting everything
    # We need enough candidates to find disjoint pairs. 
    # Heuristic: n_pairs * 10 should be plenty, but let's grab n_pairs * n_samples to be safe/lazy 
    # actually n_pairs * 50 is likely enough unless we have a star graph
    k_candidates = min(n_pairs * 100, n_samples * n_samples)
    vals, indices = torch.topk(dist_matrix.view(-1), k=k_candidates, largest=True)
    
    # Move to CPU for fast iteration
    indices_cpu = indices.cpu().numpy()
    if shuffle:
        np.random.shuffle(indices_cpu)
    
    distant_pairs = []
    seen = set()
    for idx in indices_cpu:
        i, j = int(idx // n_samples), int(idx % n_samples)
        if i < j and i not in seen and j not in seen:
            distant_pairs.append((i, j))
            seen.update([i, j])
        if len(distant_pairs) >= n_pairs:
            break
            
    # --- Close Pairs ---
    print("Finding close pairs...")
    # We want min non-zero. Set diagonal to inf.
    dist_matrix.fill_diagonal_(float('inf'))
    
    # Use topk with largest=False for smallest values
    vals, indices = torch.topk(dist_matrix.view(-1), k=k_candidates, largest=False)
    
    # Move to CPU
    indices_cpu = indices.cpu().numpy()
    if shuffle:
        np.random.shuffle(indices_cpu)
    
    close_pairs = []
    seen = set()
    for idx in indices_cpu:
        i, j = int(idx // n_samples), int(idx % n_samples)
        if i < j and (i, j) not in distant_pairs and i not in seen and j not in seen:
            close_pairs.append((i, j))
            seen.update([i, j])
        if len(close_pairs) >= n_pairs:
            break
            
    return {"distant": distant_pairs, "close": close_pairs}

def compute_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    if union == 0:
        return 1.0
    return (intersection / union).item()

def visualize_interpolation(
    model: LevelSetDecoder, 
    embeddings: torch.Tensor, 
    pair: Tuple[int, int], 
    pair_name: str, 
    save_dir: Path,
    n_steps: int = 20,
    grid_res: int = 8
):
    """Visualize interpolation metrics for a specific pair of latent codes."""
    print(f"  Visualizing interpolation for {pair_name} ({n_steps} steps)...")
    z0, z1 = embeddings[pair[0]], embeddings[pair[1]]
    alphas = np.linspace(0, 1, n_steps)
    
    # Path in latent space
    z_path = torch.stack([(1-a)*z0 + a*z1 for a in alphas]) # (n_steps, latent_dim)
    
    # 1. Decode joint limits
    with torch.no_grad():
        lowers, uppers, _, _, _ = model.decode_from_embedding(z_path)
        lowers = lowers.cpu().numpy() # (n_steps, 4)
        uppers = uppers.cpu().numpy() # (n_steps, 4)
        
    # 2. Compute Dense-Grid IoU
    # Define grid bounds (broad range for all joints)
    grid_lower = torch.tensor([-3.14, -3.14, -3.14, -3.14], device=DEVICE)
    grid_upper = torch.tensor([3.14, 3.14, 3.14, 3.14], device=DEVICE)
    print(f"    Creating evaluation grid (res={grid_res})...")
    grid = create_evaluation_grid(grid_lower, grid_upper, resolution=grid_res, device=DEVICE)
    
    iou_source = []
    iou_target = []
    
    print(f"    Computing IoU over {len(z_path)} steps...")
    with torch.no_grad():
        # Compute masks for all steps
        # This might be memory intensive if N is large, so we do it in steps
        # evaluate_level_set_batched returns (B, N)
        logits = evaluate_level_set_batched(
            grid, 
            *model.decode_from_embedding(z_path)
        )
        masks = (logits >= 0) # (n_steps, n_points)
        
        m_start = masks[0]
        m_end = masks[-1]
        
        for i, m in enumerate(masks):
            if i % 5 == 0:
                 print(f"      Step {i}/{len(masks)}")
            iou_source.append(compute_iou(m, m_start))
            iou_target.append(compute_iou(m, m_end))

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Joint Limits Smoothness
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    for i in range(4):
        ax1.plot(alphas, lowers[:, i], color=colors[i], linestyle='--', label=f'J{i} Lower')
        ax1.plot(alphas, uppers[:, i], color=colors[i], linestyle='-', label=f'J{i} Upper')
    ax1.set_title(f'Joint Limit Interpolation ({pair_name}: {pair[0]} -> {pair[1]})')
    ax1.set_xlabel('Alpha (Interpolation Factor)')
    ax1.set_ylabel('Joint Angle (rad)')
    ax1.legend(ncol=2, fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: IoU Smoothness
    ax2 = axes[1]
    ax2.plot(alphas, iou_source, 'b-o', label='IoU with Source (z0)')
    ax2.plot(alphas, iou_target, 'r-s', label='IoU with Target (z1)')
    ax2.set_title(f'Dense-Grid IoU Stability ({grid_res}^4 points)')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('IoU')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"robustness_{pair_name}.png"
    plt.savefig(save_dir / filename)
    plt.close()
    print(f"  Saved visualization: {filename}")
    
    # 3. Plot Detailed Start/End Comparison
    print(f"    Plotting latent neighborhood comparison for {pair_name}...")
    try:
        # Load config to get joint names (or infer default)
        config = load_config()
        joint_names = config.get('prior', {}).get('joint_names', ['joint_0', 'joint_1', 'joint_2', 'joint_3'])
        
        # Define limits (using standard range if not in config)
        limits = {j: (-3.5, 3.5) for j in joint_names}
        
        comp_filename = f"robustness_struct_{pair_name}.png"
        comp_save_path = save_dir / comp_filename
        
        # Reuse plot_latent_comparison but change titles in it? 
        # Actually plot_latent_comparison hardcodes "Ground Truth" and "Prediction".
        # We might want "Source (z0)" and "Target (z1)".
        # For quickly utilizing existing code, we can just use it and accept the labels, 
        # or better, update visualize_latent_comparison to accept labels.
        # But since I can't easily change the library code right now without another step, 
        # I will just call it. Usage: plot_latent_comparison(z_gt, z_pred, ...)
        # So "Ground Truth" -> z0, "Prediction" -> z1.
        
        plot_latent_comparison(
            z0, 
            z1, 
            model, 
            joint_names, 
            limits, 
            str(comp_save_path),
            labels=("Source (z0)", "Target (z1)")
        )
        print(f"  Saved structure visualization: {comp_filename}")
        
    except Exception as e:
        print(f"Failed to plot structure comparison: {e}")

def plot_latent_scatter(
    pca_embeddings: np.ndarray,
    pair: Tuple[int, int],
    pair_name: str,
    original_dist: float,
    save_dir: Path
):
    """Plot global latent space with the selected pair highlighted."""
    z0_proj = pca_embeddings[pair[0]]
    z1_proj = pca_embeddings[pair[1]]
    
    plt.figure(figsize=(10, 8))
    
    # Plot all points in background
    plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c='lightgray', s=10, alpha=0.5, label='All Embeddings')
    
    # Plot path between points
    plt.plot([z0_proj[0], z1_proj[0]], [z0_proj[1], z1_proj[1]], 'k--', linewidth=1, alpha=0.7)
    
    # Plot selected points
    plt.scatter(z0_proj[0], z0_proj[1], c='blue', s=100, edgecolors='black', label=f'Source (z0)', zorder=5)
    plt.scatter(z1_proj[0], z1_proj[1], c='red', s=100, edgecolors='black', marker='s', label=f'Target (z1)', zorder=5)
    
    plt.title(f"Latent Space (PCA Projection)\nPair: {pair_name} | Distance: {original_dist:.4f}")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"robustness_scatter_{pair_name}.png"
    plt.savefig(save_dir / filename, dpi=150)
    plt.close()
    print(f"  Saved scatter visualization: {filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--dataset", type=str, default="models/training_data.npz")
    parser.add_argument("--out-dir", type=str, default="active_learning/images/decoder_robustness")
    parser.add_argument("--n-steps", type=int, default=21)
    parser.add_argument("--grid-res", type=int, default=10)
    parser.add_argument("--random", action="store_true", help="Shuffle candidate pairs to get different results each run")
    args = parser.parse_args()
    
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model
    model, embeddings = load_decoder_and_embeddings(args.model)
    
    # Compute Global PCA for scatterplots
    print("Computing global PCA...")
    # Center the data
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    # Use torch.pca_lowrank for SVD-based PCA
    U, S, V = torch.pca_lowrank(centered, q=2)
    # Project: centered @ V[:, :2]
    pca_proj = torch.matmul(centered, V[:, :2]).cpu().numpy()
    
    # 2. Find pairs
    pairs_dict = get_interesting_pairs(embeddings, n_pairs=2, shuffle=args.random)
    
    # 3. Process each pair type
    for ptype, pairs in pairs_dict.items():
        print(f"Analyzing {ptype} pairs...")
        for i, pair in enumerate(pairs):
            print(f"  Pair {i+1}/{len(pairs)}: {pair}")
            visualize_interpolation(
                model, 
                embeddings, 
                pair, 
                f"{ptype}_{i}", 
                save_dir,
                n_steps=args.n_steps,
                grid_res=args.grid_res
            )
            
            # Scatterplot
            dist = torch.norm(embeddings[pair[0]] - embeddings[pair[1]]).item()
            plot_latent_scatter(
                pca_proj,
                pair,
                f"{ptype}_{i}",
                dist,
                save_dir
            )

if __name__ == "__main__":
    main()
