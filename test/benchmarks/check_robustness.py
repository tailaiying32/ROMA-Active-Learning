
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.config import DEVICE, load_config
from active_learning.test.diagnostics.utils import load_diagnostic_model, get_diagnostic_grid
from infer_params.training.level_set_torch import evaluate_level_set_batched
from active_learning.test.visualization_utils import plot_latent_comparison

def get_interesting_pairs(embeddings: torch.Tensor, n_pairs: int = 1, shuffle: bool = False) -> Dict[str, List[Tuple[int, int]]]:
    """Find pairs of embeddings that are distant and close."""
    n_samples = embeddings.shape[0]
    print(f"Computing distance matrix for {n_samples} embeddings...")
    dist_matrix = torch.cdist(embeddings, embeddings)
    
    # --- Distant Pairs ---
    print("Finding distant pairs...")
    dist_matrix.fill_diagonal_(0.0)
    
    # Use topk to find largest distances
    k_candidates = min(n_pairs * 100, n_samples * n_samples)
    vals, indices = torch.topk(dist_matrix.view(-1), k=k_candidates, largest=True)
    
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
    dist_matrix.fill_diagonal_(float('inf'))
    vals, indices = torch.topk(dist_matrix.view(-1), k=k_candidates, largest=False)
    
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
    intersection = (mask1 & mask2).float().sum()
    union = (mask1 | mask2).float().sum()
    if union == 0:
        return 1.0
    return (intersection / union).item()

def visualize_latent_interpolation(
    decoder, 
    embeddings: torch.Tensor, 
    pair: Tuple[int, int], 
    pair_name: str, 
    save_dir: Path,
    n_steps: int = 20,
    grid_res: int = 8
):
    """Visualize interpolation metrics for a specific pair, including Euclidean distance in title."""
    z0, z1 = embeddings[pair[0]], embeddings[pair[1]]
    dist = torch.norm(z0 - z1).item()
    
    print(f"  Visualizing interpolation for {pair_name} (Dist: {dist:.4f})...")
    
    alphas = np.linspace(0, 1, n_steps)
    z_path = torch.stack([(1-a)*z0 + a*z1 for a in alphas]) # (n_steps, latent_dim)
    
    # 1. Decode joint limits
    with torch.no_grad():
        lowers, uppers, _, _, _ = decoder.decode_from_embedding(z_path)
        lowers = lowers.cpu().numpy()
        uppers = uppers.cpu().numpy()
        
    # 2. Compute Dense-Grid IoU
    grid = get_diagnostic_grid(resolution=grid_res, device=DEVICE)
    
    iou_source = []
    iou_target = []
    
    print(f"    Computing IoU over {len(z_path)} steps...")
    with torch.no_grad():
        logits = evaluate_level_set_batched(
            grid, 
            *decoder.decode_from_embedding(z_path)
        )
        masks = (logits >= 0)
        
        m_start = masks[0]
        m_end = masks[-1]
        
        for i, m in enumerate(masks):
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
    ax1.set_title(f'Joint Limit Interpolation\n{pair_name} (L2 Dist: {dist:.4f})')
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Joint Angle (rad)')
    # ax1.legend(ncol=2, fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: IoU Smoothness
    ax2 = axes[1]
    ax2.plot(alphas, iou_source, 'b-o', label='IoU with Source')
    ax2.plot(alphas, iou_target, 'r-s', label='IoU with Target')
    ax2.set_title(f'Dense-Grid IoU Stability')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('IoU')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"robustness_{pair_name}.png"
    plt.savefig(save_dir / filename)
    plt.close()
    
    # 3. Structure Comparison (Source vs Target)
    try:
        config = load_config()
        joint_names = config.get('prior', {}).get('joint_names', ['joint_0', 'joint_1', 'joint_2', 'joint_3'])
        limits = {j: (-3.5, 3.5) for j in joint_names}
        
        comp_filename = f"robustness_struct_{pair_name}.png"
        comp_save_path = save_dir / comp_filename
        
        plot_latent_comparison(
            z0, z1, decoder, joint_names, limits, str(comp_save_path),
            labels=(f"Source (Idx {pair[0]})", f"Target (Idx {pair[1]})")
        )
    except Exception as e:
        print(f"Failed to plot structure comparison: {e}")


def visualize_latent_scatter(embeddings: torch.Tensor, save_dir: Path):
    """Plot global PCA of latent space."""
    print("Computing global PCA for scatter plot...")
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    U, S, V = torch.pca_lowrank(centered, q=2)
    pca_proj = torch.matmul(centered, V[:, :2]).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_proj[:, 0], pca_proj[:, 1], c='blue', alpha=0.5, s=20)
    plt.title(f"Latent Space PCA Projection\n({embeddings.shape[0]} Samples)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_dir / "latent_space_pca.png")
    plt.close()
    print("Saved latent space PCA plot.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="active_learning/images/decoder_robustness")
    parser.add_argument("--n-pairs", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    decoder, config, embeddings = load_diagnostic_model(args.model)
    
    # 1. Global Scatter
    visualize_latent_scatter(embeddings, save_dir)
    
    # 2. Pair Interpolation
    pairs_dict = get_interesting_pairs(embeddings, n_pairs=args.n_pairs, shuffle=args.shuffle)
    
    for ptype, pairs in pairs_dict.items():
        print(f"\nAnalyzing {ptype} pairs...")
        for i, pair in enumerate(pairs):
            visualize_latent_interpolation(
                decoder, embeddings, pair, f"{ptype}_{i}", save_dir
            )

if __name__ == "__main__":
    main()
