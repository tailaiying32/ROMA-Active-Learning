
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.config import DEVICE, load_config, get_bounds_from_config
from active_learning.test.diagnostics.utils import load_diagnostic_model, get_diagnostic_grid
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.metrics import compute_reachability_metrics
from infer_params.training.dataset import LevelSetDataset
from active_learning.src.latent_oracle import LatentOracle

def check_optimization_dynamics(
    decoder, 
    gt_z: torch.Tensor, 
    ground_truth_params: dict,
    bounds: torch.Tensor,
    save_dir: Path,
    n_observations: int = 20
):
    """
    Verify if optimization can recover gt_z and if IoU improves.
    Mimics pipeline logic (BCE, standard optimizer).
    """
    print(f"\n--- Checking Optimization Dynamics ---")
    
    # 1. Setup Environment
    # Start from random point or prior (random point for harder test)
    z = torch.randn_like(gt_z, requires_grad=True, device=DEVICE)
    init_dist = (z.detach() - gt_z).norm().item()
    print(f"  Initial Latent Distance: {init_dist:.4f}")
    
    # Create evaluation grid for IoU
    gt_lower = ground_truth_params['box_lower']
    gt_upper = ground_truth_params['box_upper']
    grid = get_diagnostic_grid(resolution=10, bounds=(-np.pi, np.pi), device=DEVICE) # Use wide bounds
    
    # Create Oracle
    oracle = LatentOracle(decoder, gt_z, 4)
    
    # Sample Observations
    print(f"  Collecting {n_observations} random observations...")
    test_points = bounds[:, 0] + torch.rand(n_observations, 4, device=DEVICE) * (bounds[:, 1] - bounds[:, 0])
    
    # Oracle only supports single query
    outcomes = []
    for tp in test_points:
        outcomes.append(oracle.query(tp))
    outcomes = torch.tensor(outcomes, device=DEVICE)
    
    # Optimizer from pipeline
    optimizer = torch.optim.Adam([z], lr=0.05)
    tau = 0.2 # Standard pipeline value
    
    history = {
        'iter': [],
        'dist': [],
        'iou': [],
        'loss': [],
        'grad_norm': [],
        'grad_cos_sim': []
    }
    
    print("  Optimizing...")
    prev_grad = None
    
    for i in range(100):
        optimizer.zero_grad()
        
        # Pipeline-exact prediction
        pred_logits = LatentFeasibilityChecker.batched_logit_values(decoder, z.unsqueeze(0), test_points).squeeze(0)
        
        # Pipeline-exact Loss
        eps = 1e-6
        pred_probs = torch.sigmoid(pred_logits / tau).clamp(eps, 1 - eps)
        target_probs = (outcomes > 0).float().clamp(eps, 1 - eps)
        
        loss = -(target_probs * torch.log(pred_probs) + (1 - target_probs) * torch.log(1 - pred_probs)).mean()
        
        loss.backward()
        
        # Log Gradient Norm and Cosine Similarity
        current_grad = z.grad.clone()
        grad_norm = current_grad.norm().item()
        
        cos_sim = 0.0
        if prev_grad is not None:
             # Cosine similarity: (A . B) / (|A| * |B|)
             cos_sim = torch.nn.functional.cosine_similarity(current_grad.unsqueeze(0), prev_grad.unsqueeze(0)).item()
        else:
             cos_sim = 1.0 # First step is "consistent" with itself/init
             
        prev_grad = current_grad
        
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            dist = (z - gt_z).norm().item()
            iou, acc, f1, _ = compute_reachability_metrics(decoder, ground_truth_params, z.unsqueeze(0), grid)
            
            history['iter'].append(i)
            history['dist'].append(dist)
            history['iou'].append(iou)
            history['loss'].append(loss.item())
            history['grad_norm'].append(grad_norm)
            history['grad_cos_sim'].append(cos_sim)
            
        if i % 20 == 0:
            print(f"    Iter {i}: Loss={loss.item():.4f}, Dist={dist:.4f}, IoU={iou:.4f}, Grad={grad_norm:.4f}, CosSim={cos_sim:.4f}")
            
    # --- Analysis ---
    final_dist = history['dist'][-1]
    final_iou = history['iou'][-1]
    final_loss = history['loss'][-1]
    init_iou = history['iou'][0]
    avg_cos_sim = np.mean(history['grad_cos_sim'][1:]) # Skip first dummy value
    
    print(f"  Final: Dist={final_dist:.4f}, IoU={final_iou:.4f}")
    print(f"  AVG Gradient Cosine Similarity: {avg_cos_sim:.4f}")
    
    if avg_cos_sim > 0.5:
        print("  [PASS] Gradients are Consistent (High positive Cosine Similarity).")
    elif avg_cos_sim > 0:
        print("  [WARN] Gradients are somewhat noisy but generally positive.")
    else:
        print("  [FAIL] Gradients are Oscillating/Inconsistent (Negative Cosine Similarity).")

    # Hypothesis Check: Is z_opt 'better' than z_gt for these observations?
    with torch.no_grad():
        gt_logits = LatentFeasibilityChecker.batched_logit_values(decoder, gt_z.unsqueeze(0), test_points).squeeze(0)
        gt_probs = torch.sigmoid(gt_logits / tau).clamp(eps, 1 - eps)
        gt_loss = -(target_probs * torch.log(gt_probs) + (1 - target_probs) * torch.log(1 - gt_probs)).mean().item()
    
    print(f"  Loss Comparison: Optimized Loss={final_loss:.6f} vs GT Loss={gt_loss:.6f}")
    
    if final_loss < gt_loss:
        print("  [INSIGHT] Optimizer found a z with LOWER loss than Ground Truth!")
        print("            => The sparse observations are better explained by z_opt than z_gt (Ambiguity/Overfitting).")
    else:
        print("  [INSIGHT] Optimizer stuck in local variance (Higher loss than GT).")

    # Check 1: Did we learn the task (IoU)?
    if final_iou > init_iou:
        print("  [PASS] Optimization improved Reachability IoU.")
    else:
        print("  [FAIL] Optimization did not improve Reachability IoU.")
        
    # Check 2: Ambiguity Analysis
    if final_dist > init_dist and final_iou >= init_iou:
        print("  [INSIGHT] Latent Distance Increased but IoU Improved/Stable.")
        print("            => Latent Space is Ambiguous/Non-unique (Different z -> Same Feasibility).")
    elif final_dist < init_dist:
        print("  [INSIGHT] Latent Distance Reduced along with Optimization.")
        
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Metrics
    ax1 = axes[0]
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Latent Distance (L2)', color='tab:blue')
    ax1.plot(history['iter'], history['dist'], color='tab:blue', label='Latent Dist')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Reachability IoU', color='tab:green')
    ax2.plot(history['iter'], history['iou'], color='tab:green', linestyle='--', label='IoU')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax1.set_title('Metrics: Latent Error vs IoU')
    
    # Plot 2: Gradients
    ax3 = axes[1]
    ax3.plot(history['iter'], history['grad_norm'], color='tab:red', label='Gradient Norm')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_yscale('log')
    ax3.set_title('Optimization Gradient Magnitude')
    ax3.grid(True, alpha=0.3)
    
    # Plot 3: Gradient Consistency (Cosine Sim)
    ax4 = axes[2]
    ax4.plot(history['iter'], history['grad_cos_sim'], color='purple', label='Grad Cosine Sim')
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(1, color='green', linestyle=':', alpha=0.3)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cosine Similarity (t vs t-1)')
    ax4.set_ylim(-1.1, 1.1)
    ax4.set_title(f'Gradient Direction Consistency\n(Avg: {avg_cos_sim:.3f})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "optimization_metrics.png")
    plt.close()
    
    # Correlation Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(history['dist'], history['iou'], c=history['iter'], cmap='viridis')
    plt.colorbar(label='Iteration')
    plt.xlabel('Latent Distance to GT')
    plt.ylabel('Reachability IoU')
    plt.title('Correlation: Latent Error vs Performance')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "metric_correlation.png")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="active_learning/images/diagnostics/optimization")
    parser.add_argument("--sample-idx", type=int, default=42)
    args = parser.parse_args()
    
    save_dir = Path(args.out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Optimization Diagnostics...")
    config = load_config()
    decoder, train_config, embeddings = load_diagnostic_model(args.model)
    bounds = get_bounds_from_config(config, DEVICE)
    
    # Load GT Params for IoU
    dataset_path = 'models/training_data.npz' # Assumption
    if not os.path.exists(dataset_path):
         # Try relative to project root
         dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models/training_data.npz'))
         
    dataset = LevelSetDataset(dataset_path)
    idx, box_lower, box_upper, box_weights, presence, blob_params = dataset[args.sample_idx]
    
    ground_truth_params = {
        'box_lower': box_lower, 'box_upper': box_upper, 
        'box_weights': box_weights, 'presence': presence, 'blob_params': blob_params
    }
    
    gt_z = embeddings[args.sample_idx].clone()
    
    check_optimization_dynamics(decoder, gt_z, ground_truth_params, bounds, save_dir)

if __name__ == "__main__":
    main()
