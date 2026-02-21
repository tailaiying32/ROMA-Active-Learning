"""Test if z optimization improves REACHABILITY even if latent distance increases."""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.metrics import compute_reachability_metrics
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from infer_params.training.model import LevelSetDecoder
from infer_params.training.dataset import LevelSetDataset
from infer_params.training.level_set_torch import create_evaluation_grid


def load_all():
    ckpt = torch.load('models/best_model.pt', map_location=DEVICE)
    cfg = ckpt['config']
    model_cfg = cfg['model']
    emb = ckpt['embeddings']
    decoder = LevelSetDecoder(
        emb.shape[0], model_cfg['latent_dim'], model_cfg['hidden_dim'],
        model_cfg['num_blocks'], 18, 6
    )
    decoder.load_state_dict(ckpt['model_state_dict'])
    decoder = decoder.to(DEVICE).eval()
    return decoder, emb.to(DEVICE)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    decoder, embeddings = load_all()
    config = load_config()
    config['seed'] = 42
    
    # Load ground truth params
    dataset = LevelSetDataset('models/training_data.npz')
    idx, box_lower, box_upper, box_weights, presence, blob_params = dataset[42]
    ground_truth_params = {
        'box_lower': box_lower, 'box_upper': box_upper, 
        'box_weights': box_weights, 'presence': presence, 'blob_params': blob_params
    }
    
    sample_idx = 42
    gt_z = embeddings[sample_idx].clone()
    
    # Generate prior
    prior_gen = LatentPriorGenerator(config, decoder)
    prior = prior_gen.get_prior(gt_z)
    z = prior.mean.clone().requires_grad_(True)
    
    oracle = LatentOracle(decoder, gt_z, 4)
    bounds = get_bounds_from_config(config, DEVICE)
    
    # Create reachability evaluation grid
    gt_lower, gt_upper = decoder.decode_from_embedding(gt_z.unsqueeze(0))[:2]
    grid_resolution = config.get('metrics', {}).get('grid_resolution', 12)
    grid = create_evaluation_grid(gt_lower.squeeze(), gt_upper.squeeze(), resolution=grid_resolution, device=DEVICE)
    
    # Initial metrics
    init_dist = (z.detach() - gt_z).norm().item()
    init_iou, init_acc, _, _ = compute_reachability_metrics(decoder, ground_truth_params, z.unsqueeze(0), grid)
    print(f"Initial: dist={init_dist:.4f}, IoU={init_iou:.4f}, Acc={init_acc:.4f}")
    
    # Collect observations
    test_points = [bounds[:, 0] + torch.rand(4, device=DEVICE) * (bounds[:, 1] - bounds[:, 0]) for _ in range(20)]
    outcomes = [oracle.query(tp) for tp in test_points]
    test_points = torch.stack(test_points)
    outcomes = torch.tensor(outcomes, device=DEVICE)
    
    # Optimize z with BCE loss
    tau = config.get('bald', {}).get('tau', 0.2)
    optimizer = torch.optim.Adam([z], lr=0.05)
    
    print("\nOptimizing z with BCE loss on 20 observations...")
    for i in range(100):
        optimizer.zero_grad()
        pred_logits = LatentFeasibilityChecker.batched_logit_values(decoder, z.unsqueeze(0), test_points).squeeze(0)
        eps = 1e-6
        pred_probs = torch.sigmoid(pred_logits / tau).clamp(eps, 1 - eps)
        target_probs = torch.sigmoid(outcomes / tau).clamp(eps, 1 - eps)
        loss = -(target_probs * torch.log(pred_probs) + (1 - target_probs) * torch.log(1 - pred_probs)).mean()
        loss.backward()
        optimizer.step()
        
        if i % 25 == 0:
            curr_dist = (z.detach() - gt_z).norm().item()
            curr_iou, curr_acc, _, _ = compute_reachability_metrics(decoder, ground_truth_params, z.unsqueeze(0), grid)
            print(f"  Iter {i:3d}: loss={loss.item():.4f}, dist={curr_dist:.4f}, IoU={curr_iou:.4f}, Acc={curr_acc:.4f}")
    
    # Final metrics
    final_dist = (z.detach() - gt_z).norm().item()
    final_iou, final_acc, _, _ = compute_reachability_metrics(decoder, ground_truth_params, z.unsqueeze(0), grid)
    
    print(f"\nFinal: dist={final_dist:.4f}, IoU={final_iou:.4f}, Acc={final_acc:.4f}")
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    dist_status = "WORSE" if final_dist > init_dist else "BETTER"
    iou_status = "WORSE" if final_iou < init_iou else "BETTER"
    acc_status = "WORSE" if final_acc < init_acc else "BETTER"
    print(f"Latent dist: {init_dist:.4f} -> {final_dist:.4f} ({dist_status})")
    print(f"IoU:         {init_iou:.4f} -> {final_iou:.4f} ({iou_status})")
    print(f"Accuracy:    {init_acc:.4f} -> {final_acc:.4f} ({acc_status})")


if __name__ == "__main__":
    main()
