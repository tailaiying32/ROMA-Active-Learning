"""Test very low kl_weight with wide prior to see if VI can find GT."""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_variational_inference import LatentVariationalInference
from infer_params.training.model import LevelSetDecoder


def load_model():
    ckpt = torch.load('models/best_model.pt', map_location=DEVICE)
    cfg = ckpt['config']
    model_cfg = cfg['model']
    emb = ckpt['embeddings']
    decoder = LevelSetDecoder(
        emb.shape[0], model_cfg['latent_dim'], model_cfg['hidden_dim'], 
        model_cfg['num_blocks'], model_cfg.get('num_slots', 18), 
        model_cfg.get('params_per_slot', 6)
    )
    decoder.load_state_dict(ckpt['model_state_dict'])
    decoder = decoder.to(DEVICE)
    decoder.eval()
    return decoder, emb.to(DEVICE), model_cfg['latent_dim']


def test_direct_optimization():
    """Test: What if we optimize z directly toward GT using observations?"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    decoder, embeddings, latent_dim = load_model()
    config = load_config()
    config['seed'] = 42
    
    sample_idx = 42
    gt_z = embeddings[sample_idx].clone()
    
    prior_gen = LatentPriorGenerator(config, decoder)
    prior = prior_gen.get_prior(gt_z)
    
    # Start from prior mean
    z = prior.mean.clone().requires_grad_(True)
    
    oracle = LatentOracle(decoder, gt_z, 4)
    bounds = get_bounds_from_config(config, DEVICE)
    
    print(f"Initial dist to GT: {(z.detach() - gt_z).norm().item():.4f}")
    
    # Collect observations
    test_points = []
    outcomes = []
    for q in range(10):
        tp = bounds[:, 0] + torch.rand(4, device=DEVICE) * (bounds[:, 1] - bounds[:, 0])
        outcome = oracle.query(tp)
        test_points.append(tp)
        outcomes.append(outcome)
    
    test_points = torch.stack(test_points)
    outcomes = torch.tensor(outcomes, device=DEVICE)
    
    print(f"\nDirect optimization of z to match observations:")
    
    # Optimize z directly (no stochastic sampling)
    tau = config.get('bald', {}).get('tau', 0.2)
    optimizer = torch.optim.Adam([z], lr=0.05)
    
    for i in range(100):
        optimizer.zero_grad()
        
        # Predict logits at test points using current z
        from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
        pred_logits = LatentFeasibilityChecker.batched_logit_values(decoder, z.unsqueeze(0), test_points)
        pred_logits = pred_logits.squeeze(0)  # (n_points,)
        
        # BCE loss
        eps = 1e-6
        pred_probs = torch.sigmoid(pred_logits / tau).clamp(eps, 1 - eps)
        target_probs = torch.sigmoid(outcomes / tau).clamp(eps, 1 - eps)
        loss = -(target_probs * torch.log(pred_probs) + (1 - target_probs) * torch.log(1 - pred_probs)).mean()
        
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            dist = (z.detach() - gt_z).norm().item()
            print(f"  Iter {i:3d}: loss={loss.item():.4f}, dist_to_GT={dist:.4f}")
    
    final_dist = (z.detach() - gt_z).norm().item()
    print(f"\nFinal dist to GT: {final_dist:.4f}")
    
    if final_dist < 1.0:
        print("SUCCESS: Direct optimization can find GT from observations!")
        print("=> The issue is with stochastic VI, not the likelihood function.")
    else:
        print("FAIL: Even direct optimization cannot find GT.")
        print("=> The observations don't provide enough signal to recover GT z.")


if __name__ == "__main__":
    test_direct_optimization()
