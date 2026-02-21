"""Test reduced kl_weight impact on VI convergence."""
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


def test_kl_weight(kl_weight, n_queries=10):
    torch.manual_seed(42)
    np.random.seed(42)
    
    decoder, embeddings, latent_dim = load_model()
    config = load_config()
    config['seed'] = 42
    config['vi']['kl_weight'] = kl_weight
    
    sample_idx = 42
    gt_z = embeddings[sample_idx].clone()
    
    prior_gen = LatentPriorGenerator(config, decoder)
    prior = prior_gen.get_prior(gt_z)
    posterior = LatentUserDistribution(latent_dim, decoder, prior.mean.clone(), prior.log_std.clone(), DEVICE)
    oracle = LatentOracle(decoder, gt_z, 4)
    vi = LatentVariationalInference(decoder, prior, posterior, config)
    bounds = get_bounds_from_config(config, DEVICE)
    
    initial_dist = (posterior.mean - gt_z).norm().item()
    print(f"Testing kl_weight={kl_weight}")
    print(f"  Initial dist to GT: {initial_dist:.4f}")
    
    for q in range(n_queries):
        tp = bounds[:, 0] + torch.rand(4, device=DEVICE) * (bounds[:, 1] - bounds[:, 0])
        outcome = oracle.query(tp)
        history = oracle.get_history()
        vi.update_posterior(history)
        
        dist_to_gt = (posterior.mean.detach() - gt_z).norm().item()
        print(f"  Query {q+1}: outcome={outcome:+.4f}, dist_to_GT={dist_to_gt:.4f}")
    
    final_dist = (posterior.mean.detach() - gt_z).norm().item()
    print(f"  Final dist to GT: {final_dist:.4f}")
    print(f"  Improvement: {(initial_dist - final_dist) / initial_dist * 100:.1f}%")
    return final_dist


if __name__ == "__main__":
    print("=" * 60)
    print("Testing different kl_weight values")
    print("=" * 60)
    
    results = {}
    for kl_weight in [0.15, 0.05, 0.01, 0.001]:
        print()
        results[kl_weight] = test_kl_weight(kl_weight)
    
    print()
    print("=" * 60)
    print("Summary (lower final dist is better):")
    print("=" * 60)
    for kl_weight, final_dist in results.items():
        print(f"  kl_weight={kl_weight}: final_dist={final_dist:.4f}")
