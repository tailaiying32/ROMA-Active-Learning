"""Compare Hybrid vs BALD vs Random acquisition strategies."""
import sys
import os
import copy
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_active_learning import LatentActiveLearner
from active_learning.src.latent_bald import LatentBALD
from active_learning.src.hybrid_acquisition import HybridAcquisition
from infer_params.training.model import LevelSetDecoder


def load_model():
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
    return decoder, emb.to(DEVICE), model_cfg['latent_dim']


class RandomStrategy:
    """Simple random test point selection."""
    def __init__(self, device=DEVICE):
        self.device = device
    
    def select_test(self, bounds, **kwargs):
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        point = lower + torch.rand(bounds.shape[0], device=self.device) * (upper - lower)
        return point, 0.0


def run_trial(decoder, embeddings, latent_dim, config, sample_idx, strategy_name, n_queries=15):
    """Run a single trial with given strategy."""
    torch.manual_seed(sample_idx)
    np.random.seed(sample_idx)
    
    # Override config to not auto-load hybrid
    config = copy.deepcopy(config)  # Deep copy to avoid mutation
    config['acquisition'] = {'strategy': 'bald'}  # Default to bald
    
    gt_z = embeddings[sample_idx].clone()
    
    prior_gen = LatentPriorGenerator(config, decoder)
    prior = prior_gen.get_prior(gt_z)
    posterior = LatentUserDistribution(latent_dim, decoder, prior.mean.clone(), prior.log_std.clone(), DEVICE)
    oracle = LatentOracle(decoder, gt_z, 4)
    bounds = get_bounds_from_config(config, DEVICE)
    
    # Create strategy
    bald = LatentBALD(decoder, posterior, prior, config)
    
    if strategy_name == 'hybrid':
        strategy = HybridAcquisition(bald, 'models/canonical_queries.npz', n_canonical=5)
    elif strategy_name == 'random':
        strategy = RandomStrategy()
    else:
        strategy = bald
    
    learner = LatentActiveLearner(decoder, prior, posterior, oracle, bounds, config, acquisition_strategy=strategy)
    
    # Track metrics
    errors = []
    for i in range(n_queries):
        learner.step(verbose=False)
        dist = (posterior.mean.detach() - gt_z).norm().item()
        errors.append(dist)
    
    return errors


def main():
    decoder, embeddings, latent_dim = load_model()
    config = load_config()
    
    # Use multiple samples for comparison
    sample_indices = [42, 100, 500]  # 3 different users
    n_queries = 15
    
    results = {s: [] for s in ['hybrid', 'bald', 'random']}
    
    print("=" * 60)
    print("Comparing Acquisition Strategies")
    print("=" * 60)
    
    for idx in sample_indices:
        print(f"\nSample {idx}:")
        for strategy in ['hybrid', 'bald', 'random']:
            errors = run_trial(decoder, embeddings, latent_dim, config, idx, strategy, n_queries)
            results[strategy].append(errors)
            print(f"  {strategy:8s}: final_error={errors[-1]:.4f}")
    
    # Average across samples
    print("\n" + "=" * 60)
    print("Final Errors (averaged across samples)")
    print("=" * 60)
    for strategy in ['hybrid', 'bald', 'random']:
        final_errors = [r[-1] for r in results[strategy]]
        mean_err = np.mean(final_errors)
        std_err = np.std(final_errors)
        print(f"  {strategy:8s}: {mean_err:.4f} ± {std_err:.4f}")
    
    # Determine winner
    means = {s: np.mean([r[-1] for r in results[s]]) for s in results}
    winner = min(means, key=means.get)
    print(f"\nWinner (lowest final error): {winner}")


if __name__ == "__main__":
    main()
