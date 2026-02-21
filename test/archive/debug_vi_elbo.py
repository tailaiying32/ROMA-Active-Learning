"""
Deeper diagnostic: Analyze whether the KL term is pulling posterior toward noisy prior.
"""

import sys
import os
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_variational_inference import LatentVariationalInference
from active_learning.src.test_history import TestHistory

from infer_params.training.model import LevelSetDecoder
from infer_params.training.dataset import LevelSetDataset


def load_decoder_model(checkpoint_path: str, device: str = DEVICE):
    """Load the LevelSetDecoder model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_config = checkpoint['config']
    model_cfg = train_config['model']

    embeddings = checkpoint['embeddings']
    num_samples = embeddings.shape[0]
    latent_dim = model_cfg['latent_dim']

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

    return model, embeddings.to(device), train_config


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load model
    decoder, embeddings, train_config = load_decoder_model("models/best_model.pt")
    print(f"Loaded decoder with {embeddings.shape[0]} samples, latent_dim={embeddings.shape[1]}")
    
    # Load config
    config = load_config()
    config['seed'] = 42
    
    latent_dim = train_config['model']['latent_dim']
    
    # Pick a sample
    sample_idx = np.random.randint(0, embeddings.shape[0])
    ground_truth_z = embeddings[sample_idx].clone()
    print(f"\nGround truth z norm: {ground_truth_z.norm().item():.4f}")
    
    # Generate prior
    prior_gen = LatentPriorGenerator(config, decoder)
    prior = prior_gen.get_prior(ground_truth_z)
    
    # Analyze prior vs ground truth
    print(f"\n{'='*60}")
    print("Prior Analysis")
    print(f"{'='*60}")
    prior_mean = prior.mean.detach()
    prior_std = torch.exp(prior.log_std).detach()
    
    gt_to_prior_dist = (ground_truth_z - prior_mean).norm().item()
    print(f"Prior mean norm: {prior_mean.norm().item():.4f}")
    print(f"Distance from GT to prior mean: {gt_to_prior_dist:.4f}")
    print(f"Prior std (mean across dims): {prior_std.mean().item():.4f}")
    print(f"Prior std range: [{prior_std.min().item():.4f}, {prior_std.max().item():.4f}]")
    
    # Check what the prior decodes to vs ground truth
    with torch.no_grad():
        gt_lower, gt_upper, _, _, _ = decoder.decode_from_embedding(ground_truth_z.unsqueeze(0))
        prior_lower, prior_upper, _, _, _ = decoder.decode_from_embedding(prior_mean.unsqueeze(0))
    
    print(f"\nGround truth joint limits:")
    print(f"  Lower: {np.rad2deg(gt_lower.cpu().numpy().flatten())}")
    print(f"  Upper: {np.rad2deg(gt_upper.cpu().numpy().flatten())}")
    
    print(f"\nPrior mean decoded to:")
    print(f"  Lower: {np.rad2deg(prior_lower.cpu().numpy().flatten())}")
    print(f"  Upper: {np.rad2deg(prior_upper.cpu().numpy().flatten())}")
    
    limit_mae = (
        torch.abs(gt_lower - prior_lower).mean().item() + 
        torch.abs(gt_upper - prior_upper).mean().item()
    ) / 2
    print(f"\nInitial joint limit MAE (prior mean): {limit_mae:.4f} rad = {np.rad2deg(limit_mae):.2f} deg")
    
    # Now simulate VI with a few test queries
    print(f"\n{'='*60}")
    print("Simulating VI With Sample Queries")
    print(f"{'='*60}")
    
    # Initialize posterior as copy of prior
    posterior = LatentUserDistribution(
        latent_dim=latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )
    
    # Create oracle
    oracle = LatentOracle(
        decoder=decoder,
        ground_truth_z=ground_truth_z,
        n_joints=4
    )
    
    # Create VI
    vi = LatentVariationalInference(
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        config=config
    )
    
    bounds = get_bounds_from_config(config, DEVICE)
    
    # Make a few random queries
    print("\nGenerating 5 random test queries:")
    for i in range(5):
        # Random test point
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        test_point = lower + torch.rand(4, device=DEVICE) * (upper - lower)
        
        # Query oracle
        outcome = oracle.query(test_point)
        
        print(f"  Query {i+1}: outcome = {outcome:+.4f} ({'FEASIBLE' if outcome >= 0 else 'INFEASIBLE'})")
    
    history = oracle.get_history()
    
    # Analyze likelihood gradients
    print(f"\n{'='*60}")
    print("Analyzing Likelihood Gradients")
    print(f"{'='*60}")
    
    # Compute gradients at different z values
    print("\nGradient of log-likelihood w.r.t. z at different positions:")
    
    # At prior mean
    posterior.mean.data = prior.mean.clone()
    posterior.mean.requires_grad_(True)
    ll_prior = vi.likelihood(history)
    ll_prior.backward()
    grad_at_prior = posterior.mean.grad.clone()
    
    # Direction from prior to GT
    direction_to_gt = ground_truth_z - prior.mean
    direction_to_gt_normalized = direction_to_gt / direction_to_gt.norm()
    
    # Gradient dot product with direction to GT
    grad_dot_gt = torch.dot(grad_at_prior, direction_to_gt_normalized).item()
    
    print(f"  At prior mean:")
    print(f"    LL = {ll_prior.item():.4f}")
    print(f"    Gradient norm = {grad_at_prior.norm().item():.4f}")
    print(f"    Gradient dot (direction to GT) = {grad_dot_gt:.4f}")
    print(f"    => {'GOOD: Gradient points toward GT' if grad_dot_gt > 0 else 'BAD: Gradient points away from GT'}")
    
    # At ground truth
    posterior.mean.grad.zero_()
    posterior.mean.data = ground_truth_z.clone()
    posterior.mean.requires_grad_(True)
    ll_gt = vi.likelihood(history)
    ll_gt.backward()
    grad_at_gt = posterior.mean.grad.clone()
    
    print(f"\n  At ground truth:")
    print(f"    LL = {ll_gt.item():.4f}")
    print(f"    Gradient norm = {grad_at_gt.norm().item():.4f}")
    
    # Compare LL values
    print(f"\n{'='*60}")
    print("Log-Likelihood Comparison")
    print(f"{'='*60}")
    print(f"LL at prior mean: {ll_prior.item():.4f}")
    print(f"LL at ground truth: {ll_gt.item():.4f}")
    if ll_gt.item() > ll_prior.item():
        print("=> GOOD: LL is higher at ground truth")
    else:
        print("=> BAD: LL is LOWER at ground truth - this explains the issue!")
        print("   The likelihood function is not correctly rewarding being close to GT!")
    
    # Now check the KL term
    print(f"\n{'='*60}")
    print("KL Divergence Analysis")
    print(f"{'='*60}")
    
    # KL at prior mean (should be low)
    posterior.mean.data = prior.mean.clone()
    kl_at_prior = vi.regularizer()
    print(f"KL at prior mean: {kl_at_prior.item():.4f}")
    
    # KL at ground truth (could be high if GT is far from prior)
    posterior.mean.data = ground_truth_z.clone()
    kl_at_gt = vi.regularizer()
    print(f"KL at ground truth: {kl_at_gt.item():.4f}")
    
    # ELBO comparison
    print(f"\n{'='*60}")
    print("ELBO Comparison")
    print(f"{'='*60}")
    elbo_prior = ll_prior.item() - kl_at_prior.item()
    elbo_gt = ll_gt.item() - kl_at_gt.item()
    print(f"ELBO at prior mean: {elbo_prior:.4f} (LL={ll_prior.item():.4f}, KL={kl_at_prior.item():.4f})")
    print(f"ELBO at ground truth: {elbo_gt:.4f} (LL={ll_gt.item():.4f}, KL={kl_at_gt.item():.4f})")
    
    if elbo_gt > elbo_prior:
        print("=> GOOD: ELBO is higher at ground truth")
    else:
        print("=> BAD: ELBO is LOWER at ground truth!")
        print("   This explains why VI moves away from GT.")
        if kl_at_gt.item() > ll_gt.item() - ll_prior.item():
            print("   CAUSE: KL penalty overwhelms likelihood improvement!")
            print(f"   KL increase: {kl_at_gt.item() - kl_at_prior.item():.4f}")
            print(f"   LL increase: {ll_gt.item() - ll_prior.item():.4f}")


if __name__ == "__main__":
    main()
