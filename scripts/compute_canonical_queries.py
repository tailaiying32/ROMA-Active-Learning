"""
Compute canonical exploration queries using BatchBALD.

This script pre-computes a fixed sequence of test points that maximize
joint mutual information across all training embeddings. These points
are designed to be maximally informative about ANY user in the training
distribution.

Usage:
    python active_learning/scripts/compute_canonical_queries.py

Output:
    models/canonical_queries.npz
"""

import sys
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from infer_params.training.model import LevelSetDecoder


def load_decoder_and_embeddings(model_path: str):
    """Load the decoder model and all training embeddings."""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    train_config = checkpoint['config']
    model_cfg = train_config['model']
    
    embeddings = checkpoint['embeddings'].to(DEVICE)
    num_samples = embeddings.shape[0]
    latent_dim = model_cfg['latent_dim']
    
    decoder = LevelSetDecoder(
        num_samples=num_samples,
        latent_dim=latent_dim,
        hidden_dim=model_cfg['hidden_dim'],
        num_blocks=model_cfg['num_blocks'],
        num_slots=model_cfg.get('num_slots', 18),
        params_per_slot=model_cfg.get('params_per_slot', 6),
    )
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder = decoder.to(DEVICE)
    decoder.eval()
    
    return decoder, embeddings


def compute_predictive_probs(decoder, embeddings, test_points, tau=0.2, batch_size=500):
    """
    Compute predictive probabilities for all embeddings at given test points.
    
    Args:
        decoder: LevelSetDecoder
        embeddings: (N, latent_dim) tensor of all training embeddings
        test_points: (M, n_joints) tensor of test point coordinates
        tau: Temperature for sigmoid
        batch_size: Batch size for embeddings
        
    Returns:
        probs: (N, M) tensor of feasibility probabilities
    """
    N = embeddings.shape[0]
    M = test_points.shape[0]
    
    probs = torch.zeros(N, M, device=DEVICE)
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_emb = embeddings[i:i+batch_size]
            # Get logits for this batch at all test points
            logits = LatentFeasibilityChecker.batched_logit_values(decoder, batch_emb, test_points)
            probs[i:i+batch_size] = torch.sigmoid(logits / tau)
    
    return probs


def binary_entropy(p, eps=1e-6):
    """Compute binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
    p = p.clamp(eps, 1 - eps)
    return -p * torch.log(p) - (1 - p) * torch.log(1 - p)


def compute_joint_mi(probs_selected, probs_candidate):
    """
    Compute joint mutual information for BatchBALD.
    
    For batch selection, we approximate the joint MI using the greedy
    conditional MI: I(y_new; embeddings | y_selected).
    
    Args:
        probs_selected: List of (N,) tensors for already selected points
        probs_candidate: (N,) tensor of probs for candidate point
        
    Returns:
        joint_mi: Scalar mutual information estimate
    """
    N = probs_candidate.shape[0]
    
    if len(probs_selected) == 0:
        # First point: just compute standard BALD score
        mean_prob = probs_candidate.mean()
        entropy_mean = binary_entropy(mean_prob)
        mean_entropy = binary_entropy(probs_candidate).mean()
        return (entropy_mean - mean_entropy).item()
    
    # Stack all probs: (n_selected + 1, N)
    all_probs = torch.stack(probs_selected + [probs_candidate], dim=0)
    n_points = all_probs.shape[0]
    
    # Compute joint entropy using Monte Carlo over embeddings
    # For each embedding, the joint probability is product of independent Bernoullis
    # We approximate this by looking at how much the candidate adds information
    
    # Simple approximation: measure how decorrelated the candidate is from selected
    # Higher correlation = less new information
    
    # Use conditional entropy approach:
    # I(y_new; z | y_1..k) ≈ H(y_new | y_1..k) - H(y_new | z, y_1..k)
    #                      ≈ H(y_new) - correlation_penalty
    
    # Compute BALD score for candidate
    mean_prob = probs_candidate.mean()
    entropy_mean = binary_entropy(mean_prob)
    mean_entropy = binary_entropy(probs_candidate).mean()
    bald_score = entropy_mean - mean_entropy
    
    # Compute redundancy with selected points
    # If candidate correlates with selected points, reduce its score
    redundancy = 0.0
    for prev_probs in probs_selected:
        # Correlation in probability space
        corr = torch.corrcoef(torch.stack([probs_candidate, prev_probs]))[0, 1]
        if not torch.isnan(corr):
            redundancy += abs(corr.item())
    
    # Discount BALD score by redundancy
    redundancy_factor = 1.0 / (1.0 + redundancy)
    conditional_mi = bald_score * redundancy_factor
    
    return conditional_mi.item()


def batch_bald_select(decoder, embeddings, bounds, n_points=5, n_candidates=1000, tau=0.2, seed=42):
    """
    Select n_points using BatchBALD greedy approximation.
    
    Args:
        decoder: LevelSetDecoder model
        embeddings: (N, latent_dim) all training embeddings
        bounds: (n_joints, 2) anatomical bounds
        n_points: Number of canonical points to select
        n_candidates: Number of candidate points to evaluate
        tau: Temperature for sigmoid
        seed: Random seed for candidate sampling
        
    Returns:
        selected_points: (n_points, n_joints) tensor
        scores: List of scores for each selected point
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_joints = bounds.shape[0]
    
    # Sample candidate test points uniformly within bounds
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    candidates = lower + torch.rand(n_candidates, n_joints, device=DEVICE) * (upper - lower)
    
    print(f"Computing predictive probabilities for {n_candidates} candidates x {embeddings.shape[0]} embeddings...")
    
    # Pre-compute probabilities for all candidates at all embeddings
    all_probs = compute_predictive_probs(decoder, embeddings, candidates, tau=tau)
    # all_probs: (N_embeddings, n_candidates)
    
    print(f"Selecting {n_points} points using BatchBALD...")
    
    selected_indices = []
    selected_probs = []
    scores = []
    
    for i in range(n_points):
        best_score = float('-inf')
        best_idx = None
        
        for c in tqdm(range(n_candidates), desc=f"Point {i+1}/{n_points}", leave=False):
            if c in selected_indices:
                continue
            
            candidate_probs = all_probs[:, c]
            score = compute_joint_mi(selected_probs, candidate_probs)
            
            if score > best_score:
                best_score = score
                best_idx = c
        
        selected_indices.append(best_idx)
        selected_probs.append(all_probs[:, best_idx])
        scores.append(best_score)
        
        print(f"  Selected point {i+1}: index={best_idx}, score={best_score:.4f}")
    
    selected_points = candidates[selected_indices]
    
    return selected_points, scores


def main():
    parser = argparse.ArgumentParser(description="Compute canonical exploration queries using BatchBALD")
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Path to trained model")
    parser.add_argument("--output", type=str, default="models/canonical_queries.npz", help="Output path")
    parser.add_argument("--n-points", type=int, default=5, help="Number of canonical points")
    parser.add_argument("--n-candidates", type=int, default=1000, help="Number of candidate test points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Computing Canonical Exploration Queries (BatchBALD)")
    print("=" * 60)
    
    # Load model and embeddings
    print(f"\nLoading model from {args.model}...")
    decoder, embeddings = load_decoder_and_embeddings(args.model)
    print(f"  Loaded {embeddings.shape[0]} embeddings, latent_dim={embeddings.shape[1]}")
    
    # Load config for bounds
    config = load_config()
    bounds = get_bounds_from_config(config, DEVICE)
    print(f"  Bounds shape: {bounds.shape}")
    
    # Get tau from config
    tau = config.get('bald', {}).get('tau', 0.2)
    print(f"  Using tau={tau}")
    
    # Run BatchBALD selection
    print()
    selected_points, scores = batch_bald_select(
        decoder, embeddings, bounds,
        n_points=args.n_points,
        n_candidates=args.n_candidates,
        tau=tau,
        seed=args.seed
    )
    
    # Convert to numpy for saving
    points_np = selected_points.cpu().numpy()
    scores_np = np.array(scores)
    
    # Save
    print(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez(args.output, points=points_np, scores=scores_np)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Selected Canonical Points (in radians):")
    print("=" * 60)
    for i, (point, score) in enumerate(zip(points_np, scores_np)):
        point_deg = np.rad2deg(point)
        print(f"  {i+1}. score={score:.4f}, point={point_deg} (deg)")
    
    print(f"\nSaved {args.n_points} canonical points to {args.output}")


if __name__ == "__main__":
    main()
