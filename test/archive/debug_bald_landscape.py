"""
Diagnostic script to visualize the BALD objective landscape.

This script fixes 2 dimensions of the latent space and plots the BALD score heatmap
for the other 2 dimensions. It also overlays the optimizer's trajectory if possible.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_bald import LatentBALD
from infer_params.training.model import LevelSetDecoder
from infer_params.training.dataset import LevelSetDataset


def load_decoder_model(checkpoint_path: str, device: str = DEVICE):
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
    model = model.to(device).eval()
    return model, embeddings.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--dataset", type=str, default="models/training_data.npz")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    decoder, embeddings = load_decoder_model(args.model, DEVICE)
    config = load_config()

    print(f"Loading sample {args.sample_index}...")
    dataset = LevelSetDataset(args.dataset)
    _, _, _, _, _, _ = dataset[args.sample_index]
    gt_z = embeddings[args.sample_index].clone()

    # Prior & Posterior
    prior_gen = LatentPriorGenerator(config, decoder)
    prior = prior_gen.get_prior(gt_z)
    posterior = LatentUserDistribution(
        latent_dim=32,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )

    # BALD
    bald = LatentBALD(decoder, posterior, config, prior=prior)
    bounds = get_bounds_from_config(config, DEVICE)
    
    # We visualizing dimensions 0 and 1, fixing 2 and 3 to their center
    dim_x, dim_y = 0, 1
    fixed_dims = {2: 0.0, 3: 0.0}
    
    x_range = torch.linspace(bounds[dim_x, 0], bounds[dim_x, 1], args.resolution, device=DEVICE)
    y_range = torch.linspace(bounds[dim_y, 0], bounds[dim_y, 1], args.resolution, device=DEVICE)
    
    X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
    grid_points = torch.zeros(args.resolution, args.resolution, 4, device=DEVICE)
    grid_points[:, :, dim_x] = X
    grid_points[:, :, dim_y] = Y
    for d, val in fixed_dims.items():
        grid_points[:, :, d] = val
        
    flat_points = grid_points.reshape(-1, 4)
    
    # Compute scores
    print("Computing BALD scores on grid...")
    scores = []
    # Sample zs ONCE for consistent landscape
    zs = bald.posterior.sample(bald.n_samples)
    
    batch_size = 100
    with torch.no_grad():
        for i in range(0, len(flat_points), batch_size):
            batch = flat_points[i:i+batch_size]
            # LatentBALD.compute_score expects a single test point, not a batch
            # But batched_logit_values supports it. LatentBALD needs update or we loop.
            # LatentBALD.compute_score currently takes (n_joints,). It does not support batching test points easily.
            # Let's just loop for now, optimization is not the goal here.
            for point in batch:
                s = bald.compute_score(point, zs=zs).item()
                scores.append(s)
                
    Z = np.array(scores).reshape(args.resolution, args.resolution)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(Z.T, origin='lower', extent=[bounds[dim_x, 0].item(), bounds[dim_x, 1].item(), bounds[dim_y, 0].item(), bounds[dim_y, 1].item()], cmap='viridis', aspect='auto')
    plt.colorbar(label='BALD Score')
    plt.xlabel(f'Joint {dim_x} ({config["prior"]["joint_names"][dim_x]})')
    plt.ylabel(f'Joint {dim_y} ({config["prior"]["joint_names"][dim_y]})')
    plt.title(f'BALD Objective Landscape (Slice at 0,0 for other dims)')
    
    save_path = "active_learning/images/bald_landscape.png"
    plt.savefig(save_path)
    print(f"Saved landscape to {save_path}")

if __name__ == "__main__":
    main()
