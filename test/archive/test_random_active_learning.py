"""
Test script for Random Active Learning Pipeline (Baseline).

This script mirrors test_latent_active_learning.py but uses RandomStrategy
instead of LatentBALD for test selection.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_active_learning import LatentActiveLearner
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.random_strategy import RandomStrategy
from active_learning.test.visualization_utils import LatentVisualizer

from infer_params.training.model import LevelSetDecoder
from infer_params.training.dataset import LevelSetDataset
from infer_params.config import load_default_config


def load_decoder_model(checkpoint_path: str, device: str = DEVICE):
    """
    Load the LevelSetDecoder model from a checkpoint.

    Args:
        checkpoint_path: Path to the best_model.pt checkpoint
        device: torch device

    Returns:
        model: LevelSetDecoder model
        embeddings: Trained embeddings tensor (num_samples, latent_dim)
        config: Training config from checkpoint
    """
    print(f"Loading decoder model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    train_config = checkpoint['config']
    model_cfg = train_config['model']

    # Get num_samples from embeddings
    embeddings = checkpoint['embeddings']
    num_samples = embeddings.shape[0]
    latent_dim = model_cfg['latent_dim']

    print(f"  Num samples: {num_samples}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Hidden dim: {model_cfg['hidden_dim']}")
    print(f"  Num blocks: {model_cfg['num_blocks']}")

    # Create model
    model = LevelSetDecoder(
        num_samples=num_samples,
        latent_dim=latent_dim,
        hidden_dim=model_cfg['hidden_dim'],
        num_blocks=model_cfg['num_blocks'],
        num_slots=model_cfg.get('num_slots', 18),
        params_per_slot=model_cfg.get('params_per_slot', 6),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Model loaded successfully (epoch {checkpoint['epoch']})")

    return model, embeddings.to(device), train_config


def load_ground_truth_from_dataset(dataset_path: str, sample_index: int = None):
    """
    Load a ground truth sample from the dataset.

    Args:
        dataset_path: Path to the dataset.npz or training_data.npz
        sample_index: Index of sample to use as ground truth (random if None)

    Returns:
        sample_index: The selected sample index
        ground_truth_params: Dict with box_lower, box_upper, box_weights, etc.
    """
    print(f"Loading ground truth from dataset: {dataset_path}...")
    dataset = LevelSetDataset(dataset_path)

    if sample_index is None:
        sample_index = np.random.randint(0, len(dataset))

    print(f"  Selected sample index: {sample_index}")
    print(f"  Total samples in dataset: {len(dataset)}")

    # Get ground truth params for this sample
    idx, box_lower, box_upper, box_weights, presence, blob_params = dataset[sample_index]

    ground_truth_params = {
        'box_lower': box_lower,
        'box_upper': box_upper,
        'box_weights': box_weights,
        'presence': presence,
        'blob_params': blob_params,
    }
    return sample_index, ground_truth_params


def extract_true_limits_from_params(ground_truth_params: dict, joint_names: list) -> dict:
    """
    Extract true joint limits from ground truth params for visualization.

    Args:
        ground_truth_params: Dict with box_lower, box_upper
        joint_names: List of joint names

    Returns:
        true_limits: Dict mapping joint_name -> (lower, upper)
    """
    box_lower = ground_truth_params['box_lower'].numpy()
    box_upper = ground_truth_params['box_upper'].numpy()

    true_limits = {}
    for i, name in enumerate(joint_names):
        true_limits[name] = (float(box_lower[i]), float(box_upper[i]))

    return true_limits


def run_random_active_learning_demo(
    model_path: str = '../models/best_model.pt',
    dataset_path: str = '../models/training_data.npz',
    sample_index: int = None,
    verbose: bool = True,
    budget_override: int = None
):
    """
    Run the random active learning demo.

    Args:
        model_path: Path to best_model.pt
        dataset_path: Path to training_data.npz or dataset.npz
        sample_index: Index of sample to use as ground truth (random if None)
        verbose: Print detailed output
        budget_override: Override the budget from config
    """
    print("=" * 60)
    print("=== Random Active Learning Demo (Baseline) ===")
    print("=" * 60)

    # 1. Load configuration
    print("\n[Step 1] Loading configuration...")
    config = load_config()
    print(f"  Device: {DEVICE}")

    # Set seeds for reproducibility
    seed = config.get('seed', None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"  Random seed: {seed}")

    # Override budget if requested
    if budget_override is not None:
        if 'stopping' not in config:
            config['stopping'] = {}
        config['stopping']['budget'] = budget_override

    # 2. Load decoder model and embeddings
    print("\n[Step 2] Loading decoder model...")
    decoder, embeddings, train_config = load_decoder_model(model_path, DEVICE)
    latent_dim = train_config['model']['latent_dim']

    # 3. Load ground truth from dataset
    print("\n[Step 3] Loading ground truth from dataset...")
    sample_idx, ground_truth_params = load_ground_truth_from_dataset(dataset_path, sample_index)

    # 4. Get ground truth latent code (from trained embeddings)
    print("\n[Step 4] Getting ground truth latent code...")
    ground_truth_z = embeddings[sample_idx].clone()
    print(f"  Ground truth z shape: {ground_truth_z.shape}")
    print(f"  Ground truth z norm: {ground_truth_z.norm().item():.4f}")

    # 5. Create ground truth checker
    print("\n[Step 5] Creating ground truth feasibility checker...")
    ground_truth_checker = LatentFeasibilityChecker(
        decoder=decoder,
        z=ground_truth_z,
        device=DEVICE
    )

    # Get decoded joint limits for display
    lower, upper = ground_truth_checker.joint_limits()
    print(f"  Decoded box lower: {lower.cpu().detach().numpy()}")
    print(f"  Decoded box upper: {upper.cpu().detach().numpy()}")

    # 6. Generate prior (perturbed ground truth)
    print("\n[Step 6] Generating prior distribution...")
    prior_gen = LatentPriorGenerator(config, decoder)
    prior = prior_gen.get_prior(ground_truth_z)

    # 7. Initialize posterior as copy of prior
    print("\n[Step 7] Initializing posterior...")
    posterior = LatentUserDistribution(
        latent_dim=latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )

    # 8. Setup oracle
    print("\n[Step 8] Setting up oracle...")
    n_joints = 4
    oracle = LatentOracle(
        decoder=decoder,
        ground_truth_z=ground_truth_z,
        n_joints=n_joints
    )

    # 9. Get bounds for test point optimization
    print("\n[Step 9] Getting test point bounds...")
    bounds = get_bounds_from_config(config, DEVICE)
    print(f"  Bounds shape: {bounds.shape}")
    print(f"  Bounds (rad):\n{bounds.cpu().numpy()}")

    # 10. Create active learner with RANDOM STRATEGY
    print("\n[Step 10] Creating active learner with RandomStrategy...")
    random_strategy = RandomStrategy(config=config)
    learner = LatentActiveLearner(
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        bounds=bounds,
        config=config,
        acquisition_strategy=random_strategy  # INJECT RANDOM STRATEGY
    )

    # 11. Setup visualizer
    print("\n[Step 11] Setting up visualizer...")
    viz_dir = 'active_learning/images/random_al/'  # NEW DIRECTORY
    joint_names = [f"joint_{i}" for i in range(n_joints)]

    # Get true limits for visualization
    true_limits = {}
    lower_np = lower.cpu().detach().numpy()
    upper_np = upper.cpu().detach().numpy()
    for i, name in enumerate(joint_names):
        true_limits[name] = (float(lower_np[i]), float(upper_np[i]))

    # Get anatomical limits from config bounds for y-axis capping
    bounds_np = bounds.cpu().numpy()  # (n_joints, 2)
    bounds_dict = {f"joint_{i}": (float(bounds_np[i, 0]), float(bounds_np[i, 1])) for i in range(n_joints)}

    # Get grid resolution from config
    grid_resolution = config.get('metrics', {}).get('grid_resolution', 12)

    visualizer = LatentVisualizer(
        save_dir=viz_dir,
        joint_names=joint_names,
        decoder=decoder,
        true_limits=true_limits,
        ground_truth_params=ground_truth_params,
        resolution=grid_resolution,
        anatomical_limits=bounds_dict,  # Pass anatomical limits for y-axis capping
        true_checker=ground_truth_checker
    )

    # Log initial prior state BEFORE any steps
    visualizer.log_initial_state(prior, ground_truth_z)

    # 12. Run active learning loop
    print("\n" + "=" * 60)
    print("Starting Random Active Learning Loop...")
    print("=" * 60)

    iteration = 1  # Start at 1 since iteration 0 is the initial prior
    while True:
        # Check stopping criteria
        should_stop, reason = learner.check_stopping_criteria()
        if should_stop:
            print(f"\nStopping Criteria Met: {reason}")
            break

        # Execute one step
        result = learner.step(verbose=verbose)

        # Log to visualizer
        visualizer.log_iteration(
            iteration=iteration,
            posterior=posterior,
            result=result,
            ground_truth_z=ground_truth_z
        )

        iteration += 1

    # 13. Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    print(f"\nSaving visualizations to {viz_dir}...")
    visualizer.plot_information_gain()
    visualizer.plot_latent_error()
    # Read visualization config
    cap_to_anatomical = config.get('visualization', {}).get('cap_joint_evolution_to_anatomical', False)
    visualizer.plot_joint_evolution_and_queries(
        true_limits, 
        cap_to_anatomical=cap_to_anatomical, 
        anatomical_limits=bounds_dict
    )
    visualizer.plot_latent_evolution()
    visualizer.plot_consolidated_metrics()  # Combined: MAE, Uncertainty, IoU, Accuracy
    visualizer.save_history()

    # 14. Print final results
    print("\n" + "=" * 60)
    print("=== Final Results (Random Baseline) ===")
    print("=" * 60)

    print(f"\nTotal queries: {len(learner.results)}")

    # Final posterior latent stats
    print("\nFinal Posterior Latent Distribution:")
    final_mean = posterior.mean.detach().cpu().numpy()
    final_std = torch.exp(posterior.log_std).detach().cpu().numpy()
    print(f"  Mean norm: {np.linalg.norm(final_mean):.4f}")
    print(f"  Std mean: {np.mean(final_std):.4f}")

    # Latent error
    gt_z = ground_truth_z.cpu().numpy()
    latent_error = np.linalg.norm(final_mean - gt_z)
    print(f"  Latent error (L2): {latent_error:.4f}")

    # Decoded joint limits comparison
    print("\nDecoded Joint Limits Comparison:")
    print("  Joint            | True Range          | Posterior Mean Range")
    print("  " + "-" * 60)

    # Get posterior mean decoded limits
    with torch.no_grad():
        post_lower, post_upper, _, _, _ = decoder.decode_from_embedding(
            posterior.mean.unsqueeze(0)
        )
        post_lower = post_lower.squeeze().cpu().numpy()
        post_upper = post_upper.squeeze().cpu().numpy()

    param_error = 0.0
    for i, name in enumerate(joint_names):
        true_l, true_u = true_limits[name]
        pred_l, pred_u = post_lower[i], post_upper[i]
        error = (abs(pred_l - true_l) + abs(pred_u - true_u)) / 2
        param_error += error
        print(f"  {name:15} | [{true_l:6.3f}, {true_u:6.3f}] | [{pred_l:6.3f}, {pred_u:6.3f}] | err={error:.4f}")

    param_error /= len(joint_names)

    print("\n=== Final Evaluation ===")
    print(f"  Latent Error (L2): {latent_error:.4f}")
    print(f"  Parameter MAE: {param_error:.4f} rad")
    
    if visualizer.history.get('reachability_iou'):
        final_iou = visualizer.history['reachability_iou'][-1]
        final_acc = visualizer.history['reachability_accuracy'][-1]
        print(f"  Reachability IoU: {final_iou:.4f}")
        print(f"  Reachability Accuracy: {final_acc:.4f}")
        
    print(f"  Final ELBO: {learner.results[-1].elbo:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Random Active Learning Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to best_model.pt checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="models/training_data.npz",
        help="Path to training_data.npz or dataset.npz"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Sample index to use as ground truth (random if not specified)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Override active learning budget"
    )

    args = parser.parse_args()

    run_random_active_learning_demo(
        model_path=args.model,
        dataset_path=args.dataset,
        sample_index=args.sample_index,
        verbose=not args.quiet,
        budget_override=args.budget
    )
