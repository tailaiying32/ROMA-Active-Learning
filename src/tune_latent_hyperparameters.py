"""
Hyperparameter tuning script for Latent Active Learning Pipeline using Optuna.

Similar to tune_hyperparameters.py but adapted for the latent space pipeline.
Uses SQLite storage for distributed optimization across multiple jobs.

Tuned Hyperparameters:
======================

BALD Acquisition:
    - bald_tau: Sigmoid temperature for feasibility boundary
    - bald_n_mc_samples: Monte Carlo samples for BALD estimation
    - bald_sampling_temperature: Temperature scaling for posterior sampling
    - bald_epsilon: Epsilon-greedy exploration rate
    - bald_epsilon_decay: Decay factor for epsilon per iteration
    - bald_use_weighted_bald: Enable weighted BALD (boundary targeting)
    - bald_weighted_bald_sigma: Width of Gaussian gate for weighted BALD
    - bald_diversity_weight: Weight for diversity penalty

Acquisition Strategy:
    - acquisition_strategy: 'bald', 'quasi-random', or 'random'
    - acquisition_n_quasi_random: Number of initial Sobol points

BALD Optimization (test point selection):
    - bald_n_restarts: Number of random restarts
    - bald_n_iters_per_restart: Iterations per restart
    - bald_lr_adam: Learning rate for ADAM phase
    - bald_lr_sgd: Learning rate for SGD phase
    - bald_switch_to_sgd_at: Fraction of iters before switching to SGD
    - bald_plateau_patience: Patience for plateau detection
    - bald_plateau_threshold: Threshold for plateau detection

Variational Inference:
    - vi_n_mc_samples: MC samples for ELBO estimation
    - vi_noise_std: Gaussian likelihood noise std
    - vi_lr: Learning rate
    - vi_optimizer_type: 'adam' or 'sgd'
    - vi_max_iters: Maximum optimization iterations
    - vi_convergence_tol: Convergence tolerance
    - vi_patience: Early stopping patience
    - vi_grad_clip: Gradient clipping threshold

KL Annealing:
    - vi_kl_annealing_enabled: Enable KL annealing
    - vi_kl_start_weight: Initial KL weight
    - vi_kl_end_weight: Final KL weight
    - vi_kl_duration: Annealing duration (iterations)
    - vi_kl_schedule: 'linear', 'cosine', or 'step'

Prior Initialization:
    - prior_init_std: Initial std for prior distribution
    - prior_mean_noise_std: Noise std for prior mean perturbation

Stopping Criteria:
    - stopping_budget: Maximum number of queries
    - stopping_uncertainty_enabled: Enable uncertainty-based stopping
    - stopping_uncertainty_threshold: Threshold for uncertainty stopping
    - stopping_bald_enabled: Enable BALD score stopping
    - stopping_bald_threshold: Threshold for BALD stopping
    - stopping_bald_patience: Patience for BALD stopping

Usage:
    python -m active_learning.src.tune_latent_hyperparameters --n-trials 100
"""
import sys
import os
import time
import numpy as np
import json
import copy
from pathlib import Path
from typing import Tuple, Dict
import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

if not torch.cuda.is_available():
    print("FATAL: CUDA not found on this node. Exiting to avoid recording a 'Failed' trial.")
    sys.exit(1)

# Add the project root to sys.path
# Add the project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '../../')))

from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_active_learning import LatentActiveLearner
from active_learning.src.config import load_config, DEVICE
from active_learning.src.metrics import compute_reachability_metrics

from infer_params.training.model import LevelSetDecoder
from infer_params.training.dataset import LevelSetDataset
from infer_params.training.level_set_torch import create_evaluation_grid
from infer_params.config import load_default_config


def load_decoder_model(checkpoint_path: str, device: str = DEVICE):
    """
    Load the LevelSetDecoder model from a checkpoint.
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

    return model, embeddings.to(device), latent_dim


def generate_prior_from_ground_truth(
    ground_truth_z: torch.Tensor,
    decoder,
    config: dict,
    device: str = DEVICE
) -> LatentUserDistribution:
    """
    Generate a prior distribution by perturbing the ground truth latent code.
    """
    prior_config = config.get('prior', {})
    mean_noise_std = prior_config.get('mean_noise_std', 0.4)
    init_std = prior_config.get('init_std', 0.4)

    latent_dim = ground_truth_z.shape[0]

    # Perturb ground truth with noise
    noise = torch.randn(latent_dim, device=device) * mean_noise_std
    prior_mean = ground_truth_z + noise
    prior_log_std = torch.full((latent_dim,), np.log(init_std), device=device)

    prior = LatentUserDistribution(
        latent_dim=latent_dim,
        decoder=decoder,
        mean=prior_mean,
        log_std=prior_log_std,
        device=device
    )

    return prior


def get_bounds_from_config(config: dict) -> torch.Tensor:
    """
    Extract test point bounds from config (anatomical limits).
    """
    # Try active learning config first
    prior_config = config.get('prior', {})
    anatomical_limits = prior_config.get('anatomical_limits', None)

    if anatomical_limits is not None:
        # Config has anatomical limits in degrees
        units = prior_config.get('units', 'degrees')
        joint_names = prior_config.get('joint_names', list(anatomical_limits.keys()))

        bounds_list = []
        for joint in joint_names:
            limits = anatomical_limits[joint]
            if units == 'degrees':
                limits = [np.deg2rad(limits[0]), np.deg2rad(limits[1])]
            bounds_list.append(limits)

        return torch.tensor(bounds_list, dtype=torch.float32, device=DEVICE)

    # Fallback to infer_params default config
    infer_config = load_default_config()
    base_lower = infer_config['joints']['base_lower']
    base_upper = infer_config['joints']['base_upper']

    bounds = torch.tensor(
        [[l, u] for l, u in zip(base_lower, base_upper)],
        dtype=torch.float32,
        device=DEVICE
    )

    return bounds


def load_ground_truth_from_dataset(dataset: LevelSetDataset, sample_index: int) -> Dict:
    """
    Load ground truth parameters from the dataset.

    This matches exactly how the pipeline (compare_bald_vs_random.py) loads ground truth.
    """
    idx, box_lower, box_upper, box_weights, presence, blob_params = dataset[sample_index]
    return {
        'box_lower': box_lower,
        'box_upper': box_upper,
        'box_weights': box_weights,
        'presence': presence,
        'blob_params': blob_params,
    }


def compute_iou_metric(
    decoder,
    posterior: LatentUserDistribution,
    ground_truth_params: Dict,
    test_grid: torch.Tensor
) -> Tuple[float, float]:
    """
    Compute IoU and accuracy between posterior prediction and ground truth.

    Uses the same method as the main pipeline (compare_bald_vs_random.py).

    Args:
        decoder: LevelSetDecoder model
        posterior: Current posterior distribution
        ground_truth_params: Ground truth parameters from dataset
        test_grid: Evaluation grid points

    Returns:
        (iou, accuracy) - both in [0, 1], higher is better
    """
    iou, accuracy, _, _ = compute_reachability_metrics(
        decoder=decoder,
        ground_truth_params=ground_truth_params,
        posterior_mean=posterior.mean.unsqueeze(0),
        test_grid=test_grid,
    )

    return iou, accuracy


def compute_latent_log_likelihood(posterior: LatentUserDistribution,
                                   ground_truth_z: torch.Tensor,
                                   decoder) -> float:
    """
    Compute log likelihood of ground truth latent code under posterior.

    Returns average LL (higher is better).
    """
    # Latent space log likelihood
    z_dist = torch.distributions.Normal(posterior.mean, torch.exp(posterior.log_std))
    latent_ll = z_dist.log_prob(ground_truth_z).mean().item()

    # Decoded parameter space comparison (for interpretability)
    with torch.no_grad():
        # Ground truth decoded limits
        gt_lower, gt_upper, _, _, _ = decoder.decode_from_embedding(ground_truth_z.unsqueeze(0))
        gt_lower = gt_lower.squeeze(0)
        gt_upper = gt_upper.squeeze(0)

        # Posterior mean decoded limits
        post_lower, post_upper, _, _, _ = decoder.decode_from_embedding(posterior.mean.unsqueeze(0))
        post_lower = post_lower.squeeze(0)
        post_upper = post_upper.squeeze(0)

        # Parameter MAE (lower is better, so negate for consistency with LL)
        param_mae = (torch.abs(post_lower - gt_lower) + torch.abs(post_upper - gt_upper)).mean().item()

    # Combined metric: latent LL - param_mae penalty
    # Higher is better
    return latent_ll - 0.5 * param_mae


def evaluate_single_seed(config, decoder, embeddings, latent_dim, seed):
    """
    Run a single latent active learning run with a specific seed.
    Returns (ll_score, time_per_iter, total_iters)
    """
    # Force seed
    if 'user_generation' not in config:
        config['user_generation'] = {}
    config['user_generation']['seed'] = seed

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    max_budget = config.get('stopping', {}).get('budget', 100)

    # Get bounds
    bounds = get_bounds_from_config(config)
    n_joints = bounds.shape[0]

    # Pick a sample from the dataset as ground truth
    # We map seed to an index
    num_samples = embeddings.shape[0]
    sample_idx = seed % num_samples
    ground_truth_z = embeddings[sample_idx].clone()

    # Initialize prior and posterior
    prior = generate_prior_from_ground_truth(ground_truth_z, decoder, config, DEVICE)
    
    # Initialize posterior as copy of prior
    posterior = LatentUserDistribution(
        latent_dim=latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )

    # Setup oracle
    oracle = LatentOracle(
        decoder=decoder,
        ground_truth_z=ground_truth_z,
        n_joints=n_joints
    )

    # Setup active learner
    learner = LatentActiveLearner(
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        bounds=bounds,
        config=config
    )

    try:
        results = learner.run(verbose=False)
        total_iters = len(results)
    except Exception as e:
        print(f"    Seed {seed} failed: {e}")
        # import traceback
        # traceback.print_exc()
        return -100.0, 0.0, max_budget  # Penalty

    elapsed = time.time() - start_time
    avg_time = elapsed / total_iters if total_iters > 0 else 0

    # Compute log likelihood metric
    ll_score = compute_latent_log_likelihood(posterior, ground_truth_z, decoder)

    return ll_score, avg_time, total_iters


def evaluate_pipeline(config, decoder, embeddings, latent_dim):
    """
    Run the latent active learning pipeline over multiple seeds and aggregate loss.
    Loss is defined as: -1 * Mean_LL + Time_Penalty
    """
    # seeds = [100]  # Reduced for fast verification
    seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]  # Fixed seeds for consistency
    
    # For testing, reduce seeds
    # seeds = seeds[:2]
    
    losses = []
    ll_scores = []
    times = []
    iters = []

    sys.stdout.flush()
    print(f"Starting Trial (Average over {len(seeds)} seeds)...")

    for seed in seeds:
        ll, t, n_iter = evaluate_single_seed(config, decoder, embeddings, latent_dim, seed=seed)
        ll_scores.append(ll)
        times.append(t)
        iters.append(n_iter)

        # Loss = -LL + alpha * Time
        losses.append(-ll + (0.00 * t))
        # print(f"  Seed {seed}: LL={ll:.4f}, Time/Iter={t:.4f}s, Iters={n_iter}")
        sys.stdout.flush()

    # Aggregate
    mean_ll = np.mean(ll_scores)
    mean_time = np.mean(times)
    mean_iters = np.mean(iters)

    loss = np.mean(losses)

    print(f"Trial Result: Mean LL: {mean_ll:.4f} | Loss: {loss:.4f}")
    return loss


def run_optuna(n_trials=100, model_path='../models/best_model.pt', study_name="latent_hyperparameter_optimization_study"):
    """
    Perform hyperparameter optimization for latent active learning using Optuna.
    """
    # Load decoder once (shared across all trials)
    decoder, embeddings, latent_dim = load_decoder_model(model_path, DEVICE)
    print(f"Loaded decoder with latent_dim={latent_dim}")

    def objective(trial):
        config = load_config()

        # ===== BALD Acquisition =====
        config['bald']['tau'] = trial.suggest_float("bald_tau", 0.05, 1.0)
        # config['bald']['n_mc_samples'] = trial.suggest_int("bald_n_mc_samples", 10, 100)
        config['bald']['sampling_temperature'] = trial.suggest_float("bald_sampling_temperature", 1.0, 2.0)

        # Epsilon-greedy exploration
        # config['bald']['epsilon'] = trial.suggest_float("bald_epsilon", 0.0, 0.3)
        # config['bald']['epsilon_decay'] = trial.suggest_float("bald_epsilon_decay", 0.8, 1.0)

        # Weighted BALD (Targeted Boundary Exploration)
        # config['bald']['use_weighted_bald'] = trial.suggest_categorical("bald_use_weighted_bald", [True, False])
        config['bald']['weighted_bald_sigma'] = trial.suggest_float("bald_weighted_bald_sigma", 0.01, 0.2)

        # Diversity penalty
        # config['bald']['diversity_weight'] = trial.suggest_float("bald_diversity_weight", 0.0, 1.0)

        # ===== Acquisition Strategy =====
        # config['acquisition']['strategy'] = trial.suggest_categorical("acquisition_strategy", ["bald", "quasi-random", "random"])
        config['acquisition']['n_quasi_random'] = trial.suggest_int("acquisition_n_quasi_random", 0, 15)

        # ===== BALD Optimization (Test Selection) =====
        # config['bald_optimization']['n_restarts'] = trial.suggest_int("bald_n_restarts", 3, 20)
        # config['bald_optimization']['n_iters_per_restart'] = trial.suggest_int("bald_n_iters_per_restart", 10, 50)
        # config['bald_optimization']['lr_adam'] = trial.suggest_float("bald_lr_adam", 0.005, 0.1, log=True)
        # config['bald_optimization']['lr_sgd'] = trial.suggest_float("bald_lr_sgd", 0.001, 0.05, log=True)
        # config['bald_optimization']['switch_to_sgd_at'] = trial.suggest_float("bald_switch_to_sgd_at", 0.5, 1.0)
        config['bald_optimization']['plateau_patience'] = trial.suggest_int("bald_plateau_patience", 2, 10)
        config['bald_optimization']['plateau_threshold'] = trial.suggest_float("bald_plateau_threshold", 0.0001, 0.01, log=True)

        # ===== Variational Inference =====
        # config['vi']['n_mc_samples'] = trial.suggest_int("vi_n_mc_samples", 100, 2000)
        config['vi']['noise_std'] = trial.suggest_float("vi_noise_std", 0.1, 2.0)
        # config['vi']['learning_rate'] = trial.suggest_float("vi_lr", 0.005, 0.1, log=True)
        # config['vi']['optimizer_type'] = trial.suggest_categorical("vi_optimizer_type", ["adam", "sgd"])
        # config['vi']['max_iters'] = trial.suggest_int("vi_max_iters", 50, 300)
        # config['vi']['convergence_tol'] = trial.suggest_float("vi_convergence_tol", 1e-6, 1e-3, log=True)
        config['vi']['patience'] = trial.suggest_int("vi_patience", 3, 15)
        # config['vi']['grad_clip'] = trial.suggest_float("vi_grad_clip", 1.0, 20.0)

        # KL Annealing
        # config['vi']['kl_annealing']['enabled'] = trial.suggest_categorical("vi_kl_annealing_enabled", [True, False])
        config['vi']['kl_annealing']['start_weight'] = trial.suggest_float("vi_kl_start_weight", 0.001, 0.1, log=True)
        config['vi']['kl_annealing']['end_weight'] = trial.suggest_float("vi_kl_end_weight", 0.05, 1.0, log=True)
        config['vi']['kl_annealing']['duration'] = trial.suggest_int("vi_kl_duration", 5, 50)
        config['vi']['kl_annealing']['schedule'] = trial.suggest_categorical("vi_kl_schedule", ["linear", "cosine", "step", "logistic"])

        # ===== Prior Initialization =====
        # config['prior']['init_std'] = trial.suggest_float("prior_init_std", 0.3, 1.5)
        # config['prior']['mean_noise_std'] = trial.suggest_float("prior_mean_noise_std", 0.1, 0.5)

        # ===== Stopping Criteria =====
        # config['stopping']['budget'] = trial.suggest_int("stopping_budget", 20, 100)

        # Uncertainty-based stopping
        # config['stopping']['uncertainty_enabled'] = trial.suggest_categorical("stopping_uncertainty_enabled", [True, False])
        # config['stopping']['uncertainty_threshold'] = trial.suggest_float("stopping_uncertainty_threshold", 0.01, 0.1)

        # BALD score stopping
        # config['stopping']['bald_enabled'] = trial.suggest_categorical("stopping_bald_enabled", [True, False])
        # config['stopping']['bald_threshold'] = trial.suggest_float("stopping_bald_threshold", 0.01, 0.3)
        # config['stopping']['bald_patience'] = trial.suggest_int("stopping_bald_patience", 2, 10)

        # ===== Metrics =====
        # Note: grid_resolution affects evaluation accuracy but not learning
        # config['metrics']['grid_resolution'] = trial.suggest_int("metrics_grid_resolution", 8, 20)

        return evaluate_pipeline(config, decoder, embeddings, latent_dim)

    print("Starting Optuna Optimization for Latent Active Learning...")

    # storage_path = "./latent_optuna_journal.log"
    # storage = JournalStorage(JournalFileBackend(storage_path))
    # Use SQLite on Windows to avoid symlink privilege issues
    # Save in active_learning directory as requested
    storage = "sqlite:///active_learning/latent_optuna.db"

    # study_name is now passed as an argument
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)
    return study.best_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Tune Latent Active Learning Hyperparameters')
    parser.add_argument('--study-name', type=str, default='latent_hyperparameter_optimization_study',
                        help='Name for the Optuna study')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                        help='Path to decoder model checkpoint')

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    best_params = run_optuna(n_trials=args.n_trials, model_path=args.model, study_name=args.study_name)

    print("\nFinal Best Parameters:")
    print(json.dumps(best_params, indent=2))

    with open("best_latent_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\nBest hyperparameters saved to 'best_latent_hyperparameters.json'")
