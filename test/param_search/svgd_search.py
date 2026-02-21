"""
Optuna Hyperparameter Search for SVGD Active Learning.

Searches over SVGD posterior inference and tau schedule parameters to maximize
AUC of the IoU curve (area under IoU-vs-iteration). Uses the same pipeline
components as the main active learning loop via the factory pattern.

Hyperparameters searched:
    - bald.tau_schedule.{start, end, duration}
    - svgd.repulsive_scaling
    - svgd.kl_annealing.{start_weight, end_weight, duration}

Usage:
    # Local (single process)
    python -m active_learning.test.param_search.svgd_search --n-trials 50 --budget 100

    # Cluster (via SLURM launcher)
    bash active_learning/test/param_search/run_search.sh 50 10 100

    # Resume existing study
    python -m active_learning.test.param_search.svgd_search --n-trials 20 \
        --study-name svgd_search_20260127_143000 \
        --storage sqlite:///active_learning/svgd_search_20260127_143000.db
"""

import sys
import os
import time
import json
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Dict, Optional

import torch
import optuna

# Ensure project root is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from active_learning.src.config import load_config, get_bounds_from_config, DEVICE
from active_learning.src.factory import build_learner
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.metrics import compute_reachability_metrics
from infer_params.training.model import LevelSetDecoder
from infer_params.training.level_set_torch import (
    create_evaluation_grid,
    evaluate_level_set_batched,
)

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def load_decoder_and_embeddings(
    model_path: str, device: str = DEVICE
) -> Tuple[LevelSetDecoder, torch.Tensor, int]:
    """Load the LevelSetDecoder model and training embeddings from a checkpoint."""
    print(f"Loading decoder from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

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

    print(f"Loaded: latent_dim={latent_dim}, embeddings={num_samples}")
    return model, embeddings.to(device), latent_dim


def setup_environment(
    seed: int,
    config: dict,
    decoder: LevelSetDecoder,
    embeddings: torch.Tensor,
) -> Dict:
    """
    Setup shared environment for a single active learning run.

    Rejection-samples a ground truth user from embeddings (>33% feasible volume),
    creates prior via LatentPriorGenerator, oracle, bounds, and evaluation grid.

    Returns dict with keys: prior, gt_z, oracle, bounds, ground_truth_params, test_grid
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    prior_gen = LatentPriorGenerator(config, decoder, verbose=False)

    # Anatomical limits for grid construction
    joint_limits = prior_gen.anatomical_limits
    grid_lowers = [joint_limits[j][0] for j in prior_gen.joint_names]
    grid_uppers = [joint_limits[j][1] for j in prior_gen.joint_names]

    def to_tensor(x):
        return torch.tensor(x, device=DEVICE, dtype=torch.float32)

    # Rejection-sample ground truth user (>33% feasible volume)
    check_grid = create_evaluation_grid(
        to_tensor(grid_lowers), to_tensor(grid_uppers),
        resolution=8,
        device=DEVICE,
    )

    gt_z = None
    max_retries = 50
    for attempt in range(max_retries):
        idx = np.random.randint(0, len(embeddings))
        z = embeddings[idx]

        with torch.no_grad():
            lower, upper, weights, pres, blob = decoder.decode_from_embedding(z.unsqueeze(0))
            logits = evaluate_level_set_batched(check_grid, lower, upper, weights, pres, blob)
            feasible_frac = (logits > 0).float().mean().item()

        if feasible_frac > 0.33:
            gt_z = z.clone()
            break

    if gt_z is None:
        # Fallback: use last sampled user
        gt_z = z.clone()

    # Extract ground truth decoded parameters
    with torch.no_grad():
        gt_lower, gt_upper, gt_weights, gt_pres_logits, gt_blob_params = (
            decoder.decode_from_embedding(gt_z.unsqueeze(0))
        )
        ground_truth_params = {
            'box_lower': gt_lower.squeeze(0),
            'box_upper': gt_upper.squeeze(0),
            'box_weights': gt_weights.squeeze(0),
            'presence': torch.sigmoid(gt_pres_logits).squeeze(0),
            'blob_params': gt_blob_params.squeeze(0),
        }

    # Evaluation grid (fine resolution for metrics)
    eval_res = config.get('metrics', {}).get('grid_resolution', 12)
    test_grid = create_evaluation_grid(
        to_tensor(grid_lowers), to_tensor(grid_uppers),
        eval_res, DEVICE,
    )

    # Prior (with empirical covariance if configured)
    prior = prior_gen.get_prior(gt_z, embeddings=embeddings)

    # Oracle
    n_joints = len(prior_gen.joint_names)
    oracle = LatentOracle(decoder=decoder, ground_truth_z=gt_z, n_joints=n_joints)

    # Bounds
    bounds = get_bounds_from_config(config, DEVICE)

    return {
        'prior': prior,
        'gt_z': gt_z,
        'oracle': oracle,
        'bounds': bounds,
        'ground_truth_params': ground_truth_params,
        'test_grid': test_grid,
    }


def run_single_seed(
    config: dict,
    decoder: LevelSetDecoder,
    embeddings: torch.Tensor,
    seed: int,
    budget: int,
) -> List[float]:
    """
    Run one active learning trial and return the IoU at each iteration.

    Returns:
        List of IoU values, length = budget
    """
    env = setup_environment(seed, config, decoder, embeddings)

    # Create posterior as copy of prior (factory will convert to particles for SVGD)
    posterior = LatentUserDistribution(
        latent_dim=decoder.latent_dim,
        decoder=decoder,
        mean=env['prior'].mean.clone(),
        log_std=env['prior'].log_std.clone(),
        device=DEVICE,
    )

    learner = build_learner(
        decoder=decoder,
        prior=env['prior'],
        posterior=posterior,
        oracle=env['oracle'],
        bounds=env['bounds'],
        config=config,
    )

    iou_values = []
    for i in range(budget):
        learner.step(verbose=False)

        # Compute IoU using the same function as diagnostics
        mean_z = learner.posterior.mean
        iou, _, _, _ = compute_reachability_metrics(
            decoder=decoder,
            ground_truth_params=env['ground_truth_params'],
            posterior_mean=mean_z.unsqueeze(0),
            test_grid=env['test_grid'],
        )
        iou_values.append(iou)

    return iou_values


def evaluate_trial(
    config: dict,
    decoder: LevelSetDecoder,
    embeddings: torch.Tensor,
    seeds: List[int],
    budget: int,
) -> float:
    """
    Run multiple seeds and return mean AUC of IoU curve.

    AUC = mean(iou_values) since iterations are evenly spaced.
    Returns mean AUC across seeds.
    """
    auc_scores = []

    for seed in seeds:
        try:
            iou_values = run_single_seed(config, decoder, embeddings, seed, budget)
            auc = float(np.mean(iou_values))
            auc_scores.append(auc)
            print(f"    Seed {seed}: AUC={auc:.4f} (final IoU={iou_values[-1]:.4f})")
        except Exception as e:
            print(f"    Seed {seed} FAILED: {e}")
            auc_scores.append(0.0)

    mean_auc = float(np.mean(auc_scores))
    std_auc = float(np.std(auc_scores))
    print(f"  Trial result: AUC = {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc


def objective(
    trial: optuna.Trial,
    decoder: LevelSetDecoder,
    embeddings: torch.Tensor,
    latent_dim: int,
    budget: int,
    seeds: List[int],
) -> float:
    """Optuna objective function. Returns negative AUC (Optuna minimizes)."""
    config = load_config()

    # --- Tau schedule ---
    tau_start = trial.suggest_float("tau_start", 0.3, 1.5)
    tau_end = trial.suggest_float("tau_end", 0.05, max(0.06, tau_start - 0.05))
    tau_duration = trial.suggest_int("tau_duration", 20, 100)

    config['bald']['tau'] = tau_end  # Fallback value
    config['bald']['tau_schedule'] = {
        'start': tau_start,
        'end': tau_end,
        'duration': tau_duration,
        'schedule': 'linear',
    }

    # --- SVGD parameters ---
    repulsive_scaling = trial.suggest_float("repulsive_scaling", 0.0, 1.5)

    kl_start = trial.suggest_float("kl_start_weight", 0.001, 1.0, log=True)
    kl_end = trial.suggest_float("kl_end_weight", kl_start, 5.0, log=True)
    kl_duration = trial.suggest_int("kl_duration", 10, 80)

    config['svgd'] = config.get('svgd', {})
    config['svgd']['repulsive_scaling'] = repulsive_scaling
    config['svgd']['kl_annealing'] = {
        'enabled': True,
        'start_weight': kl_start,
        'end_weight': kl_end,
        'duration': kl_duration,
        'schedule': 'linear',
    }

    # Force SVGD posterior + prior_boundary acquisition
    config['posterior'] = config.get('posterior', {})
    config['posterior']['method'] = 'svgd'
    config['acquisition'] = config.get('acquisition', {})
    config['acquisition']['strategy'] = 'prior_boundary'

    # Log trial params
    print(f"\n--- Trial {trial.number} ---")
    print(f"  tau: {tau_start:.3f} -> {tau_end:.3f} over {tau_duration} iters")
    print(f"  repulsive_scaling: {repulsive_scaling:.3f}")
    print(f"  kl_annealing: {kl_start:.4f} -> {kl_end:.4f} over {kl_duration} iters")
    sys.stdout.flush()

    auc = evaluate_trial(config, decoder, embeddings, seeds, budget)
    return -auc  # Optuna minimizes


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Optuna Hyperparameter Search for SVGD Active Learning'
    )
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials for this worker')
    parser.add_argument('--budget', type=int, default=100,
                        help='Number of queries per active learning run')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds per trial')
    parser.add_argument('--study-name', type=str, default='svgd_param_search',
                        help='Optuna study name')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                        help='Path to decoder model checkpoint')
    parser.add_argument('--storage', type=str,
                        default='sqlite:///active_learning/svgd_optuna.db',
                        help='Optuna storage URL (SQLite path)')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials: {args.n_trials}, Budget: {args.budget}, Seeds: {args.seeds}")
    sys.stdout.flush()

    # Load model once (shared across all trials)
    decoder, embeddings, latent_dim = load_decoder_and_embeddings(args.model, DEVICE)

    # Select seeds
    seeds = DEFAULT_SEEDS[:args.seeds]

    # Create / resume study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, decoder, embeddings, latent_dim, args.budget, seeds),
        n_trials=args.n_trials,
    )

    # Report best result
    best = study.best_trial
    print(f"\nBest Trial #{best.number}")
    print(f"  AUC = {-best.value:.4f}")
    print(f"  Params: {json.dumps(best.params, indent=2)}")

    # Save best params
    out_path = f"best_svgd_params_{args.study_name}.json"
    with open(out_path, 'w') as f:
        json.dump({
            'study_name': args.study_name,
            'best_trial': best.number,
            'best_auc': -best.value,
            'params': best.params,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
