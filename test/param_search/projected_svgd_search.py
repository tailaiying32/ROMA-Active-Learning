"""
Optuna Hyperparameter Search for Projected SVGD Active Learning.

Searches over Projected SVGD posterior inference parameters to minimize
a composite loss function prioritizing accuracy > efficiency > stability.

Uses stratified user sampling by feasibility (easy/medium/hard) to ensure
generalization across user difficulty levels.

Hyperparameters searched:
    - projected_svgd.step_decay.{start_lr, end_lr_ratio, duration}
    - projected_svgd.kl_annealing.{start_weight, end_ratio, duration}

Fixed parameters:
    - bald.tau_schedule: 0.5 -> 0.2 (linear over budget)
    - projected_svgd.n_slices: 20
    - projected_svgd.eigen_smoothing: 0.5
    - projected_svgd.max_iters: 100
    - projected_svgd.kernel_type: imq
    - posterior.n_particles: 50

Loss function components:
    1. Final IoU (accuracy) - weight: 1.0
    2. Efficiency penalty (late convergence) - weight: 0.15
    3. Stability penalty (sustained degradation) - weight: 0.05

Usage:
    python -m active_learning.test.param_search.projected_svgd_search --n-trials 50 --budget 100
"""

import sys
import os
import json
import time
import numpy as np
from typing import List, Tuple, Dict

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
from active_learning.src.metrics import compute_reachability_metrics
from infer_params.training.model import LevelSetDecoder
from infer_params.training.level_set_torch import (
    create_evaluation_grid,
    evaluate_level_set_batched,
)

# Difficulty buckets for stratified sampling
DIFFICULTY_BUCKETS = {
    'hard': (0.30, 0.50),    # Sparse feasible region
    'medium': (0.50, 0.70),  # Typical case
    'easy': (0.70, 0.90),    # Large feasible region
}


def compute_feasibility(decoder: LevelSetDecoder, z: torch.Tensor, grid: torch.Tensor) -> float:
    """Compute fraction of grid points that are feasible for user z."""
    with torch.no_grad():
        lower, upper, weights, pres, blob = decoder.decode_from_embedding(z.unsqueeze(0))
        logits = evaluate_level_set_batched(grid, lower, upper, weights, pres, blob)
        return (logits > 0).float().mean().item()


def compute_loss(iou_values: List[float], budget: int) -> float:
    """
    Compute loss prioritizing: accuracy > efficiency > stability.

    Components:
    1. Final IoU (accuracy) - PRIMARY, weight=1.0
    2. Efficiency penalty (late convergence) - SECONDARY, weight=0.15
    3. Stability penalty (sustained degradation after warmup) - TERTIARY, weight=0.05

    Returns:
        Loss value (lower is better, Optuna minimizes)
    """
    # 1. ACCURACY: Final IoU (average last 5 to reduce noise)
    final_iou = np.mean(iou_values[-5:]) if len(iou_values) >= 5 else iou_values[-1]

    # 2. EFFICIENCY: When did we reach 80% of final IoU?
    threshold = 0.8 * final_iou
    convergence_iter = next(
        (i for i, v in enumerate(iou_values) if v >= threshold),
        len(iou_values)
    )
    # Normalize: 0 = converged instantly, 1 = converged at last iter
    # Only penalize if convergence happens after 50% of budget
    efficiency_penalty = max(0, (convergence_iter / budget) - 0.5) * 2

    # 3. STABILITY: Time spent below running maximum after warmup
    warmup = 15
    if len(iou_values) > warmup:
        post_warmup = iou_values[warmup:]
        running_max = post_warmup[0]
        degradation_iters = 0

        for iou in post_warmup[1:]:
            if iou < running_max - 0.02:  # Below peak by margin
                degradation_iters += 1
            else:
                running_max = max(running_max, iou)

        stability_penalty = degradation_iters / len(post_warmup)
    else:
        stability_penalty = 0.0

    # Combined loss (negative final_iou because we want to maximize it)
    loss = -final_iou + 0.15 * efficiency_penalty + 0.05 * stability_penalty
    return loss


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


def build_user_pool(
    decoder: LevelSetDecoder,
    embeddings: torch.Tensor,
    users_per_bucket: int = 10,
) -> Dict[str, List[torch.Tensor]]:
    """
    Pre-sample users into difficulty buckets based on feasibility fraction.

    Args:
        decoder: LevelSetDecoder model
        embeddings: All training embeddings
        users_per_bucket: Number of users to sample per difficulty level

    Returns:
        Dict mapping difficulty name to list of user latent codes
    """
    print("Building stratified user pool...")

    # Create evaluation grid for feasibility check (coarse for speed)
    config = load_config()
    prior_gen = LatentPriorGenerator(config, decoder, verbose=False)
    joint_limits = prior_gen.anatomical_limits
    grid_lowers = [joint_limits[j][0] for j in prior_gen.joint_names]
    grid_uppers = [joint_limits[j][1] for j in prior_gen.joint_names]

    check_grid = create_evaluation_grid(
        torch.tensor(grid_lowers, device=DEVICE, dtype=torch.float32),
        torch.tensor(grid_uppers, device=DEVICE, dtype=torch.float32),
        resolution=8,
        device=DEVICE,
    )

    # Compute feasibility for all embeddings
    feasibilities = []
    for i, z in enumerate(embeddings):
        feas = compute_feasibility(decoder, z, check_grid)
        feasibilities.append((i, feas))

    # Sort into buckets
    user_pool = {name: [] for name in DIFFICULTY_BUCKETS}

    for name, (low, high) in DIFFICULTY_BUCKETS.items():
        candidates = [(i, f) for i, f in feasibilities if low <= f <= high]
        if len(candidates) < users_per_bucket:
            print(f"  Warning: only {len(candidates)} users in {name} bucket (wanted {users_per_bucket})")
            # Expand range slightly if needed
            candidates = [(i, f) for i, f in feasibilities if low - 0.1 <= f <= high + 0.1]

        # Randomly sample from candidates
        np.random.shuffle(candidates)
        selected = candidates[:users_per_bucket]
        user_pool[name] = [embeddings[i].clone() for i, _ in selected]
        print(f"  {name.capitalize()} bucket: {len(user_pool[name])} users (feas {low:.0%}-{high:.0%})")

    return user_pool


def run_single_user(
    config: dict,
    decoder: LevelSetDecoder,
    embeddings: torch.Tensor,
    gt_z: torch.Tensor,
    budget: int,
    trial: optuna.Trial = None,
    step_offset: int = 0,
    difficulty: str = "unknown",
) -> List[float]:
    """
    Run one active learning trial for a specific user.

    Args:
        config: Configuration dict
        decoder: LevelSetDecoder model
        embeddings: Training embeddings (for prior generation)
        gt_z: Ground truth user latent code
        budget: Number of queries
        trial: Optuna trial for pruning
        step_offset: Step offset for pruning reports
        difficulty: Difficulty label for logging

    Returns:
        List of IoU values, length = budget
    """
    log_every = config.get('search', {}).get('log_every', 5)
    log_every = max(1, int(log_every))
    print(f"    {difficulty.capitalize()} user: starting {budget} steps")
    start_time = time.perf_counter()

    # Setup environment for this user
    prior_gen = LatentPriorGenerator(config, decoder, verbose=False)
    joint_limits = prior_gen.anatomical_limits
    grid_lowers = [joint_limits[j][0] for j in prior_gen.joint_names]
    grid_uppers = [joint_limits[j][1] for j in prior_gen.joint_names]

    def to_tensor(x):
        return torch.tensor(x, device=DEVICE, dtype=torch.float32)

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

    # Evaluation grid
    eval_res = config.get('metrics', {}).get('grid_resolution', 12)
    test_grid = create_evaluation_grid(
        to_tensor(grid_lowers), to_tensor(grid_uppers),
        eval_res, DEVICE,
    )

    # Prior (use population mean)
    prior = prior_gen.get_prior(gt_z, embeddings=embeddings)

    # Oracle
    n_joints = len(prior_gen.joint_names)
    oracle = LatentOracle(decoder=decoder, ground_truth_z=gt_z, n_joints=n_joints)

    # Bounds
    bounds = get_bounds_from_config(config, DEVICE)

    # Create posterior
    posterior = LatentUserDistribution(
        latent_dim=decoder.latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE,
    )

    # Build learner
    learner = build_learner(
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        bounds=bounds,
        config=config,
    )

    iou_values = []

    for i in range(budget):
        learner.step(verbose=False)

        # Compute IoU
        mean_z = learner.posterior.mean
        iou, _, _, _ = compute_reachability_metrics(
            decoder=decoder,
            ground_truth_params=ground_truth_params,
            posterior_mean=mean_z.unsqueeze(0),
            test_grid=test_grid,
        )
        iou_values.append(iou)

        # Report to Optuna for pruning
        if trial is not None:
            current_loss = compute_loss(iou_values, budget)
            trial.report(current_loss, step=step_offset + i)
            if trial.should_prune():
                print(f"      [PRUNED] at step {step_offset + i}")
                raise optuna.TrialPruned()

        if (i + 1) % log_every == 0 or i == 0 or (i + 1) == budget:
            elapsed = time.perf_counter() - start_time
            print(f"      step {i + 1:3d}/{budget} | IoU={iou:.4f} | {elapsed:.1f}s")

    return iou_values


def evaluate_trial(
    config: dict,
    decoder: LevelSetDecoder,
    embeddings: torch.Tensor,
    user_pool: Dict[str, List[torch.Tensor]],
    budget: int,
    trial: optuna.Trial = None,
) -> float:
    """
    Run trials on stratified users (easy/medium/hard) and return combined loss.

    Returns:
        Combined loss (lower is better)
    """
    all_iou_curves = []

    for idx, difficulty in enumerate(['hard', 'medium', 'easy']):
        # Pick a user from this difficulty bucket (rotate through pool)
        user_list = user_pool[difficulty]
        user_idx = trial.number % len(user_list) if trial else 0
        gt_z = user_list[user_idx]

        try:
            iou_values = run_single_user(
                config, decoder, embeddings, gt_z, budget,
                trial=trial,
                step_offset=idx * budget,
                difficulty=difficulty,
            )
            all_iou_curves.append(iou_values)

            loss = compute_loss(iou_values, budget)
            print(f"    {difficulty.capitalize()} user: loss={loss:.4f}, final_iou={iou_values[-1]:.4f}")

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"    {difficulty.capitalize()} user FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise optuna.TrialPruned(f"{difficulty} user failed: {e}")

    # Compute mean loss across difficulty levels (equal weight)
    losses = [compute_loss(curve, budget) for curve in all_iou_curves]
    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))

    print(f"  Trial result: loss = {mean_loss:.4f} +/- {std_loss:.4f}")
    return mean_loss


def objective(
    trial: optuna.Trial,
    decoder: LevelSetDecoder,
    embeddings: torch.Tensor,
    user_pool: Dict[str, List[torch.Tensor]],
    budget: int,
) -> float:
    """Optuna objective function. Returns loss (Optuna minimizes)."""
    config = load_config()

    # --- Tau schedule (fixed, not tuned) ---
    config['bald']['tau'] = 0.2
    config['bald']['tau_schedule'] = {
        'start': 0.5,
        'end': 0.2,
        'duration': budget,
        'schedule': 'linear',
    }

    # --- Projected SVGD parameters ---
    # Step decay: linear schedule from start_lr to end_lr over lr_duration iterations
    start_lr = trial.suggest_float("start_lr", 0.01, 0.1, log=True)
    end_lr_ratio = trial.suggest_float("end_lr_ratio", 0.1, 0.5)
    end_lr = start_lr * end_lr_ratio
    lr_duration = trial.suggest_int("lr_duration", 20, 100)

    # KL annealing: start_weight increases to end_weight over kl_duration iterations
    kl_start = trial.suggest_float("kl_start_weight", 0.05, 0.5, log=True)
    kl_end_ratio = trial.suggest_float("kl_end_ratio", 1.0, 4.0)
    kl_end = kl_start * kl_end_ratio
    kl_duration = trial.suggest_int("kl_duration", 20, 100)

    config['projected_svgd'] = config.get('projected_svgd', {})
    config['projected_svgd']['step_decay'] = {
        'enabled': True,
        'schedule': 'linear',
        'start_lr': start_lr,
        'end_lr': end_lr,
        'duration': lr_duration,
    }
    config['projected_svgd']['max_iters'] = 100  # Fixed
    config['projected_svgd']['n_slices'] = 20  # Fixed
    config['projected_svgd']['eigen_smoothing'] = 0.5  # Fixed
    config['projected_svgd']['max_eigenweight'] = 3.0  # Fixed
    config['projected_svgd']['variance_threshold'] = 0.95  # Fixed
    config['projected_svgd']['kernel_type'] = 'imq'  # Fixed
    config['projected_svgd']['kl_annealing'] = {
        'enabled': True,
        'start_weight': kl_start,
        'end_weight': kl_end,
        'duration': kl_duration,
        'schedule': 'linear',
    }

    # Force projected_svgd posterior + prior_boundary acquisition
    config['posterior'] = config.get('posterior', {})
    config['posterior']['method'] = 'projected_svgd'
    config['posterior']['n_particles'] = 50  # Fixed
    config['acquisition'] = config.get('acquisition', {})
    config['acquisition']['strategy'] = 'prior_boundary'

    # Log trial params
    print(f"\n--- Trial {trial.number} ---")
    print(f"  tau: 0.5 -> 0.2 (fixed)")
    print(f"  lr: {start_lr:.4f} -> {end_lr:.4f} (duration={lr_duration})")
    print(f"  kl_annealing: {kl_start:.4f} -> {kl_end:.4f} (duration={kl_duration})")
    print(f"  n_slices: 20, eigen_smoothing: 0.5 (fixed)")
    sys.stdout.flush()

    loss = evaluate_trial(config, decoder, embeddings, user_pool, budget, trial=trial)
    return loss


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Optuna Hyperparameter Search for Projected SVGD Active Learning'
    )
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials for this worker')
    parser.add_argument('--budget', type=int, default=100,
                        help='Number of queries per active learning run')
    parser.add_argument('--study-name', type=str, default='projected_svgd_search',
                        help='Optuna study name')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                        help='Path to decoder model checkpoint')
    parser.add_argument('--storage', type=str,
                        default='sqlite:///active_learning/optuna_studies.db',
                        help='Optuna storage URL (SQLite path)')
    parser.add_argument('--users-per-bucket', type=int, default=10,
                        help='Number of users to pre-sample per difficulty bucket')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials: {args.n_trials}, Budget: {args.budget}")
    sys.stdout.flush()

    # Load model once (shared across all trials)
    decoder, embeddings, latent_dim = load_decoder_and_embeddings(args.model, DEVICE)

    # Build stratified user pool (replaces fixed seeds)
    user_pool = build_user_pool(decoder, embeddings, users_per_bucket=args.users_per_bucket)

    # Create / resume study with pruner
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
            interval_steps=10,
        ),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, decoder, embeddings, user_pool, args.budget),
        n_trials=args.n_trials,
    )

    # Report best result
    best = study.best_trial
    print(f"\nBest Trial #{best.number}")
    print(f"  Loss = {best.value:.4f}")
    print(f"  Params: {json.dumps(best.params, indent=2)}")

    # Save best params
    out_path = f"best_projected_svgd_params_{args.study_name}.json"
    with open(out_path, 'w') as f:
        json.dump({
            'study_name': args.study_name,
            'best_trial': best.number,
            'best_loss': best.value,
            'params': best.params,
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
