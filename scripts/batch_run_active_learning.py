import copy
import sys
import os
import torch
import numpy as np
import time
from pathlib import Path

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from active_learning.src.legacy.user_distribution import UserDistribution
from active_learning.src.legacy.feasibility_checker import FeasibilityChecker
from active_learning.src.legacy.oracle import Oracle
from active_learning.src.legacy.active_learning_pipeline import ActiveLearner
from active_learning.src.config import load_config, DEVICE
from active_learning.src.legacy.sample_user import UserGenerator
from active_learning.src.legacy.prior_generation import PriorGenerator

def run_single_trial(seed):
    # 1. Setup Configuration
    config = load_config()

    # Set seed for reproducibility of this trial
    np.random.seed(seed)
    torch.manual_seed(seed)
    config['seed'] = seed
    config['user_generation']['seed'] = seed

    # 2. Setup Prior / Problem Structure
    prior_gen = PriorGenerator(config)
    joint_names = prior_gen.joint_names
    pairs = prior_gen.pairs
    anatomical_limits = prior_gen.anatomical_limits

    # 3. Generate "True" User (Ground Truth)
    generator = UserGenerator(
        config,
        joint_names=joint_names,
        anatomical_limits=anatomical_limits,
        pairs=pairs
    )
    true_limits, true_bumps, true_checker = generator.generate_user()

    # 4. Setup Prior
    prior = prior_gen.get_prior(true_limits, true_bumps)

    # 5. Setup Active Learning Components
    posterior = copy.deepcopy(prior)
    oracle = Oracle(ground_truth=true_checker, joint_names=joint_names)
    learner = ActiveLearner(
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        config=config
    )

    # 6. Run Loop
    start_time = time.time()
    results = []
    iteration = 0

    while True:
        should_stop, reason = learner.check_stopping_criteria()
        if should_stop:
            break

        result = learner.step(verbose=False)
        results.append(result)
        iteration += 1

    total_time = time.time() - start_time
    total_iters = len(results)
    avg_time_per_iteration = total_time / total_iters if total_iters > 0 else 0

    # 7. Calculate Metrics

    # --- Parameter Error & Std ---
    param_error = 0.0
    param_std_sum = 0.0
    for joint in joint_names:
        stats = posterior.get_limit_stats(joint)
        l_mean, u_mean = stats['lower_mean'], stats['upper_mean']
        l_true, u_true = true_limits[joint]
        param_error += (abs(l_mean - l_true) + abs(u_mean - u_true)) / 2
        param_std_sum += (stats['lower_std'] + stats['upper_std']) / 2
    param_error /= len(joint_names)
    avg_param_std = param_std_sum / len(joint_names)

    # --- Reachability Error Evaluation ---
    n_posterior_samples = 50
    posterior_samples = posterior.sample(n_posterior_samples)
    posterior_checkers = []
    for theta in posterior_samples:
        cpu_theta = {'joint_limits': {}, 'pairwise_constraints': {}}
        for j, (l, u) in theta['joint_limits'].items():
            l_val = l.detach().cpu().item() if isinstance(l, torch.Tensor) else l
            u_val = u.detach().cpu().item() if isinstance(u, torch.Tensor) else u
            cpu_theta['joint_limits'][j] = (l_val, u_val)
        for pair, constraint in theta['pairwise_constraints'].items():
            cpu_constraint = {'bumps': []}
            if 'box' in constraint:
                cpu_constraint['box'] = constraint['box']
            for bump in constraint['bumps']:
                cpu_bump = {}
                for k, v in bump.items():
                    if isinstance(v, torch.Tensor):
                        cpu_bump[k] = v.detach().cpu().numpy()
                    else:
                        cpu_bump[k] = v
                cpu_constraint['bumps'].append(cpu_bump)
            cpu_theta['pairwise_constraints'][pair] = cpu_constraint
        checker = FeasibilityChecker(
            joint_limits=cpu_theta['joint_limits'],
            pairwise_constraints=cpu_theta['pairwise_constraints'],
            config=posterior.config if hasattr(posterior, 'config') else None
        )
        posterior_checkers.append(checker)

    # Build a grid of points across the global box limits
    grid_res = 15
    grid_axes = [np.linspace(anatomical_limits[j][0], anatomical_limits[j][1], grid_res) for j in joint_names]
    mesh = np.meshgrid(*grid_axes, indexing='ij')
    grid_points = np.stack([m.flatten() for m in mesh], axis=-1)

    # Evaluate ground truth reachability
    gt_reach = (true_checker.h_value(grid_points) >= 0)

    # --- Posterior Mean Reachability Checker ---
    mean_joint_limits = {}
    for joint in joint_names:
        stats = posterior.get_limit_stats(joint)
        mean_joint_limits[joint] = (stats['lower_mean'], stats['upper_mean'])
    mean_checker = FeasibilityChecker(
        joint_limits=mean_joint_limits,
        pairwise_constraints={},
        config=posterior.config if hasattr(posterior, 'config') else None
    )
    mean_reach = (mean_checker.h_value(grid_points) >= 0)
    mean_mismatch_error = (gt_reach != mean_reach).mean()

    # --- Posterior Sampled Reachability Error ---
    mismatch_errors = []
    all_post_reach = []
    for checker in posterior_checkers:
        post_reach = (checker.h_value(grid_points) >= 0)
        all_post_reach.append(post_reach)
        mismatch = (gt_reach != post_reach)
        mismatch_errors.append(mismatch.mean())
    sample_mismatch_error = np.mean(mismatch_errors)

    # --- Uncertainty Evaluation ---
    all_post_reach_stack = np.stack(all_post_reach, axis=0)
    mean_pred = np.mean(all_post_reach_stack, axis=0)
    prediction_variance = mean_pred * (1 - mean_pred)
    uncertainty_score = np.mean(prediction_variance)

    return {
        'param_error': param_error,
        'mean_mismatch_error': mean_mismatch_error,
        'sample_mismatch_error': sample_mismatch_error,
        'avg_param_std': avg_param_std,
        'uncertainty_score': uncertainty_score,
        'avg_time_per_iteration': avg_time_per_iteration,
        'total_iters': total_iters
    }

def main():
    n_trials = 10
    seeds = np.random.randint(0, 10000, size=n_trials)

    all_metrics = {
        'param_error': [],
        'mean_mismatch_error': [],
        'sample_mismatch_error': [],
        'avg_param_std': [],
        'uncertainty_score': [],
        'avg_time_per_iteration': [],
        'total_iters': []
    }

    print(f"Starting batch run of {n_trials} trials...")

    for i, seed in enumerate(seeds):
        print(f"Running trial {i+1}/{n_trials} (seed={seed})...")
        try:
            metrics = run_single_trial(int(seed))

            for k, v in metrics.items():
                all_metrics[k].append(v)

            print(f"  Trial {i+1} metrics:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")

        except Exception as e:
            print(f"  Trial {i+1} failed with error: {e}")

    print("\n=== Batch Run Complete ===")
    print("Average Statistics:")
    for k, v in all_metrics.items():
        if v:
            avg_val = np.mean(v)
            std_val = np.std(v)
            print(f"  {k}: {avg_val:.4f} ± {std_val:.4f}")
        else:
            print(f"  {k}: N/A (no successful trials)")

if __name__ == "__main__":
    main()
