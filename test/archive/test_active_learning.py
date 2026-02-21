import copy
import sys
import os
import torch
import numpy as np
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
from visualization_utils import Visualizer
import itertools

def run_active_learning_demo():
    print("=== Starting Active Learning Run ===")

    # 1. Setup Configuration
    config = load_config()
    print(f"Using device: {DEVICE}")

    # 2. Setup Prior / Problem Structure
    print("Initializing Prior Generator and Problem Structure...")
    prior_gen = PriorGenerator(config)
    joint_names = prior_gen.joint_names
    pairs = prior_gen.pairs
    anatomical_limits = prior_gen.anatomical_limits

# 3. Generate "True" User (Ground Truth)
    print("Generating Ground Truth User...")
    generator = UserGenerator(
        config,
        joint_names=joint_names,
        anatomical_limits=anatomical_limits,
        pairs=pairs
    )
    true_limits, true_bumps, true_checker = generator.generate_user()

    print(f"Generated {len(joint_names)} joints: {joint_names}")
    print(f"Generated {len(true_bumps)} pairs with bumps.")

    # 2. Setup Prior (Simulated LLM/RAG output based on ground truth)
    print("Initializing Prior (Simulating LLM estimation)...")
    prior = prior_gen.get_prior(
        true_limits,
        true_bumps,
    )

    # 4. Setup Active Learning Components

    # Posterior: Starts same as prior
    posterior = copy.deepcopy(prior)

    # Oracle: Wraps the true user
    oracle = Oracle(ground_truth=true_checker, joint_names=joint_names)

    # Active Learner
    learner = ActiveLearner(
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        config=config
    )

    # 5. Run Loop
    print(f"\nStarting Active Learning Loop...")

    # Initialize Visualizer with true_limits for diagnostics
    viz_dir = 'images/'
    visualizer = Visualizer(save_dir=viz_dir, joint_names=joint_names, true_limits=true_limits)

    results = []
    iteration = 0

    while True:
        # Check stopping criteria
        should_stop, reason = learner.check_stopping_criteria()
        if should_stop:
            print(f"\nStopping Criteria Met: {reason}")
            break

        result = learner.step(verbose=True)
        results.append(result)
        # Pass prior and true_checker for diagnostic tracking
        visualizer.log_iteration(iteration, posterior, result, prior=prior, true_checker=true_checker)
        iteration += 1

    print("\n=== Run Finished ===")
    print(f"Total queries: {len(results)}")

    # Generate Plots
    print(f"Saving visualizations to {viz_dir}...")
    visualizer.plot_metrics()
    visualizer.plot_param_error_over_time()
    visualizer.save_history()
    visualizer.plot_joint_evolution_and_queries(true_limits)

    # Generate diagnostic plots and report
    visualizer.plot_diagnostics()
    visualizer.plot_gradient_vs_tau_analysis()
    visualizer.print_diagnostic_report()
    # visualizer.plot_reachability(true_checker, posterior)


    # Print final posterior parameters (mean)
    print("\nFinal Posterior Means:")
    for joint in joint_names:
        stats = posterior.get_limit_stats(joint)
        print(f"  {joint}: [{stats['lower_mean']:.3f}, {stats['upper_mean']:.3f}] "
              f"(std: lower={stats['lower_std']:.3f}, upper={stats['upper_std']:.3f})")

    print("\nTrue Limits:")
    for joint in joint_names:
        print(f"  {joint}: [{true_limits[joint][0]:.3f}, {true_limits[joint][1]:.3f}]")


   # --- 1. Compute Parameter Mean Error (MAE) ---
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

    # # Build a grid of points across the global box limits
    # grid_res = 15
    # grid_axes = [np.linspace(anatomical_limits[j][0], anatomical_limits[j][1], grid_res) for j in joint_names]
    # mesh = np.meshgrid(*grid_axes, indexing='ij')
    # grid_points = np.stack([m.flatten() for m in mesh], axis=-1)

    # # Evaluate ground truth reachability
    # gt_reach = (true_checker.h_value(grid_points) >= 0)

    # # --- Posterior Mean Reachability Checker ---
    # mean_joint_limits = {}
    # for joint in joint_names:
    #     stats = posterior.get_limit_stats(joint)
    #     mean_joint_limits[joint] = (stats['lower_mean'], stats['upper_mean'])
    # mean_checker = FeasibilityChecker(
    #     joint_limits=mean_joint_limits,
    #     pairwise_constraints={},
    #     config=posterior.config if hasattr(posterior, 'config') else None
    # )
    # mean_reach = (mean_checker.h_value(grid_points) >= 0)
    # mean_mismatch_error = (gt_reach != mean_reach).mean()

    # # --- Posterior Sampled Reachability Error ---
    # mismatch_errors = []
    # all_post_reach = []
    # for checker in posterior_checkers:
    #     post_reach = (checker.h_value(grid_points) >= 0)
    #     all_post_reach.append(post_reach)
    #     mismatch = (gt_reach != post_reach)
    #     mismatch_errors.append(mismatch.mean())
    # sample_mismatch_error = np.mean(mismatch_errors)

    # --- Uncertainty Evaluation ---
    # Calculate variance of predictions across samples
    # all_post_reach_stack = np.stack(all_post_reach, axis=0)
    # mean_pred = np.mean(all_post_reach_stack, axis=0)
    # prediction_variance = mean_pred * (1 - mean_pred)
    # uncertainty_score = np.mean(prediction_variance)

    print("\n=== Final Evaluation ===")
    print(f"Parameter Mean Absolute Error: {param_error:.4f}")
    print(f"Average Parameter Std Dev: {avg_param_std:.4f}")
    # print(f"Reachability Mean Mismatch Error: {mean_mismatch_error:.4f}")
    # print(f"Reachability Sample Mismatch Error: {sample_mismatch_error:.4f}")
    # print(f"Average Reachability Prediction Uncertainty: {uncertainty_score:.6f}")
    print("=================================\n")

if __name__ == "__main__":
    run_active_learning_demo()
