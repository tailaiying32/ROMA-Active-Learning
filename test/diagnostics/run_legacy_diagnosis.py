"""
Diagnostic script for the Legacy Active Learning Pipeline.

Runs a single trial with extensive diagnostics enabled and generates:
1. Posterior concentration plots (std dev over time).
2. BALD score landscape visualizations.
3. Weighted BALD gate activation analysis.
4. VI convergence metrics.
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from copy import deepcopy

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.config import load_config, DEVICE
from active_learning.src.legacy.prior_generation import PriorGenerator
from active_learning.src.legacy.sample_user import UserGenerator
from active_learning.src.legacy.oracle import Oracle
from active_learning.src.legacy.active_learning_pipeline import ActiveLearner
from active_learning.src.diagnostics import Diagnostics

# Configuration for plots
PLOT_DIR = "active_learning/images/diagnostics/legacy"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_posterior_concentration(history, joint_names, save_dir):
    """Plot posterior standard deviation evolution for each joint."""
    iterations = [s.iteration for s in history]
    
    for joint in joint_names:
        lower_stds = [s.posterior_stds[joint]['lower'] for s in history]
        upper_stds = [s.posterior_stds[joint]['upper'] for s in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, lower_stds, label='Lower Limit Std', marker='o', markersize=3)
        plt.plot(iterations, upper_stds, label='Upper Limit Std', marker='o', markersize=3)
        
        # Add reference lines
        plt.axhline(y=0.4, color='r', linestyle='--', label='Initial Prior Std (0.4)')
        plt.axhline(y=0.1, color='g', linestyle=':', label='Target Std (0.1)')
        
        plt.xlabel('Iteration')
        plt.ylabel('Posterior Std (radians)')
        plt.title(f'Posterior Concentration: {joint}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'posterior_std_{joint}.png'))
        plt.close()

def plot_bald_optimization_stats(history, save_dir):
    """Plot Weighted BALD gate activation and p_mean distribution."""
    # Flatten stats: each iteration has multiple restarts
    all_restarts = []
    
    for snap in history:
        for stat in snap.bald_opt_stats:
            stat_copy = stat.copy()
            stat_copy['iteration'] = snap.iteration
            all_restarts.append(stat_copy)
            
    if not all_restarts:
        print("No BALD optimization stats found.")
        return

    iterations = [r['iteration'] for r in all_restarts]
    p_means = [r['p_mean'] for r in all_restarts]
    gates = [r['gate'] for r in all_restarts]
    
    # Scatter: Iteration vs p_mean (colored by gate)
    plt.figure(figsize=(12, 6))
    sc = plt.scatter(iterations, p_means, c=gates, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(sc, label='Gate Value')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Candidate p_mean')
    plt.title('Weighted BALD: Candidate p_means and Gate Activation')
    plt.savefig(os.path.join(save_dir, 'weighted_bald_gate.png'))
    plt.close()

def plot_vi_metrics(history, save_dir):
    """Plot ELBO, Likelihood, and KL/Regularizer over iterations."""
    iterations = [s.iteration for s in history]
    elbos = [s.elbo_history[-1] if s.elbo_history else 0 for s in history]
    likelihoods = [s.likelihood for s in history]
    kls = [s.kl_divergence for s in history] # This is weighted KL (regularizer term)
    
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, likelihoods, label='Likelihood (LL)', color='blue')
    plt.plot(iterations, kls, label='Weighted KL (Reg)', color='orange')
    plt.plot(iterations, elbos, label='ELBO (LL - Reg)', color='green', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Variational Inference Convergence Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'vi_metrics.png'))
    plt.close()

def get_plot_schedule(budget):
    """
    Generate iteration indices to plot.
    Rule:
    - 0-49: Every iteration
    - 50+: Every 10 iterations
    """
    schedule = []
    for i in range(budget):
        if i < 50:
            schedule.append(i)
        else:
            if i % 10 == 0:
                schedule.append(i)
    return schedule

def compute_landscape_at_iter(learner, iteration, res=30):
    """
    Compute BALD scores for all 6 pairs at a specific iteration.
    Returns: Dict {pair_idx: (X, Y, Z, j1_name, j2_name, bounds)}
    """
    config = learner.config
    joint_names = config['prior']['joint_names']
    pairs = config['prior']['pair_names'] # List of [j1, j2]
    
    bounds = learner.posterior.anatomical_limits
    
    # Pre-compute centers for all joints (fix unused dims to center)
    centers = [(bounds[j][0] + bounds[j][1])/2 for j in joint_names]
    
    true_checker = learner.oracle.ground_truth
    results = {}
    
    for pair_idx, pair in enumerate(pairs):
        j1, j2 = pair[0], pair[1]
        idx1 = joint_names.index(j1)
        idx2 = joint_names.index(j2)
        
        b1 = bounds[j1]
        b2 = bounds[j2]
        
        x = torch.linspace(b1[0], b1[1], res)
        y = torch.linspace(b2[0], b2[1], res)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Create grid points
        # (res, res, n_joints)
        grid = torch.tensor(centers, device=DEVICE).repeat(res, res, 1)
        grid[:, :, idx1] = X.to(DEVICE)
        grid[:, :, idx2] = Y.to(DEVICE)
        
        flat_points = grid.reshape(-1, len(joint_names))
        
        with torch.no_grad():
             scores = learner.bald.compute_score(flat_points, iteration=iteration).cpu().numpy()
        
        # Compute True H for this grid
        true_h = true_checker.h_value(flat_points.cpu().numpy())
        Z_true = true_h.reshape(res, res)
             
        Z = scores.reshape(res, res)
        
        results[pair_idx] = {
            'X': X.numpy(),
            'Y': Y.numpy(),
            'Z': Z,
            'Z_true': Z_true, # Store true H
            'j1': j1,
            'j2': j2,
            'extent': [b1[0], b1[1], b2[0], b2[1]]
        }
        
    return results

def plot_master_grid(landscapes_history, save_dir):
    """
    Plot one giant vertical figure.
    Rows = Scheduled Iterations
    Cols = 6 Pairs
    Uses a single global color scale with horizontal color bars at top and bottom.
    """
    if not landscapes_history:
        return

    iterations = sorted(landscapes_history.keys())
    n_rows = len(iterations)
    n_cols = 6 # Fixed 6 pairs

    # Compute global vmin/vmax across ALL iterations and pairs
    global_vmin = float('inf')
    global_vmax = float('-inf')
    for it in iterations:
        data = landscapes_history[it]
        for pair_idx in range(n_cols):
            if pair_idx in data:
                global_vmin = min(global_vmin, data[pair_idx]['Z'].min())
                global_vmax = max(global_vmax, data[pair_idx]['Z'].max())

    # Handle case where no data
    if global_vmin == float('inf'):
         return

    norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)

    # Leave vertical space for color bars: bottom and top
    cbar_height = 0.02  # fraction of figure height
    cbar_pad = 0.04     # padding between color bar and plots
    title_pad = 0.03    # extra padding for the suptitle above the top color bar

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 3 + 2.5),  # extra height for color bars
    )
    # Ensure axes is 2D even if n_rows == 1
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, it in enumerate(iterations):
        data = landscapes_history[it]

        for pair_idx in range(n_cols):
            ax = axes[row_idx, pair_idx]

            if pair_idx not in data: 
                ax.set_visible(False)
                continue
            
            pdata = data[pair_idx]
            
            # Plot Heatmap
            im = ax.imshow(
                pdata['Z'].T, 
                origin='lower', 
                extent=pdata['extent'], 
                cmap='viridis', 
                aspect='auto',
                vmin=global_vmin, 
                vmax=global_vmax
            )
            
            # Overlay Ground Truth Boundary (h=0)
            try:
                ax.contour(
                    pdata['X'], pdata['Y'], pdata['Z_true'],
                    levels=[0],
                    colors='white',
                    linestyles='dashed',
                    linewidths=1.5
                )
            except Exception as e:
                print(f"Warning: Could not plot contour for pair {pair_idx}: {e}")

            if row_idx == 0:
                ax.set_title(f"{pdata['j1']}\nvs {pdata['j2']}")
                
            if pair_idx == 0:
                ax.set_ylabel(f"Iter {it}")

    # Adjust layout to leave room for color bars and title
    bottom_margin = cbar_height + cbar_pad + 0.02
    top_margin = 1.0 - (cbar_height + cbar_pad + title_pad + 0.02)
    fig.subplots_adjust(left=0.06, right=0.96, bottom=bottom_margin, top=top_margin,
                        hspace=0.35, wspace=0.30)

    # Create ScalarMappable for the color bars
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    # Bottom color bar
    cax_bottom = fig.add_axes([0.15, 0.01, 0.70, cbar_height])
    cb_bottom = fig.colorbar(sm, cax=cax_bottom, orientation='horizontal')
    cb_bottom.set_label('BALD Score')

    # Top color bar
    cax_top = fig.add_axes([0.15, top_margin + cbar_pad, 0.70, cbar_height])
    cb_top = fig.colorbar(sm, cax=cax_top, orientation='horizontal')
    cax_top.xaxis.set_ticks_position('top')
    cax_top.xaxis.set_label_position('top')
    cb_top.set_label('BALD Score')

    # Title above everything
    fig.suptitle('Legacy BALD Score Landscape (Global Scale)',
                 fontsize=14, fontweight='bold',
                 y=top_margin + cbar_pad + cbar_height + title_pad)

    save_path = os.path.join(save_dir, "master_landscape_grid.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved master grid to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--strategy", type=str, default=None,
                        choices=['bald', 'random', 'quasi_random', 'canonical',
                                 'gp', 'grid', 'heuristic', 'version_space'],
                        help="Test selection strategy")
    parser.add_argument("--posterior-method", type=str, default='vi',
                        choices=['vi', 'ensemble', 'svgd', 'sliced_svgd', 'projected_svgd'],
                        help="Posterior inference method (vi, ensemble, svgd, sliced_svgd)")
    args = parser.parse_args()
    
    # Load config and enable all adaptive/diagnostic features
    config = load_config(os.path.join(os.path.dirname(__file__), '../../configs/legacy.yaml'))
    
    # FORCE CONFIG FOR DIAGNOSTICS
    config['stopping']['budget'] = args.budget
    
    # Strategy override
    if args.strategy is not None:
        config['acquisition']['strategy'] = args.strategy
        
    # Posterior method override (not fully impl in legacy factory yet?)
    # Legacy mostly assumes VI via config 'prior' params, but let's check config structure
    # For now, we mainly trust the pipeline to use what it has.
    
    strategy_name = config.get('acquisition', {}).get('strategy', 'bald')
    print(f"Running Legacy Diagnostics (Seed {args.seed}, Budget {args.budget}, Strategy {strategy_name})...")

    
    # Ensure Weighted BALD and Annealing are ON
    config['bald']['use_weighted_bald'] = True
    config['vi']['kl_annealing']['enabled'] = True
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    prior_gen = PriorGenerator(config)
    prior = prior_gen.get_prior()
    
    # Generate ground truth user
    user_gen = UserGenerator(config, prior_gen.joint_names, prior_gen.anatomical_limits, prior_gen.pairs)
    true_limits, true_bumps, true_checker = user_gen.generate_user()
    
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    oracle = Oracle(true_checker, prior_gen.joint_names)
    
    # Initialize Learner
    learner = ActiveLearner(prior, posterior, oracle, config)
    
    if learner.diagnostics is None:
        print("Error: Diagnostics not initialized in ActiveLearner!")
        return

    # Run Loop manually to inspect landscape at intervals
    print("Starting Active Learning Loop...")
    landscape_data = {}
    plot_schedule = get_plot_schedule(args.budget)
    print(f"Scheduled plots for iterations: {plot_schedule}")

    for i in range(args.budget):
        learner.step(verbose=True)
        
        if i in plot_schedule:
             # Capture Landscape
            #  print(f"  Capturing landscape for iter {i}...")
             iteration_data = compute_landscape_at_iter(learner, i)
             
             landscape_data[i] = iteration_data
             
             # Diagnostic: Check max BALD in grid vs what optimizer found
             avg_max_grid_bald = 0.0
             global_max_grid_bald = -1.0
             count = 0
             for _, pair_data in iteration_data.items():
                 # Z is (res, res)
                 m = pair_data['Z'].max()
                 global_max_grid_bald = max(global_max_grid_bald, m)
                 avg_max_grid_bald += m
                 count += 1
             if count > 0:
                 avg_max_grid_bald /= count
             
             # Get the actual selected score from the learner
             selected_score = learner.results[-1].bald_score if learner.results else 0.0
             
             print(f"  [Diagnostic] Grid Max BALD: {global_max_grid_bald:.5f} (Avg Max: {avg_max_grid_bald:.5f}) vs Optimizer Selected: {selected_score:.5f}")
             
    # Create Master Grid
    print("Generating Master Landscape Grid...")
    plot_master_grid(landscape_data, PLOT_DIR)
             
    # Generate post-run plots
    history = learner.diagnostics.history
    
    print("Generating plots...")
    plot_posterior_concentration(history, learner.posterior.joint_names, PLOT_DIR)
    plot_bald_optimization_stats(history, PLOT_DIR)
    plot_vi_metrics(history, PLOT_DIR)
    
    # Print final summary report
    learner.diagnostics.print_final_report()
    
    print(f"Diagnostics complete. Results saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
