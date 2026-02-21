"""
Diagnostic script for the Latent Active Learning Pipeline.

Runs a single trial with extensive diagnostics enabled and generates:
1. Posterior concentration plots (latent z-std over time).
2. BALD score landscape visualizations (slice of latent space).
3. Weighted BALD gate activation analysis.
4. VI convergence metrics.

Two orthogonal configuration axes:
  - Strategy (--strategy): test selection method (bald, random, gp, grid, etc.)
    CLI names use hyphens (quasi-random, prior-boundary, version-space).
  - Posterior Method (--posterior-method): inference method (vi, ensemble, svgd, full_cov_vi)
    Defaults to the value in latent.yaml (posterior.method); CLI overrides when explicit.

Usage:
    # Default: strategy + posterior method both from config
    python run_latent_diagnosis.py --budget 40

    # BALD + Ensemble
    python run_latent_diagnosis.py --strategy bald --posterior-method ensemble --ensemble-size 5

    # Random + SVGD
    python run_latent_diagnosis.py --strategy random --posterior-method svgd --n-particles 20

    # quasi-random with config-driven posterior (e.g. posterior.method: svgd in YAML)
    python run_latent_diagnosis.py --strategy quasi-random --n-particles 30
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.factory import build_learner
from active_learning.src.utils import load_decoder_model
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.boundary_projection import compute_marginal_envelope, compute_marginal_envelope_for_decoder
from infer_params.training.dataset import LevelSetDataset
from infer_params.training.level_set_torch import create_evaluation_grid
from active_learning.src.metrics import compute_reachability_metrics, compute_ensemble_reachability_metrics, precompute_gt_metrics

# Configuration for plots
PLOT_DIR = "active_learning/images/diagnostics/latent"
os.makedirs(PLOT_DIR, exist_ok=True)

def compute_detailed_parameter_error(decoder, posterior_mean, gt_z):
    """Compute separated parameter errors."""
    with torch.no_grad():
        gt_lower, gt_upper, gt_weights, gt_pres_logits, gt_blob_params = decoder.decode_from_embedding(gt_z.unsqueeze(0))
        gt_pres = torch.sigmoid(gt_pres_logits)

        p_lower, p_upper, p_weights, p_pres_logits, p_blob_params = decoder.decode_from_embedding(posterior_mean.unsqueeze(0))
        p_pres = torch.sigmoid(p_pres_logits)

        # 1. Box Error (L2 norm of bounds and weights)
        box_err = (torch.norm(p_lower - gt_lower) + 
                  torch.norm(p_upper - gt_upper) + 
                  torch.norm(p_weights - gt_weights)).item()
        
        # 2. Presence Error (L2 norm of probabilities)
        pres_err = torch.norm(p_pres - gt_pres).item()

        # 3. Blob Error (L2 norm of params)
        blob_err = torch.norm(p_blob_params - gt_blob_params).item()

    return {
        'box_error': box_err,
        'presence_error': pres_err,
        'blob_error': blob_err
    }


def compute_parameter_error(decoder, posterior_mean, gt_z):
    """Compute parameter-space error via decoder forward pass."""
    with torch.no_grad():
        p_lower, p_upper, _, _, _ = decoder.decode_from_embedding(posterior_mean.unsqueeze(0))
        gt_lower, gt_upper, _, _, _ = decoder.decode_from_embedding(gt_z.unsqueeze(0))

        lower_error = torch.norm(p_lower - gt_lower).item()
        upper_error = torch.norm(p_upper - gt_upper).item()
        param_error = np.sqrt(lower_error**2 + upper_error**2)

    return param_error


def compute_uncertainty(posterior):
    """Compute mean posterior std in latent space."""
    std = torch.exp(posterior.log_std)
    return std.mean().item()

def plot_performance_metrics(history, save_dir):
    """Plot all performance metrics in a single 4x2 grid."""
    iterations = history['iteration']

    # Get ceiling values (upper bounds) if available
    ceiling_iou = history.get('ceiling_iou', None)
    ceiling_f1 = history.get('ceiling_f1', None)
    ceiling_boundary_acc = history.get('ceiling_boundary_acc', None)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))

    # 1. REACHABILITY METRICS [0,0]
    ax = axes[0, 0]
    iou_final = history['iou'][-1] if history['iou'] else 0
    f1_final = history['f1'][-1] if history['f1'] else 0
    ax.plot(iterations, history['iou'], label=f'IoU (final: {iou_final:.3f})', marker='.')
    ax.plot(iterations, history['f1'], label=f'F1 (final: {f1_final:.3f})', marker='.')
    # Add ceiling lines
    if ceiling_iou is not None:
        ax.axhline(y=ceiling_iou, color='blue', linestyle='--', alpha=0.7, label=f'Ceiling IoU: {ceiling_iou:.3f}')
    if ceiling_f1 is not None:
        ax.axhline(y=ceiling_f1, color='orange', linestyle='--', alpha=0.7, label=f'Ceiling F1: {ceiling_f1:.3f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Reachability Metrics (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. BOX ERROR [0,1]
    ax = axes[0, 1]
    box_final = history['box_error'][-1] if history['box_error'] else 0
    ax.plot(iterations, history['box_error'], label=f'Box Error (final: {box_final:.3f})', color='tab:orange')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('L2 Error')
    ax.set_title('Box Error (Main Joint Ranges)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. BLOB ERROR [1,0]
    ax = axes[1, 0]
    blob_final = history['blob_error'][-1] if history['blob_error'] else 0
    ax.plot(iterations, history['blob_error'], label=f'Blob Error (final: {blob_final:.3f})', color='tab:green')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('L2 Error')
    ax.set_title('Blob Error (Interference/Cutouts)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. PRESENCE ERROR [1,1]
    ax = axes[1, 1]
    pres_final = history['presence_error'][-1] if history['presence_error'] else 0
    ax.plot(iterations, history['presence_error'], label=f'Presence Error (final: {pres_final:.3f})', color='tab:red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('L2 Error (Probabilities)')
    ax.set_title('Presence Error (Active Blobs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. UNCERTAINTY [2,0]
    ax = axes[2, 0]
    unc_final = history['uncertainty'][-1] if history['uncertainty'] else 0
    ax.plot(iterations, history['uncertainty'], label=f'Uncertainty (final: {unc_final:.3f})', color='purple', marker='.')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Latent Std')
    ax.set_title('Uncertainty Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. LATENT ERROR [2,1]
    ax = axes[2, 1]
    if history.get('latent_error'):
        le_final = history['latent_error'][-1]
        ax.plot(iterations, history['latent_error'], label=f'Latent Error (final: {le_final:.3f})', color='tab:blue', marker='.')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('L2 Distance to GT z')
    ax.set_title('Latent Error (L2 to Ground Truth)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. BOUNDARY-LOCAL ACCURACY [3,0]
    ax = axes[3, 0]
    if history.get('boundary_accuracy'):
        ba_final = history['boundary_accuracy'][-1]
        ax.plot(iterations, history['boundary_accuracy'],
                label=f'Boundary Accuracy (final: {ba_final:.3f})', color='tab:cyan', marker='.')
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Boundary-Local Accuracy (Near GT Boundary)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [3,1] unused
    axes[3, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics.png'))
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
    plt.title('Variational Inference Convergence Metrics (Latent)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'vi_metrics.png'))
    plt.close()

def plot_bald_optimization_stats(history, save_dir):
    """Plot BALD/G-BALD optimization statistics.

    Handles both regular BALD (p_mean, gate) and G-BALD (bald, diversity, boundary) formats.
    """
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

    # Detect format: G-BALD has 'diversity' and 'boundary', regular BALD has 'p_mean'
    is_gbald = 'diversity' in all_restarts[0]

    iterations = [r['iteration'] for r in all_restarts]

    if is_gbald:
        # G-BALD format: plot component scores
        bald_scores = [r['bald'] for r in all_restarts]
        diversity_scores = [r['diversity'] for r in all_restarts]
        boundary_scores = [r['boundary'] for r in all_restarts]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # BALD component
        sc1 = axes[0].scatter(iterations, bald_scores, c=iterations, cmap='viridis', alpha=0.6, s=20)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('BALD Score')
        axes[0].set_title('G-BALD: BALD Component')
        axes[0].grid(True, alpha=0.3)

        # Diversity component
        sc2 = axes[1].scatter(iterations, diversity_scores, c=iterations, cmap='viridis', alpha=0.6, s=20)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Diversity Score')
        axes[1].set_title('G-BALD: Diversity Component')
        axes[1].grid(True, alpha=0.3)

        # Boundary component
        sc3 = axes[2].scatter(iterations, boundary_scores, c=iterations, cmap='viridis', alpha=0.6, s=20)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Boundary Score')
        axes[2].set_title('G-BALD: Boundary Component')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weighted_bald_gate.png'))
        plt.close()
    else:
        # Regular BALD format: plot p_mean vs gate
        p_means = [r.get('p_mean', 0.5) for r in all_restarts]
        gates = [r.get('gate', 1.0) for r in all_restarts]

        plt.figure(figsize=(12, 6))
        sc = plt.scatter(iterations, p_means, c=gates, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(sc, label='Gate Value')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Candidate p_mean')
        plt.title('Weighted BALD: Candidate p_means (Latent)')
        plt.savefig(os.path.join(save_dir, 'weighted_bald_gate.png'))
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

def compute_landscape_at_iter(learner, iteration, res=12, selected_test=None, query_history=None,
                               use_marginal_envelope=None, n_marginal_samples=None):
    """
    Compute BALD scores and predicted boundary for all 6 pairs at a specific iteration.

    If use_marginal_envelope=True, computes envelope projection instead of 2D slice.
    This shows the range of possible boundary positions when marginalizing over
    non-displayed dimensions.

    Z_pred contains raw level-set logits from the posterior mean (contour at 0 = predicted boundary).
    Returns: Dict {pair_idx: {X, Y, Z, Z_true, Z_pred, ...}, '_selected_test': ..., '_query_history': ...}
    """
    config = learner.config
    joint_names = config['prior']['joint_names']
    pairs = config['prior']['pair_names'] # List of [j1, j2]

    # Get envelope settings from config if not specified
    vis_config = config.get('visualization', {})
    if use_marginal_envelope is None:
        use_marginal_envelope = vis_config.get('use_marginal_envelope', False)
    if n_marginal_samples is None:
        n_marginal_samples = vis_config.get('n_marginal_samples', 50)

    bounds = learner.bounds

    # Pre-compute centers for all joints (fix unused dims to center)
    centers = (bounds[:, 0] + bounds[:, 1]) / 2 # Tensor (J,)
    
    results = {}
    
    # Pre-compute samples / decoded params once for landscape consistency across pairs
    from active_learning.src.ensemble.ensemble_bald import EnsembleBALD as _EnsembleBALDType
    from active_learning.src.svgd.particle_bald import ParticleBALD as _ParticleBALDType
    is_ensemble_bald = isinstance(learner.bald_calculator, _EnsembleBALDType)
    is_particle_bald = isinstance(learner.bald_calculator, _ParticleBALDType)

    # Posterior mean for predicted boundary (raw level-set logits, no tau)
    mean_z = learner.get_posterior().mean.unsqueeze(0)  # (1, latent_dim)

    with torch.no_grad():
        if is_ensemble_bald:
            K = len(learner.bald_calculator.posteriors)
            n_per_member = max(1, learner.bald_calculator.n_samples // K)
            member_decoded_params = learner.bald_calculator._sample_and_decode_all(n_per_member)
        elif is_particle_bald:
            zs = learner.bald_calculator.posterior.get_particles()
        else:
            zs = learner.bald_calculator.posterior.sample(learner.bald_calculator.n_samples)

        for pair_idx, pair in enumerate(pairs):
            j1, j2 = pair[0], pair[1]
            idx1 = joint_names.index(j1)
            idx2 = joint_names.index(j2)

            b1 = bounds[idx1]
            b2 = bounds[idx2]

            x = torch.linspace(b1[0], b1[1], res, device=DEVICE)
            y = torch.linspace(b2[0], b2[1], res, device=DEVICE)
            X, Y = torch.meshgrid(x, y, indexing='ij')

            # Create grid points (res, res, n_joints)
            grid = centers.unsqueeze(0).unsqueeze(0).repeat(res, res, 1)
            grid[:, :, idx1] = X
            grid[:, :, idx2] = Y

            flat_points = grid.reshape(-1, len(joint_names))

            scores = []
            true_h_list = []
            pred_probs = []

            batch_size = 100
            for i in range(0, len(flat_points), batch_size):
                 batch = flat_points[i:i+batch_size]

                 # BALD Score (ensemble vs single)
                 try:
                     if is_ensemble_bald:
                         s_batch = learner.bald_calculator.compute_score(
                             batch, member_decoded_params=member_decoded_params, iteration=iteration
                         )
                     else:
                         s_batch = learner.bald_calculator.compute_score(batch, zs=zs, iteration=iteration)
                     scores.append(s_batch.cpu())
                 except:
                     for p in batch:
                         if is_ensemble_bald:
                             s = learner.bald_calculator.compute_score(
                                 p, member_decoded_params=member_decoded_params, iteration=iteration
                             )
                         else:
                             s = learner.bald_calculator.compute_score(p, zs=zs, iteration=iteration)
                         scores.append(s.cpu().unsqueeze(0))

                 # Predicted boundary from posterior mean (raw level-set logits, no tau)
                 logits_mean = LatentFeasibilityChecker.batched_logit_values(learner.decoder, mean_z, batch)
                 pred_probs.append(logits_mean.squeeze(0).cpu())

                 # Ground Truth H
                 try:
                     h_batch = learner.oracle.ground_truth_checker.logit_value(batch)
                     true_h_list.append(h_batch.detach().cpu().squeeze())
                 except:
                     for p in batch:
                         h = learner.oracle.ground_truth_checker.logit_value(p)
                         true_h_list.append(h.detach().cpu().unsqueeze(0))

            scores = torch.cat(scores).numpy()
            Z = scores.reshape(res, res)

            true_h = torch.cat([t.view(-1) for t in true_h_list]).numpy()
            Z_true = true_h.reshape(res, res)

            # Predicted boundary grid (raw level-set logits from posterior mean)
            Z_pred = None
            if pred_probs:
                Z_pred = torch.cat(pred_probs).numpy().reshape(res, res)

            # Build base result dict
            pair_result = {
                'X': X.cpu().numpy(),
                'Y': Y.cpu().numpy(),
                'Z': Z,
                'Z_true': Z_true,
                'Z_pred': Z_pred,
                'j1': j1,
                'j2': j2,
                'idx1': idx1,
                'idx2': idx2,
                'extent': [b1[0].item(), b1[1].item(), b2[0].item(), b2[1].item()],
                'use_envelope': use_marginal_envelope,
            }

            # Compute marginal envelope if enabled
            if use_marginal_envelope:
                # Ground truth envelope
                gt_envelope = compute_marginal_envelope(
                    checker=learner.oracle.ground_truth_checker,
                    grid_j1=x, grid_j2=y,
                    idx1=idx1, idx2=idx2,
                    bounds=bounds,
                    n_samples=n_marginal_samples,
                    device=DEVICE
                )
                pair_result['gt_h_min'] = gt_envelope['h_min'].cpu().numpy()
                pair_result['gt_h_max'] = gt_envelope['h_max'].cpu().numpy()

                # Predicted envelope from posterior mean
                pred_envelope = compute_marginal_envelope_for_decoder(
                    decoder=learner.decoder,
                    z=mean_z.squeeze(0),
                    grid_j1=x, grid_j2=y,
                    idx1=idx1, idx2=idx2,
                    bounds=bounds,
                    n_samples=n_marginal_samples,
                    device=DEVICE
                )
                pair_result['pred_h_min'] = pred_envelope['h_min'].cpu().numpy()
                pair_result['pred_h_max'] = pred_envelope['h_max'].cpu().numpy()
                pair_result['gt_p_feasible'] = gt_envelope['p_feasible'].cpu().numpy()
                pair_result['pred_p_feasible'] = pred_envelope['p_feasible'].cpu().numpy()

            results[pair_idx] = pair_result

    # Attach global metadata for plotting
    results['_selected_test'] = selected_test.cpu().numpy() if selected_test is not None and torch.is_tensor(selected_test) else selected_test
    results['_query_history'] = query_history

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

            # Plot Heatmap with global normalization
            ax.imshow(
                pdata['Z'].T,
                origin='lower',
                extent=pdata['extent'],
                cmap='viridis',
                aspect='auto',
                vmin=global_vmin,
                vmax=global_vmax
            )

            # Overlay boundaries (envelope or slice based on configuration)
            use_env = pdata.get('use_envelope', False)
            has_gt_env = 'gt_p_feasible' in pdata
            if use_env and has_gt_env:
                # Envelope mode: show probability contours (p=0.25, 0.5, 0.75)
                # These always exist unlike h_min/h_max=0 which may not cross zero

                # Ground Truth probability contours
                try:
                    # p=0.5 contour (main boundary - 50% chance of feasibility)
                    ax.contour(
                        pdata['X'], pdata['Y'], pdata['gt_p_feasible'],
                        levels=[0.5],
                        colors='white',
                        linestyles='solid',
                        linewidths=1.5
                    )
                    # p=0.25 and p=0.75 contours (uncertainty band)
                    ax.contour(
                        pdata['X'], pdata['Y'], pdata['gt_p_feasible'],
                        levels=[0.25, 0.75],
                        colors='white',
                        linestyles='dashed',
                        linewidths=0.8
                    )
                except Exception as e:
                    print(f"Warning: Could not plot GT envelope for pair {pair_idx}: {e}")

                # Prediction probability contours
                if 'pred_p_feasible' in pdata:
                    try:
                        # p=0.5 contour (main boundary)
                        ax.contour(
                            pdata['X'], pdata['Y'], pdata['pred_p_feasible'],
                            levels=[0.5],
                            colors='red',
                            linestyles='solid',
                            linewidths=1.5
                        )
                        # p=0.25 and p=0.75 contours (uncertainty band)
                        ax.contour(
                            pdata['X'], pdata['Y'], pdata['pred_p_feasible'],
                            levels=[0.25, 0.75],
                            colors='red',
                            linestyles='dashed',
                            linewidths=0.8
                        )
                    except Exception as e:
                        print(f"Warning: Could not plot pred envelope for pair {pair_idx}: {e}")
            else:
                # Slice mode: single boundary contour (existing behavior)
                try:
                    ax.contour(
                        pdata['X'], pdata['Y'], pdata['Z_true'],
                        levels=[0],
                        colors='white',
                        linestyles='dashed',
                        linewidths=1.5
                    )
                except Exception as e:
                    print(f"Warning: Could not plot GT contour for pair {pair_idx}: {e}")

                # Overlay Predicted Boundary (level-set f=0, red solid)
                if pdata.get('Z_pred') is not None:
                    try:
                        ax.contour(
                            pdata['X'], pdata['Y'], pdata['Z_pred'],
                            levels=[0],
                            colors='red',
                            linestyles='solid',
                            linewidths=1.5
                        )
                    except Exception:
                        pass

            # Overlay Selected Test Point (red star)
            selected_test = data.get('_selected_test')
            if selected_test is not None:
                idx1 = pdata['idx1']
                idx2 = pdata['idx2']
                ax.plot(selected_test[idx1], selected_test[idx2],
                        marker='*', color='red', markersize=10,
                        markeredgecolor='white', markeredgewidth=0.5,
                        zorder=10)

            # Overlay Query History (forestgreen=feasible, magenta=infeasible)
            query_hist = data.get('_query_history')
            if query_hist:
                idx1 = pdata['idx1']
                idx2 = pdata['idx2']
                for qpt, qoutcome in query_hist:
                    if qoutcome > 0.5:
                        ax.plot(qpt[idx1], qpt[idx2],
                                marker='o', color='forestgreen', markersize=4,
                                markeredgecolor='white', markeredgewidth=0.3,
                                zorder=9)
                    else:
                        ax.plot(qpt[idx1], qpt[idx2],
                                marker='x', color='magenta', markersize=4,
                                markeredgecolor='white', markeredgewidth=0.3,
                                zorder=9)

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
    fig.suptitle('BALD Score Landscape (Global Scale)',
                 fontsize=14, fontweight='bold',
                 y=top_margin + cbar_pad + cbar_height + title_pad)

    save_path = os.path.join(save_dir, "bald_landscape.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved master grid to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--model", type=str, default="models/best_model.pt")
    parser.add_argument("--dataset", type=str, default="models/training_data.npz")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=['bald', 'gbald', 'kbald', 'random', 'quasi-random', 'prior-boundary',
                                 'multi-stage-warmup', 'canonical', 'gp', 'grid',
                                 'heuristic', 'version-space'],
                        help="Test selection strategy")
    parser.add_argument("--posterior-method", type=str, default=None,
                        choices=['vi', 'ensemble', 'svgd', 'full_cov_vi', 'sliced-svgd', 'projected-svgd', 'projected-svn'],
                        help="Posterior inference method (default: from config)")
    parser.add_argument("--ensemble-size", type=int, default=None,
                        help="Number of ensemble members (for --posterior-method ensemble)")
    parser.add_argument("--n-particles", type=int, default=None,
                        help="Number of SVGD particles (for --posterior-method svgd)")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Epsilon-greedy exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=None,
                        help="Epsilon decay factor per iteration")
    args = parser.parse_args()
    

    if args.seed is None:
        args.seed = np.random.randint(0, 1000)

    # Load config and force diagnostic settings
    config = load_config(os.path.join(os.path.dirname(__file__), '../../configs/latent.yaml'))
    config['stopping']['budget'] = args.budget
    config['bald']['use_weighted_bald'] = False
    config['vi']['kl_annealing']['enabled'] = False
    config['latent']['dataset_path'] = args.dataset # override dataset path if needed

    # Strategy override (Axis 1: test selection)
    if args.strategy is not None:
        config['acquisition']['strategy'] = args.strategy.replace('-', '_')

    # Posterior method override (Axis 2: inference method)
    if args.posterior_method is not None:
        config.setdefault('posterior', {})['method'] = args.posterior_method.replace('-', '_')
    posterior_method = config.get('posterior', {}).get('method', 'vi')
    if args.ensemble_size is not None:
        config.setdefault('ensemble', {})['ensemble_size'] = args.ensemble_size
    if args.n_particles is not None:
        config.setdefault('posterior', {})['n_particles'] = args.n_particles

    # Epsilon-greedy overrides
    if args.epsilon is not None:
        config.setdefault('acquisition', {})['epsilon'] = args.epsilon
    if args.epsilon_decay is not None:
        config.setdefault('acquisition', {})['epsilon_decay'] = args.epsilon_decay

    strategy_name = config.get('acquisition', {}).get('strategy', 'bald')
    display_name = strategy_name
    if posterior_method != 'vi':
        display_name += f" + {posterior_method}"
        if posterior_method == 'ensemble':
            display_name += f" (K={config.get('ensemble', {}).get('ensemble_size', 5)})"
        elif posterior_method in ('svgd', 'sliced_svgd', 'projected_svgd'):
            display_name += f" (N={config.get('posterior', {}).get('n_particles', 50)})"
    effective_eps = config.get('acquisition', {}).get('epsilon', 0.0)
    if effective_eps > 0:
        display_name += f" \u03b5={effective_eps}"
    print(f"Running Latent Diagnostics (Seed {args.seed}, Budget {args.budget}, "
          f"Strategy: {display_name})...")
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Loading model...")
    decoder, embeddings, _ = load_decoder_model(args.model, DEVICE)
    
    # Select random user from embeddings as Ground Truth (with rejection sampling)
    # We want a user with a non-trivial feasibility region.
    prior_gen = LatentPriorGenerator(config, decoder)
    
    # Pre-compute grid for feasibility check (coarse)
    joint_limits = prior_gen.anatomical_limits
    grid_lowers = [joint_limits[j][0] for j in prior_gen.joint_names]
    grid_uppers = [joint_limits[j][1] for j in prior_gen.joint_names]
    check_grid = create_evaluation_grid(
        torch.tensor(grid_lowers, device=DEVICE), 
        torch.tensor(grid_uppers, device=DEVICE), 
        resolution=8, # Coarse resolution for speed
        device=DEVICE
    )
    
    max_retries = 50
    gt_idx = -1
    gt_z = None
    
    print(f"Selecting Ground Truth User (min volume check)...")
    for attempt in range(max_retries):
        idx = np.random.randint(0, len(embeddings))
        z = embeddings[idx]
        
        # Check volume
        with torch.no_grad():
            decoded = decoder.decode_from_embedding(z.unsqueeze(0))
            # unpack tuple
            lower, upper, weights, pres, blob = decoded
            
            # Evaluate on grid
            from infer_params.training.level_set_torch import evaluate_level_set_batched
            
            # check_grid is (N, 4)
            # lower/upper etc are (1, 4) from decoder
            # This broadcasts correctly inside evaluate_level_set_batched
            logits = evaluate_level_set_batched(
                check_grid, lower, upper, weights, pres, blob
            )
            # Feasible if logit > 0
            feasible_frac = (logits > 0).float().mean().item()
            
        if feasible_frac > 0.33: # At least 1% of anatomical space is feasible
            gt_idx = idx
            gt_z = z.clone()
            print(f"Selected User {gt_idx} on attempt {attempt+1} (Feasible Volume: {feasible_frac:.1%})")
            break
        else:
            if attempt % 10 == 0:
                print(f"Attempt {attempt+1}: User {idx} too restrictive ({feasible_frac:.1%}), retrying...")

    if gt_z is None:
        print("Warning: Could not find feasible user after max retries. Using last.")
        gt_idx = idx
        gt_z = z.clone()

    # Setup Priors/Posterior
    # prior_gen was already init above

    # Setup Priors/Posterior
    prior = prior_gen.get_prior(gt_z, embeddings=embeddings)

    posterior = LatentUserDistribution(
        latent_dim=prior.latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )
    
    bounds = get_bounds_from_config(config, DEVICE)
    oracle = LatentOracle(decoder, gt_z, bounds.shape[0])
    
    # Initialize Learner (factory routes all strategies through LatentActiveLearner)
    learner = build_learner(
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        bounds=bounds,
        config=config,
        embeddings=embeddings
    )
    
    is_ensemble = isinstance(learner.posterior, list)
    is_svgd = hasattr(learner.posterior, 'get_particles')

    if learner.diagnostics is None:
        print("Error: Diagnostics not initialized!")
        return
        
    # --- Setup Evaluation Environment for Metrics ---
    print("Setting up evaluation grid and ground truth params...")
    # Decode GT params
    with torch.no_grad():
        gt_lower, gt_upper, gt_weights, gt_pres_logits, gt_blob_params = decoder.decode_from_embedding(gt_z.unsqueeze(0))
        ground_truth_params = {
            'box_lower': gt_lower.squeeze(0),
            'box_upper': gt_upper.squeeze(0),
            'box_weights': gt_weights.squeeze(0),
            'presence': torch.sigmoid(gt_pres_logits).squeeze(0),
            'blob_params': gt_blob_params.squeeze(0)
        }
    
    # Create Test Grid
    joint_limits = prior_gen.anatomical_limits
    eval_res = config.get('metrics', {}).get('grid_resolution', 12)
    
    def to_tensor(x):
        return torch.tensor(x, device=DEVICE, dtype=torch.float32)

    grid_lowers = [joint_limits[j][0] for j in prior_gen.joint_names]
    grid_uppers = [joint_limits[j][1] for j in prior_gen.joint_names]
    test_grid = create_evaluation_grid(
        to_tensor(grid_lowers), to_tensor(grid_uppers), eval_res, DEVICE
    )

    # Precompute GT metrics once (avoid recomputing every iteration)
    cached_gt = precompute_gt_metrics(
        ground_truth_params=ground_truth_params,
        test_grid=test_grid,
        device=DEVICE
    )

    # Compute ceiling IoU: what IoU do we get if we use the perfect latent (gt_z)?
    # This measures decoder reconstruction fidelity and serves as the upper bound.
    ceiling_iou, _, ceiling_f1, ceiling_boundary_acc = compute_reachability_metrics(
        decoder=decoder, ground_truth_params=ground_truth_params,
        posterior_mean=gt_z.unsqueeze(0), test_grid=test_grid,
        cached_gt=cached_gt
    )
    print(f"Ceiling IoU (decode(gt_z) vs ground_truth): {ceiling_iou:.4f}")
    print(f"Ceiling F1: {ceiling_f1:.4f}, Ceiling Boundary Acc: {ceiling_boundary_acc:.4f}")

    # Init Metrics History
    metrics_history = {
        'iteration': [],
        'iou': [], 'f1': [], 'boundary_accuracy': [],
        'param_error': [], 'box_error': [], 'presence_error': [], 'blob_error': [],
        'uncertainty': [], 'latent_error': [],
        'ceiling_iou': ceiling_iou,  # Store as scalar for plotting
        'ceiling_f1': ceiling_f1,
        'ceiling_boundary_acc': ceiling_boundary_acc
    }
    
    # --- Rich Visualizations (LatentVisualizer) ---
    visualizer = None
    try:
        from active_learning.test.visualization_utils import LatentVisualizer, plot_latent_comparison

        joint_names = config['prior']['joint_names']
        true_limits = {}
        with torch.no_grad():
            gt_lower_v, gt_upper_v, _, _, _ = decoder.decode_from_embedding(gt_z.unsqueeze(0))
            gt_lower_np = gt_lower_v.squeeze().cpu().numpy()
            gt_upper_np = gt_upper_v.squeeze().cpu().numpy()
            for i, name in enumerate(joint_names):
                true_limits[name] = (float(gt_lower_np[i]), float(gt_upper_np[i]))

        bounds_np = bounds.cpu().numpy()
        bounds_dict = {name: (float(bounds_np[i, 0]), float(bounds_np[i, 1])) for i, name in enumerate(joint_names)}
        grid_resolution = config.get('metrics', {}).get('grid_resolution', 12)

        visualizer = LatentVisualizer(
            save_dir=PLOT_DIR,
            joint_names=joint_names,
            decoder=decoder,
            true_limits=true_limits,
            ground_truth_params=ground_truth_params,
            resolution=grid_resolution,
            anatomical_limits=bounds_dict,
            true_checker=oracle.ground_truth_checker if hasattr(oracle, 'ground_truth_checker') else None
        )

        rep_posterior_init = learner.get_posterior()
        visualizer.log_initial_state(rep_posterior_init, gt_z)
    except Exception as e:
        print(f"Warning: Could not initialize LatentVisualizer: {e}")

    print("Starting Loop...")
    landscape_data = {}
    plot_schedule = get_plot_schedule(args.budget)
    print(f"Scheduled plots for iterations: {plot_schedule}")

    # Initial Metric Log (Iter 0 before any queries)
    def log_metrics(iter_idx):
        if is_ensemble:
            posteriors = learner.posterior  # List of posteriors
            iou, _, f1, boundary_acc = compute_ensemble_reachability_metrics(
                decoder=decoder, ground_truth_params=ground_truth_params,
                posteriors=posteriors, test_grid=test_grid,
                cached_gt=cached_gt
            )
            representative = learner.get_posterior()
            param_err = compute_parameter_error(decoder, representative.mean, gt_z)
            detailed_err = compute_detailed_parameter_error(decoder, representative.mean, gt_z)
            unc = float(np.mean([compute_uncertainty(p) for p in posteriors]))
        elif is_svgd:
            # SVGD: use particle mean as the representative posterior
            representative = learner.get_posterior()
            iou, _, f1, boundary_acc = compute_reachability_metrics(
                decoder=decoder, ground_truth_params=ground_truth_params,
                posterior_mean=representative.mean.unsqueeze(0), test_grid=test_grid,
                cached_gt=cached_gt
            )
            param_err = compute_parameter_error(decoder, representative.mean, gt_z)
            detailed_err = compute_detailed_parameter_error(decoder, representative.mean, gt_z)
            # Uncertainty from particle spread
            particles = learner.posterior.get_particles()
            unc = float(particles.std(dim=0).mean().item())
        else:
            posterior_ref = learner.get_posterior()
            iou, _, f1, boundary_acc = compute_reachability_metrics(
                decoder=decoder, ground_truth_params=ground_truth_params,
                posterior_mean=posterior_ref.mean.unsqueeze(0), test_grid=test_grid,
                cached_gt=cached_gt
            )
            param_err = compute_parameter_error(decoder, posterior_ref.mean, gt_z)
            detailed_err = compute_detailed_parameter_error(decoder, posterior_ref.mean, gt_z)
            unc = compute_uncertainty(posterior_ref)

        metrics_history['iteration'].append(iter_idx)
        metrics_history['iou'].append(iou)
        metrics_history['f1'].append(f1)
        metrics_history['boundary_accuracy'].append(boundary_acc)
        metrics_history['param_error'].append(param_err)
        metrics_history['box_error'].append(detailed_err['box_error'])
        metrics_history['presence_error'].append(detailed_err['presence_error'])
        metrics_history['blob_error'].append(detailed_err['blob_error'])
        metrics_history['uncertainty'].append(unc)

        # Latent error (L2 distance to ground truth z)
        representative = learner.get_posterior()
        latent_err = torch.norm(representative.mean - gt_z).item()
        metrics_history['latent_error'].append(latent_err)

        # Simplified console log
        print(f"  [Metrics] IoU: {iou:.3f}, ParamErr: {param_err:.3f}, Unc: {unc:.3f}")

    # Log initial state
    log_metrics(0)
    plot_schedule = get_plot_schedule(args.budget)
    print(f"Scheduled plots for iterations: {plot_schedule}")

    for i in range(args.budget):
        learner.step(verbose=True)

        # Log metrics after step
        log_metrics(i+1)

        # Log to LatentVisualizer
        if visualizer is not None:
            rep_posterior = learner.get_posterior()
            visualizer.log_iteration(
                iteration=i+1,
                posterior=rep_posterior,
                result=learner.results[-1],
                ground_truth_z=gt_z
            )

        if i in plot_schedule:
             # Build query history and selected test for overlay
             selected_test = None
             query_history = []
             if learner.results:
                 last = learner.results[-1]
                 selected_test = last.test_point
                 for r in learner.results:
                     pt_np = r.test_point.cpu().numpy() if torch.is_tensor(r.test_point) else r.test_point
                     query_history.append((pt_np, r.outcome))

             iteration_data = compute_landscape_at_iter(
                 learner, i,
                 selected_test=selected_test,
                 query_history=query_history
             )
             landscape_data[i] = iteration_data

             # Diagnostic: Check max BALD in grid vs what optimizer found
             avg_max_grid_bald = 0.0
             global_max_grid_bald = -1.0
             count = 0
             for key, pair_data in iteration_data.items():
                 if isinstance(key, str) and key.startswith('_'):
                     continue
                 m = pair_data['Z'].max()
                 global_max_grid_bald = max(global_max_grid_bald, m)
                 avg_max_grid_bald += m
                 count += 1
             if count > 0:
                 avg_max_grid_bald /= count
             
             # Get the actual selected score from the learner
             # learner.results[-1].bald_score relates to the *selected* test point
             selected_score = learner.results[-1].bald_score if learner.results else 0.0
             
             print(f"  [Diagnostic] Grid Max BALD: {global_max_grid_bald:.5f} (Avg Max: {avg_max_grid_bald:.5f}) vs Optimizer Selected: {selected_score:.5f}")
             
    # Create Master Grid
    print("Generating Master Landscape Grid...")
    plot_master_grid(landscape_data, PLOT_DIR)

    # Post-run plots
    history = learner.diagnostics.history
    print("Generating plots...")

    plot_bald_optimization_stats(history, PLOT_DIR)
    plot_vi_metrics(history, PLOT_DIR)
    plot_performance_metrics(metrics_history, PLOT_DIR)
    
    # Rich Visualizations from LatentVisualizer
    if visualizer is not None:
        print("Generating rich visualizations...")
        try:
            visualizer.plot_information_gain()
            cap_to_anatomical = config.get('visualization', {}).get('cap_joint_evolution_to_anatomical', False)
            visualizer.plot_joint_evolution_and_queries(true_limits, cap_to_anatomical=cap_to_anatomical, anatomical_limits=bounds_dict)
            visualizer.plot_latent_evolution()
        except Exception as e:
            print(f"Warning: Error generating rich visualizations: {e}")

        # Final boundary comparison
        try:
            rep_posterior_final = learner.get_posterior()
            plot_latent_comparison(gt_z, rep_posterior_final.mean, decoder, joint_names, bounds_dict,
                                   os.path.join(PLOT_DIR, "final_boundary_comparison.png"))
        except Exception as e:
            print(f"Warning: Could not generate boundary comparison: {e}")

    learner.diagnostics.print_final_report()
    print(f"Done. Saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
