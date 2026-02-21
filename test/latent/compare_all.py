"""
Master Comparison: Latent Pipeline vs All Strategies.

Two orthogonal axes for experiment configuration:
  - Strategy (--strategies): test selection method (bald, random, gp, grid, etc.)
    CLI names use hyphens (quasi-random, prior-boundary).
  - Posterior Method (--posterior-method): inference method (vi, ensemble, svgd, full_cov_vi)
    Defaults to the value in latent.yaml (posterior.method); CLI overrides when explicit.

Any strategy can be combined with any posterior method.

Usage:
    # Run all strategies with all metrics (posterior method from config)
    python compare_all.py --strategies all --metrics all --trials 5 --budget 20

    # Run specific strategies
    python compare_all.py --strategies bald random --metrics iou param_error

    # Compare strategies with SVGD posterior
    python compare_all.py --strategies bald random --posterior-method svgd \
        --n-particles 20 --trials 5 --budget 75

    # Compare strategies with Ensemble posterior
    python compare_all.py --strategies bald random --posterior-method ensemble \
        --ensemble-size 5 --trials 5 --budget 40

    # Available strategies: bald, bald-grid, gp, legacy, random,
    #   quasi-random, weighted-bald, batchbald-direct, kl-annealed, tau-annealed,
    #   heuristic, prior-boundary, all
    # Available posterior methods: vi (default from config), ensemble, svgd, full_cov_vi
    # Available metrics: iou, accuracy, f1, param_error, uncertainty,
    #   box_error, presence_error, blob_error, all
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import glob
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_sample_user import LatentUserGenerator
from active_learning.src.latent_variational_inference import LatentVariationalInference
from active_learning.src.latent_bald import LatentBALD
from active_learning.src.metrics import compute_reachability_metrics
from active_learning.src.baselines.grid_strategy import GridStrategy
from active_learning.src.baselines.gp_strategy import GPStrategy
from active_learning.src.baselines.quasi_random_strategy import QuasiRandomStrategy
from active_learning.src.baselines.heuristic_strategy import HeuristicStrategy
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.utils import load_decoder_model
from active_learning.src.factory import build_learner
from active_learning.src.factory import build_learner
from infer_params.training.level_set_torch import create_evaluation_grid, evaluate_level_set_batched


# =============================================================================
# Color scheme: base color per strategy, linestyle per posterior method
# =============================================================================

STRATEGY_COLORS = {
    'Latent BALD': 'blue',
    'Latent G-BALD': 'darkblue',
    'Latent k-BALD': 'royalblue',
    'Latent BALD (Grid)': 'cyan',
    'GP': 'green',
    'Direct Param Inference': 'gray',
    'Latent Random': 'red',
    'Latent Quasi-Random': 'purple',
    'Latent Weighted BALD': 'brown',
    'BatchBALD + Direct Opt': 'gold',
    'KL Annealed BALD': 'pink',
    'Tau Annealed BALD': 'orange',
    'Latent Heuristic (Dense)': 'magenta',
    'Latent Prior Boundary': 'teal',
    'Latent Multi-Stage Warmup': 'darkgreen',
}

POSTERIOR_LINESTYLES = {
    'vi': '-',
    'ensemble': '--',
    'svgd': ':',
    'full_cov_vi': '-.',
}

# Metric labels for plots
METRIC_LABELS = {
    'iou': 'Reachability IoU',
    # 'accuracy': 'Accuracy',
    'boundary_accuracy': 'Boundary Accuracy (Near GT)',
    'f1': 'F1 Score',
    'param_error': 'Parameter Error (L2)',
    'uncertainty': 'Mean Posterior Std',
    'box_error': 'Box Error (L2)',
    'presence_error': 'Presence Probability Error (L2)',
    'blob_error': 'Blob Parameter Error (L2)'
}


# =============================================================================
# Utility Functions
# =============================================================================

def _deep_merge(base, override):
    """Deep merge override dict into base dict (in-place)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def compute_detailed_parameter_error(decoder, posterior_mean, gt_z):
    """Compute separated parameter errors."""
    with torch.no_grad():
        gt_lower, gt_upper, gt_weights, gt_pres_logits, gt_blob_params = decoder.decode_from_embedding(gt_z.unsqueeze(0))
        gt_pres = torch.sigmoid(gt_pres_logits)

        p_lower, p_upper, p_weights, p_pres_logits, p_blob_params = decoder.decode_from_embedding(posterior_mean.unsqueeze(0))
        p_pres = torch.sigmoid(p_pres_logits)

        box_err = (torch.norm(p_lower - gt_lower) +
                  torch.norm(p_upper - gt_upper) +
                  torch.norm(p_weights - gt_weights)).item()
        pres_err = torch.norm(p_pres - gt_pres).item()
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
    """Compute mean posterior std in latent space (for single LatentUserDistribution)."""
    std = torch.exp(posterior.log_std)
    return std.mean().item()


def get_posterior_estimate(learner):
    """Get posterior mean and uncertainty, handling all posterior types.

    Returns:
        (mean, uncertainty): Tensor and float
    """
    if isinstance(learner.posterior, list):
        # Ensemble: use best ELBO member
        best = learner.get_posterior()
        mean = best.mean
        unc = np.mean([torch.exp(p.log_std).mean().item() for p in learner.posterior])
    elif hasattr(learner.posterior, 'get_particles'):
        # SVGD: use particle mean
        particles = learner.posterior.get_particles()
        mean = particles.mean(dim=0)
        unc = particles.std(dim=0).mean().item()
    else:
        # Standard VI
        mean = learner.posterior.mean
        unc = torch.exp(learner.posterior.log_std).mean().item()
    return mean, unc


def compute_all_metrics(decoder, ground_truth_params, test_grid, gt_z, learner):
    """Compute all metrics for the current learner state, handling all posterior types."""
    mean, unc = get_posterior_estimate(learner)

    # Reachability metrics
    if isinstance(learner.posterior, list):
        from active_learning.src.metrics import compute_ensemble_reachability_metrics
        iou, acc, f1, boundary_acc = compute_ensemble_reachability_metrics(
            decoder=decoder, ground_truth_params=ground_truth_params,
            posteriors=learner.posterior, test_grid=test_grid
        )
    else:
        iou, acc, f1, boundary_acc = compute_reachability_metrics(
            decoder=decoder, ground_truth_params=ground_truth_params,
            posterior_mean=mean.unsqueeze(0), test_grid=test_grid
        )

    param_err = compute_parameter_error(decoder, mean, gt_z)
    detailed_err = compute_detailed_parameter_error(decoder, mean, gt_z)

    return {
        'iou': iou, 'accuracy': acc, 'f1': f1, 'boundary_accuracy': boundary_acc,
        'param_error': param_err, 'uncertainty': unc,
        'box_error': detailed_err['box_error'],
        'presence_error': detailed_err['presence_error'],
        'blob_error': detailed_err['blob_error']
    }


def compute_gp_metrics(gp, gt_checker, test_grid_np):
    """Compute IoU and Accuracy for GP model."""
    if len(gp.X_train) == 0:
        return 0.0, 0.5, 0.0, 0.0

    mu = gp.gp.predict(test_grid_np)
    pred_feasible = (mu >= 0.5)

    grid_tensor = torch.tensor(test_grid_np, dtype=torch.float32, device=gt_checker.device)
    batch_z = gt_checker.z.unsqueeze(0)
    logits = gt_checker.batched_logit_values(gt_checker.decoder, batch_z, grid_tensor)
    logits = logits.squeeze(0).cpu().detach().numpy()
    gt_feasible = (logits >= 0)

    intersection = np.logical_and(pred_feasible, gt_feasible).sum()
    union = np.logical_or(pred_feasible, gt_feasible).sum()
    iou = intersection / union if union > 0 else 0.0
    accuracy = np.mean(pred_feasible == gt_feasible)

    tp = np.logical_and(pred_feasible, gt_feasible).sum()
    fp = np.logical_and(pred_feasible, ~gt_feasible).sum()
    fn = np.logical_and(~pred_feasible, gt_feasible).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Compute Boundary-Local Accuracy (closest 20% of points to GT boundary)
    # Note: h_true for GT is logits from checker
    boundary_distances = np.abs(logits)
    threshold = np.quantile(boundary_distances, 0.2)
    boundary_mask = boundary_distances <= threshold
    
    if boundary_mask.sum() > 0:
        boundary_accuracy = (gt_feasible[boundary_mask] == pred_feasible[boundary_mask]).mean()
    else:
        boundary_accuracy = 0.0

    return iou, accuracy, f1, boundary_accuracy


def compute_ceiling_metrics(decoder, ground_truth_params, gt_z, test_grid):
    """Compute ceiling metrics: the best achievable IoU/F1 if posterior perfectly matches gt_z.

    This measures decoder reconstruction fidelity and serves as the upper bound for learning.
    """
    ceiling_iou, _, ceiling_f1, ceiling_boundary_acc = compute_reachability_metrics(
        decoder=decoder, ground_truth_params=ground_truth_params,
        posterior_mean=gt_z.unsqueeze(0), test_grid=test_grid
    )
    return {
        'ceiling_iou': ceiling_iou,
        'ceiling_f1': ceiling_f1,
        'ceiling_boundary_accuracy': ceiling_boundary_acc
    }


def setup_latent_environment(seed, config, decoder, embeddings=None):
    """Setup shared environment for latent pipeline trials.

    If embeddings is provided, samples random user from embeddings with rejection sampling
    to ensure non-trivial feasibility volume (similar to run_latent_diagnosis.py).
    Otherwise, generates synthetic user from LatentUserGenerator.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    prior_gen = LatentPriorGenerator(config, decoder, verbose=False)
    
    # Pre-compute grid for feasibility check (coarse)
    joint_limits = prior_gen.anatomical_limits
    grid_lowers = [joint_limits[j][0] for j in prior_gen.joint_names]
    grid_uppers = [joint_limits[j][1] for j in prior_gen.joint_names]
    
    def to_tensor(x):
        return torch.tensor(x, device=DEVICE, dtype=torch.float32)

    # 1. Select Ground Truth User
    gt_z = None
    gt_idx = -1
    
    if embeddings is not None:
        # Rejection sampling from embeddings
        check_grid = create_evaluation_grid(
            to_tensor(grid_lowers), to_tensor(grid_uppers), 
            resolution=8, # Coarse resolution for speed
            device=DEVICE
        )
        
        max_retries = 50
        for attempt in range(max_retries):
            idx = np.random.randint(0, len(embeddings))
            z = embeddings[idx]
            
            # Check volume
            with torch.no_grad():
                decoded = decoder.decode_from_embedding(z.unsqueeze(0))
                # unpack tuple
                lower, upper, weights, pres, blob = decoded
                
                logits = evaluate_level_set_batched(
                    check_grid, lower, upper, weights, pres, blob
                )
                feasible_frac = (logits > 0).float().mean().item()
                
            if feasible_frac > 0.01: # At least 1% (relaxed from 33% for broader coverage, but prevents 0%)
                # Note: run_latent_diagnosis used 0.33, maybe we stick to 0.1 or so?
                # User asked for "same logic", but 0.33 is quite high. 
                # run_latent_diagnosis had: if feasible_frac > 0.33:
                # I will stick to 0.33 as requested "same logic".
                if feasible_frac > 0.33:
                    gt_idx = idx
                    gt_z = z.clone()
                    # print(f"Selected User {gt_idx} on attempt {attempt+1} (Feasible Volume: {feasible_frac:.1%})")
                    break
            
            # if attempt % 10 == 0:
            #     print(f"Attempt {attempt+1}: User {idx} too restrictive ({feasible_frac:.1%}), retrying...", flush=True)
        
        if gt_z is None:
            print(f"Warning: Could not find feasible user (>33%) after {max_retries} retries. Using last sampled.")
            gt_z = z.clone() # Use last one
            
        # Create checker for this z
        gt_checker = LatentFeasibilityChecker(decoder=decoder, z=gt_z, device=DEVICE)
        
    else:
        # Fallback to synthetic generation (no volume check by default unless we implement it here too)
        # Using vanilla LatentUserGenerator
        user_gen = LatentUserGenerator(config, decoder, decoder.latent_dim)
        gt_z, gt_checker = user_gen.generate_user()

    # 2. Extract ground truth params
    with torch.no_grad():
        gt_lower, gt_upper, gt_weights, gt_pres_logits, gt_blob_params = decoder.decode_from_embedding(
            gt_z.unsqueeze(0)
        )
        ground_truth_params = {
            'box_lower': gt_lower.squeeze(0),
            'box_upper': gt_upper.squeeze(0),
            'box_weights': gt_weights.squeeze(0),
            'presence': torch.sigmoid(gt_pres_logits).squeeze(0),
            'blob_params': gt_blob_params.squeeze(0)
        }

    # 3. Setup evaluation grid (Fine resolution for metrics)
    eval_res = config.get('metrics', {}).get('grid_resolution', 12)
    test_grid = create_evaluation_grid(
        to_tensor(grid_lowers), to_tensor(grid_uppers), eval_res, DEVICE
    )

    # 4. Compute ceiling metrics (upper bound if posterior = gt_z)
    ceiling_metrics = compute_ceiling_metrics(decoder, ground_truth_params, gt_z, test_grid)

    return prior_gen, gt_z, gt_checker, ground_truth_params, test_grid, ceiling_metrics


def _make_display_name(baseline, posterior_method, epsilon=0.0, multi_epsilon=False):
    """Generate display name for baseline + posterior method + epsilon combination."""
    base_names = {
        'bald': 'Latent BALD',
        'gbald': 'Latent G-BALD',
        'kbald': 'Latent k-BALD',
        'bald-grid': 'Latent BALD (Grid)',
        'gp': 'GP',
        'legacy': 'Direct Param Inference',
        'random': 'Latent Random',
        'quasi-random': 'Latent Quasi-Random',
        'weighted-bald': 'Latent Weighted BALD',
        'batchbald-direct': 'BatchBALD + Direct Opt',
        'kl-annealed': 'KL Annealed BALD',
        'tau-annealed': 'Tau Annealed BALD',
        'heuristic': 'Latent Heuristic (Dense)',
        'prior-boundary': 'Latent Prior Boundary',
        'multi-stage-warmup': 'Latent Multi-Stage Warmup',
    }
    name = base_names.get(baseline, baseline)
    if posterior_method != 'vi':
        suffix = 'Full-Cov VI' if posterior_method == 'full_cov_vi' else posterior_method.upper()
        name += f' ({suffix})'
    if multi_epsilon and epsilon > 0:
        name += f' (\u03b5={epsilon})'
    return name


# =============================================================================
# Unified Latent Runner (replaces 7 near-identical functions)
# =============================================================================

def run_trial_latent(seed, budget, config, decoder, strategy, posterior_method='vi',
                     config_overrides=None, embeddings=None):
    """Run a latent pipeline trial with any strategy x posterior method combination.

    This unified runner replaces run_trial_latent_bald, run_trial_latent_random,
    run_trial_weighted_bald, run_trial_kl_annealed, run_trial_tau_annealed,
    run_trial_svgd, and run_trial_ensemble.

    Args:
        seed: Random seed
        budget: Number of queries
        config: Base config dict
        decoder: Decoder model
        strategy: Strategy name (bald, random, quasi_random, etc.)
        posterior_method: Posterior inference method (vi, ensemble, svgd)
        config_overrides: Optional dict to deep-merge into config (e.g. weighted BALD settings)
    """
    trial_config = deepcopy(config)

    # Set strategy axis
    trial_config['acquisition'] = trial_config.get('acquisition', {})
    trial_config['acquisition']['strategy'] = strategy

    # Set posterior method axis
    trial_config['posterior'] = trial_config.get('posterior', {})
    trial_config['posterior']['method'] = posterior_method

    # Default: disable weighted BALD (overrides can re-enable)
    trial_config['bald'] = trial_config.get('bald', {})
    trial_config['bald']['use_weighted_bald'] = False

    # Apply config overrides (e.g. weighted BALD, KL annealing, tau schedule)
    if config_overrides:
        _deep_merge(trial_config, config_overrides)

    # Ensure SVGD/ensemble sub-configs exist
    trial_config['svgd'] = trial_config.get('svgd', {})
    trial_config['ensemble'] = trial_config.get('ensemble', {})

    prior_gen, gt_z, gt_checker, ground_truth_params, test_grid, _ = setup_latent_environment(
        seed, trial_config, decoder, embeddings
    )

    prior = prior_gen.get_prior(gt_z, embeddings=embeddings)
    posterior = LatentUserDistribution(
        latent_dim=decoder.latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )

    n_joints = len(prior_gen.joint_names)
    oracle = LatentOracle(decoder=decoder, ground_truth_z=gt_z, n_joints=n_joints)
    bounds = get_bounds_from_config(trial_config, DEVICE)

    learner = build_learner(
        decoder=decoder, prior=prior, posterior=posterior,
        oracle=oracle, bounds=bounds, config=trial_config
    )

    display_name = strategy + (f' ({posterior_method})' if posterior_method != 'vi' else '')

    history = {k: [] for k in ['iou', 'f1', 'boundary_accuracy', 'param_error', 'uncertainty',
                                'box_error', 'presence_error', 'blob_error']}

    def log():
        metrics = compute_all_metrics(decoder, ground_truth_params, test_grid, gt_z, learner)
        for k, v in metrics.items():
            history[k].append(v)

    history = {k: [] for k in ['iou', 'accuracy', 'f1', 'boundary_accuracy', 'param_error', 'uncertainty',
                                'box_error', 'presence_error', 'blob_error']}

    def log():
        metrics = compute_all_metrics(decoder, ground_truth_params, test_grid, gt_z, learner)
        for k, v in metrics.items():
            if k == 'accuracy': continue # Suppress accuracy
            history[k].append(v)
    for i in range(budget):
        print(f"[{display_name}] Query {i+1}/{budget}", flush=True)
        learner.step(verbose=False)
        log()

    return history


# =============================================================================
# Special-Case Runners (fundamentally different from build_learner flow)
# =============================================================================



def run_trial_latent_bald_grid(seed, budget, config, decoder, grid_resolution=5, embeddings=None):
    """Run Latent BALD with grid search for test selection (standard VI only)."""
    trial_config = deepcopy(config)

    prior_gen, gt_z, gt_checker, ground_truth_params, test_grid, _ = setup_latent_environment(
        seed, trial_config, decoder, embeddings
    )

    prior = prior_gen.get_prior(gt_z, embeddings=embeddings)
    posterior = LatentUserDistribution(
        latent_dim=decoder.latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )

    n_joints = len(prior_gen.joint_names)
    oracle = LatentOracle(decoder=decoder, ground_truth_z=gt_z, n_joints=n_joints)

    vi = LatentVariationalInference(decoder, prior, posterior, trial_config)
    bald = LatentBALD(decoder, posterior, trial_config)
    grid_strat = GridStrategy(prior_gen.anatomical_limits, resolution=grid_resolution)

    history = {k: [] for k in ['iou', 'f1', 'boundary_accuracy', 'param_error', 'uncertainty',
                                'box_error', 'presence_error', 'blob_error']}

    def log():
        iou, acc, f1, boundary_acc = compute_reachability_metrics(
            decoder=decoder, ground_truth_params=ground_truth_params,
            posterior_mean=posterior.mean.unsqueeze(0), test_grid=test_grid
        )
        param_err = compute_parameter_error(decoder, posterior.mean, gt_z)
        detailed_err = compute_detailed_parameter_error(decoder, posterior.mean, gt_z)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        # history['accuracy'].append(acc)
        history['boundary_accuracy'].append(boundary_acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)
        history['box_error'].append(detailed_err['box_error'])
        history['presence_error'].append(detailed_err['presence_error'])
        history['blob_error'].append(detailed_err['blob_error'])

    log()
    for i in range(budget):
        print(f"[Latent BALD (Grid)] Query {i+1}/{budget}", flush=True)
        def acq_fn(points):
            trial_config['bald'] = trial_config.get('bald', {})
            trial_config['bald']['use_weighted_bald'] = False
            return bald.compute_score(points)

        test_point = grid_strat.select_test_point(acq_fn)
        oracle.query(test_point)
        vi.update_posterior(oracle.get_history())
        log()

    return history


def run_trial_gp(seed, budget, config, decoder, embeddings=None):
    """Run GP baseline trial (no latent structure)."""
    trial_config = deepcopy(config)

    prior_gen, gt_z, gt_checker, ground_truth_params, test_grid, _ = setup_latent_environment(
        seed, trial_config, decoder, embeddings
    )
    test_grid_np = test_grid.cpu().numpy()

    n_joints = len(prior_gen.joint_names)
    oracle = LatentOracle(decoder=decoder, ground_truth_z=gt_z, n_joints=n_joints)
    gp = GPStrategy(prior_gen.anatomical_limits)

    history = {'iou': [], 'f1': [], 'boundary_accuracy': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1, boundary_acc = compute_gp_metrics(gp, gt_checker, test_grid_np)
        history['iou'].append(iou)
        # history['accuracy'].append(acc)
        history['boundary_accuracy'].append(boundary_acc)
        history['f1'].append(f1)
        history['param_error'].append(np.nan)
        history['uncertainty'].append(np.nan)

    log()
    for i in range(budget):
        print(f"[GP] Query {i+1}/{budget}", flush=True)
        test_point = gp.select_test_point()
        outcome = oracle.query(test_point)
        gp.update(test_point, outcome > 0)
        log()

    return history


def run_trial_legacy(seed, budget, legacy_config, decoder, gt_z, gt_checker, test_grid):
    """Run Legacy (Direct Parameter Inference) on Latent oracle."""
    from active_learning.src.legacy.active_learning_pipeline import ActiveLearner as LegacyLearner
    from active_learning.src.legacy.oracle import Oracle as LegacyOracle
    from active_learning.src.legacy.prior_generation import PriorGenerator as LegacyPriorGen
    from active_learning.src.metrics import compute_legacy_reachability_metrics

    trial_config = deepcopy(legacy_config)
    trial_config['acquisition'] = trial_config.get('acquisition', {})
    trial_config['acquisition']['strategy'] = 'bald'

    legacy_prior_gen = LegacyPriorGen(trial_config)
    legacy_prior = legacy_prior_gen.get_prior()
    legacy_posterior = legacy_prior_gen.get_prior()
    legacy_oracle = LegacyOracle(gt_checker, legacy_prior.joint_names)
    legacy_learner = LegacyLearner(legacy_prior, legacy_posterior, legacy_oracle, trial_config)

    test_grid_np = test_grid.cpu().numpy()

    history = {'iou': [], 'f1': [], 'boundary_accuracy': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1, boundary_acc = compute_legacy_reachability_metrics(gt_checker, legacy_posterior, test_grid_np)
        history['iou'].append(iou)
        # history['accuracy'].append(acc)
        history['boundary_accuracy'].append(boundary_acc)
        history['f1'].append(f1)
        history['param_error'].append(np.nan)
        history['uncertainty'].append(np.nan)

    log()
    for i in range(budget):
        print(f"[Legacy/Direct] Query {i+1}/{budget}", flush=True)
        legacy_learner.step(verbose=False)
        log()

    return history


def run_trial_heuristic(seed, budget, config, decoder, embeddings_from_main, bank_size=2000):
    """Run Latent Heuristic Baseline (Version Space / Dense Banking)."""
    trial_config = deepcopy(config)
    prior_gen, gt_z, gt_checker, ground_truth_params, test_grid, _ = setup_latent_environment(
        seed, trial_config, decoder, embeddings_from_main
    )

    prior = prior_gen.get_prior(gt_z, embeddings=embeddings_from_main)
    posterior = LatentUserDistribution(
        latent_dim=decoder.latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )

    n_joints = len(prior_gen.joint_names)
    oracle = LatentOracle(decoder=decoder, ground_truth_z=gt_z, n_joints=n_joints)
    bounds = get_bounds_from_config(trial_config, DEVICE)

    embeddings = embeddings_from_main.detach()

    from active_learning.src.baselines.version_space_strategy import VersionSpaceStrategy

    res = int(bank_size ** (1.0 / len(prior_gen.joint_names)))
    res = max(2, res)

    grid_lowers = [prior_gen.anatomical_limits[j][0] for j in prior_gen.joint_names]
    grid_uppers = [prior_gen.anatomical_limits[j][1] for j in prior_gen.joint_names]

    from active_learning.test.diagnostics.utils import create_evaluation_grid as create_eval_grid

    grid_queries = create_eval_grid(
        torch.tensor(grid_lowers, device=DEVICE, dtype=torch.float32),
        torch.tensor(grid_uppers, device=DEVICE, dtype=torch.float32),
        res,
        DEVICE
    )

    strat = VersionSpaceStrategy(
        embeddings=embeddings,
        queries=grid_queries,
        decoder=decoder,
        device=DEVICE
    )

    history = {k: [] for k in ['iou', 'f1', 'boundary_accuracy', 'param_error', 'uncertainty',
                                'box_error', 'presence_error', 'blob_error']}

    def log():
        z_est = strat.get_posterior_mean()

        iou, acc, f1, boundary_acc = compute_reachability_metrics(
            decoder=decoder, ground_truth_params=ground_truth_params,
            posterior_mean=z_est.unsqueeze(0), test_grid=test_grid
        )
        param_err = compute_parameter_error(decoder, z_est, gt_z)
        detailed_err = compute_detailed_parameter_error(decoder, z_est, gt_z)

        unc_vec = strat.get_posterior_std()
        unc = unc_vec.mean().item()

        history['iou'].append(iou)
        # history['accuracy'].append(acc)
        history['boundary_accuracy'].append(boundary_acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)
        history['box_error'].append(detailed_err['box_error'])
        history['presence_error'].append(detailed_err['presence_error'])
        history['blob_error'].append(detailed_err['blob_error'])

    log()
    for i in range(budget):
        print(f"[Heuristic] Query {i+1}/{budget}", flush=True)
        test_point = strat.select_test_point()
        outcome = oracle.query(test_point)
        is_feasible = (outcome > 0)
        strat.update(test_point, is_feasible)
        log()

    return history


# =============================================================================
# BatchBALD + Direct Optimization Helpers
# =============================================================================

from active_learning.src.utils import binary_entropy

def compute_joint_mi(probs_selected, probs_candidate):
    """Compute joint mutual information (Greedy approx) for BatchBALD."""
    if len(probs_selected) == 0:
        mean_prob = probs_candidate.mean()
        return (binary_entropy(mean_prob) - binary_entropy(probs_candidate).mean()).item()

    mean_prob = probs_candidate.mean()
    bald_score = binary_entropy(mean_prob) - binary_entropy(probs_candidate).mean()

    redundancy = 0.0
    for prev_probs in probs_selected:
        corr = torch.corrcoef(torch.stack([probs_candidate, prev_probs]))[0, 1]
        if not torch.isnan(corr):
            redundancy += abs(corr.item())

    return (bald_score / (1.0 + redundancy)).item()

def select_batch_bald(decoder, prior, bounds, n_points, n_candidates=1000, seed=42):
    """Select K points utilizing information from the prior via BatchBALD."""
    torch.manual_seed(seed)

    candidates = bounds[:, 0] + torch.rand(n_candidates, len(bounds), device=DEVICE) * (bounds[:, 1] - bounds[:, 0])

    z_samples = prior.sample(32)
    if isinstance(z_samples, list):
         z_samples = torch.stack(z_samples)

    logits = LatentFeasibilityChecker.batched_logit_values(decoder, z_samples, candidates)
    probs = torch.sigmoid(logits)

    selected_indices = []
    selected_probs_list = []

    for _ in range(n_points):
        best_score = -float('inf')
        best_idx = None

        for c in range(n_candidates):
            if c in selected_indices: continue

            cand_probs = probs[:, c]
            score = compute_joint_mi(selected_probs_list, cand_probs)

            if score > best_score:
                best_score = score
                best_idx = c

        selected_indices.append(best_idx)
        selected_probs_list.append(probs[:, best_idx])

    return candidates[selected_indices]


def run_trial_batchbald_direct(seed, budget, config, decoder, embeddings=None):
    """Run BatchBALD + Direct Optimization baseline."""
    trial_config = deepcopy(config)
    prior_gen, gt_z, gt_checker, ground_truth_params, test_grid, _ = setup_latent_environment(
        seed, trial_config, decoder, embeddings
    )
    prior = prior_gen.get_prior(gt_z, embeddings=embeddings)

    bounds = get_bounds_from_config(trial_config, DEVICE)
    batch_points = select_batch_bald(decoder, prior, bounds, n_points=budget, seed=seed)

    oracle = LatentOracle(decoder=decoder, ground_truth_z=gt_z, n_joints=len(bounds))
    outcomes = []
    for point in batch_points:
        outcomes.append(oracle.query(point))
    outcomes = torch.tensor(outcomes, device=DEVICE, dtype=torch.float32)

    history = {k: [] for k in ['iou', 'f1', 'boundary_accuracy', 'param_error', 'uncertainty',
                                'box_error', 'presence_error', 'blob_error']}

    def evaluate(z_est, uncertainty=0.0):
        iou, acc, f1, boundary_acc = compute_reachability_metrics(
            decoder=decoder, ground_truth_params=ground_truth_params,
            posterior_mean=z_est.unsqueeze(0), test_grid=test_grid
        )
        param_err = compute_parameter_error(decoder, z_est, gt_z)
        detailed_err = compute_detailed_parameter_error(decoder, z_est, gt_z)

        history['iou'].append(iou)
        # history['accuracy'].append(acc)
        history['boundary_accuracy'].append(boundary_acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['box_error'].append(detailed_err['box_error'])
        history['presence_error'].append(detailed_err['presence_error'])
        history['blob_error'].append(detailed_err['blob_error'])
        history['uncertainty'].append(uncertainty)

    evaluate(prior.mean, uncertainty=prior.log_std.exp().mean().item())

    tau = trial_config.get('bald', {}).get('tau', 0.1)

    for k in range(1, budget + 1):
        print(f"[BatchBALD Direct] Query {k}/{budget}", flush=True)
        k_points = batch_points[:k]
        k_outcomes = outcomes[:k]

        z = prior.mean.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=0.05)

        for _ in range(100):
            optimizer.zero_grad()
            logits = LatentFeasibilityChecker.batched_logit_values(decoder, z.unsqueeze(0), k_points).squeeze(0)
            probs = torch.sigmoid(logits / tau)
            probs = torch.clamp(probs, 1e-6, 1-1e-6)
            loss = -(k_outcomes * torch.log(probs) + (1-k_outcomes) * torch.log(1-probs)).mean()
            loss.backward()
            optimizer.step()

        evaluate(z.detach(), uncertainty=0.0)

    return history


# =============================================================================
# Plotting
# =============================================================================

def plot_metric(results_dict, metric, save_dir, n_trials, budget, posterior_method='vi', ceiling_metrics=None):
    """Plot a single metric across all baselines.

    Args:
        ceiling_metrics: Optional dict with 'ceiling_iou', 'ceiling_f1', 'ceiling_boundary_accuracy'
                        averaged across trials. Used to draw horizontal ceiling lines.
    """
    max_len = 0
    for trials in results_dict.values():
        if trials:
            valid_lens = [len(trial[metric]) for trial in trials if not all(np.isnan(v) for v in trial[metric])]
            if valid_lens:
                max_len = max(max_len, max(valid_lens))

    if max_len == 0:
        print(f"Warning: No data to plot for {metric}")
        return

    plt.figure(figsize=(10, 6))

    actual_n_trials = 0
    for trials in results_dict.values():
        if trials:
            actual_n_trials = max(actual_n_trials, len(trials))

    if actual_n_trials == 0:
        actual_n_trials = n_trials

    # Color cycle for dynamic display names
    color_cycle = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'brown',
                   'magenta', 'gold', 'pink', 'gray', 'olive', 'navy', 'teal']

    for idx, (name, trials) in enumerate(results_dict.items()):
        if not trials:
            continue

        data = []
        for trial in trials:
            vals = trial[metric]
            if all(np.isnan(v) for v in vals):
                continue
            if len(vals) < max_len:
                vals = vals + [vals[-1]] * (max_len - len(vals))
            data.append(vals[:max_len])

        if not data:
            continue

        arr = np.array(data)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        x = range(len(mean))

        # Look up color by baseline key, fall back to color cycle
        color = STRATEGY_COLORS.get(name, color_cycle[idx % len(color_cycle)])
        linestyle = POSTERIOR_LINESTYLES.get(posterior_method, '-')

        plt.plot(x, mean, label=f"{name} (Final: {mean[-1]:.3f})",
                color=color, linewidth=2, linestyle=linestyle)
        plt.fill_between(x, mean - std, mean + std,
                        color=color, alpha=0.15)

    # Draw ceiling line if applicable
    if ceiling_metrics is not None:
        ceiling_key = f'ceiling_{metric}'
        if ceiling_key in ceiling_metrics:
            ceiling_val = ceiling_metrics[ceiling_key]
            plt.axhline(y=ceiling_val, color='black', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Ceiling: {ceiling_val:.3f}')

    posterior_suffix = f' [{posterior_method.upper()}]' if posterior_method != 'vi' else ''
    plt.title(f'{METRIC_LABELS.get(metric, metric)} - Latent Pipeline ({actual_n_trials} trials){posterior_suffix}')
    plt.xlabel('Queries')
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.legend()
    plt.grid(True, alpha=0.3)

    if metric in ['iou', 'accuracy']:
        plt.ylim(0, 1.05)
    else:
        plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{metric}.png'), dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master Comparison: Latent Pipeline vs All Strategies (Two-Axis Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare BALD vs Random (posterior method from config)
  python compare_all.py --strategies bald random --trials 5 --budget 40

  # Compare BALD vs Random with SVGD posterior
  python compare_all.py --strategies bald random --posterior-method svgd \\
      --n-particles 20 --trials 5 --budget 75

  # Compare BALD vs Random with Ensemble posterior
  python compare_all.py --strategies bald random --posterior-method ensemble \\
      --ensemble-size 5 --trials 5 --budget 40

  # Run all strategies with all metrics
  python compare_all.py --strategies all --metrics all --trials 5 --budget 20
        """)
    parser.add_argument("--strategies", nargs='+',
                       choices=['bald', 'gbald', 'kbald', 'bald-grid', 'gp', 'legacy', 'random',
                                'quasi-random', 'weighted-bald', 'batchbald-direct',
                                'kl-annealed', 'tau-annealed', 'heuristic',
                                'prior-boundary', 'multi-stage-warmup', 'all'],
                       default=['all'],
                       help="Strategies to compare")
    parser.add_argument("--posterior-method", type=str, default=None,
                       choices=['vi', 'ensemble', 'svgd', 'full_cov_vi', 'sliced_svgd', 'projected_svgd', 'projected_svn'],
                       help="Posterior inference method (default: from config)")
    parser.add_argument("--metrics", nargs='+',
                       choices=['iou', 'accuracy', 'f1', 'boundary_accuracy', 'param_error', 'uncertainty',
                                'box_error', 'presence_error', 'blob_error', 'all'],
                       default=['all'],
                       help="Metrics to plot")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--budget", type=int, default=20, help="Query budget")
    parser.add_argument("--model-path", type=str, default='models/best_model.pt',
                       help="Path to trained decoder checkpoint")
    parser.add_argument("--ensemble-size", type=int, default=None,
                       help="Number of ensemble members (for --posterior-method ensemble)")
    parser.add_argument("--n-particles", type=int, default=None,
                       help="Number of SVGD particles (for --posterior-method svgd)")
    parser.add_argument("--heuristic-bank-size", type=int, default=2000,
                       help="Size of query bank for heuristic baseline")
    parser.add_argument("--seed", type=int, default=None,
                       help="Base random seed")
    parser.add_argument("--trial-idx", type=int, default=None,
                       help="Index of the trial to run (0 to trials-1). If set, runs only this trial.")
    parser.add_argument("--epsilon", nargs='*', type=float, default=None,
                       help="Epsilon-greedy exploration rates to compare (e.g., --epsilon 0.0 0.1)")
    parser.add_argument("--epsilon-decay", type=float, default=0.95,
                       help="Epsilon decay factor per iteration")
    parser.add_argument("--results-dir", type=str, default="active_learning/results/latent_master_comparison",
                       help="Directory to save/load individual trial results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save output images (uses default if not specified)")
    parser.add_argument("--merge", action='store_true',
                       help="Merge existing results from results-dir and plot")
    args = parser.parse_args()

    # Expand 'all'
    if 'all' in args.strategies:
        baselines_to_run = ['bald', 'gbald', 'kbald', 'bald-grid', 'gp', 'legacy', 'random',
                            'quasi-random', 'weighted-bald', 'batchbald-direct',
                            'kl-annealed', 'tau-annealed', 'heuristic',
                            'prior-boundary', 'multi-stage-warmup']
    else:
        baselines_to_run = args.strategies

    if 'all' in args.metrics:
        metrics_to_plot = ['iou', 'f1', 'boundary_accuracy', 'param_error', 'uncertainty',
                           'box_error', 'presence_error', 'blob_error']
    else:
        metrics_to_plot = args.metrics

    # Resolve epsilon values
    epsilon_values = args.epsilon if args.epsilon is not None else [0.0]
    multi_epsilon = len(epsilon_values) > 1

    # Load configs
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    latent_config = load_config(os.path.join(root_dir, 'configs/latent.yaml'))
    legacy_config = load_config(os.path.join(root_dir, 'configs/legacy.yaml'))

    latent_config['stopping']['budget'] = args.budget
    legacy_config['stopping']['budget'] = args.budget

    # Inject posterior method config — only if CLI overrides
    if args.posterior_method is not None:
        latent_config['posterior'] = latent_config.get('posterior', {})
        latent_config['posterior']['method'] = args.posterior_method.replace('-', '_')
    posterior_method = latent_config.get('posterior', {}).get('method', 'vi')
    if args.ensemble_size is not None:
        latent_config.setdefault('ensemble', {})['ensemble_size'] = args.ensemble_size
        latent_config['ensemble'].setdefault('init_noise_std', 0.4)
    if args.n_particles is not None:
        latent_config.setdefault('posterior', {})['n_particles'] = args.n_particles

    # Load decoder
    decoder, embeddings, train_config = load_decoder_model(args.model_path, DEVICE)

    # Setup output
    save_dir = args.output_dir if args.output_dir else "active_learning/images/latent_master_comparison"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Non-factory baselines that don't support epsilon > 0
    NON_FACTORY_BASELINES = {'bald-grid', 'gp', 'legacy', 'heuristic', 'batchbald-direct'}

    # Build display names for every (baseline, epsilon) pair
    def _display_name_for(baseline, eps):
        return _make_display_name(baseline, posterior_method, epsilon=eps, multi_epsilon=multi_epsilon)

    # =================================================================
    # Baseline -> (strategy_name, config_overrides) for unified runner
    # =================================================================
    # Baselines routed through run_trial_latent (factory-based)
    FACTORY_BASELINES = {
        'bald': ('bald', None),
        'gbald': ('gbald', None),
        'kbald': ('kbald', None),
        'random': ('random', None),
        'quasi-random': ('quasi_random', None),
        'weighted-bald': ('bald', {
            'bald': {'use_weighted_bald': True, 'weighted_bald_sigma': 0.1}
        }),
        'kl-annealed': ('bald', {
            'vi': {'kl_annealing': {
                'enabled': True,
                'start_weight': 0.001,
                'end_weight': 1.0,
                'duration': min(20, args.budget),
                'schedule': 'linear'
            }}
        }),
        'tau-annealed': ('bald', {
            'bald': {'tau_schedule': {
                'start': 1.0,
                'end': 0.05,
                'duration': min(20, args.budget),
                'schedule': 'linear'
            }}
        }),
        'prior-boundary': ('prior_boundary', None),
        'multi-stage-warmup': ('multi_stage_warmup', None),
    }

    # Mode 1: Merge Results
    if args.merge:
        print(f"Merging results from {args.results_dir}...")
        # Build results dict keyed by (baseline, epsilon) display names
        results = {}
        ceiling_metrics_list = []  # Empty for merge mode (ceiling not saved in pickles)
        for b in baselines_to_run:
            for eps in epsilon_values:
                if eps > 0 and b in NON_FACTORY_BASELINES:
                    continue
                name = _display_name_for(b, eps)
                results[name] = []

        files = glob.glob(os.path.join(args.results_dir, "*.pkl"))
        print(f"Found {len(files)} result files.")

        for fpath in tqdm(files, desc="Loading results"):
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                    b_name = data['baseline']
                    pkl_eps = data.get('epsilon', 0.0)
                    if b_name in baselines_to_run:
                        disp_name = _display_name_for(b_name, pkl_eps)
                        if disp_name in results:
                            results[disp_name].append(data['result'])
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

    # Mode 2: Run Trials (Sequential or Single)
    else:
        # Generate seeds
        if args.seed is not None:
            np.random.seed(args.seed)
            seeds = [args.seed + i for i in range(args.trials)]
        else:
            seeds = [np.random.randint(0, 100000) for _ in range(args.trials)]

        if args.trial_idx is not None:
            if args.trial_idx < 0 or args.trial_idx >= len(seeds):
                print(f"Error: trial-idx {args.trial_idx} is out of range for {args.trials} trials.")
                return
            seeds_to_run = [seeds[args.trial_idx]]
            indices = [args.trial_idx]
            print(f"Running SINGLE trial {args.trial_idx} (Seed: {seeds_to_run[0]})")
        else:
            seeds_to_run = seeds
            indices = range(len(seeds))
            print(f"Running ALL {len(seeds)} trials sequentially.")

        print(f"Running baselines: {baselines_to_run}")
        print(f"Posterior method: {posterior_method}")
        print(f"Epsilon values: {epsilon_values}")
        print(f"Metrics to plot: {metrics_to_plot}")
        print(f"Budget: {args.budget}")

        # Build results dict keyed by (baseline, epsilon) display names
        results = {}
        ceiling_metrics_list = []  # Store ceiling metrics from each trial
        for b in baselines_to_run:
            for eps in epsilon_values:
                if eps > 0 and b in NON_FACTORY_BASELINES:
                    continue
                name = _display_name_for(b, eps)
                results[name] = []

        for i, seed in zip(indices, seeds_to_run):
            # Setup shared environment for this seed
            # Use same embeddings-based sampling as runners for consistency in 'legacy' runner
            prior_gen, gt_z, gt_checker, ground_truth_params, test_grid, ceiling_metrics = setup_latent_environment(
                seed, latent_config, decoder, embeddings
            )

            print(f"\n--- Trial {i} (Seed {seed}) ---")
            print(f"    Ceiling IoU: {ceiling_metrics['ceiling_iou']:.4f}, "
                  f"Ceiling F1: {ceiling_metrics['ceiling_f1']:.4f}, "
                  f"Ceiling Boundary Acc: {ceiling_metrics['ceiling_boundary_accuracy']:.4f}")
            ceiling_metrics_list.append(ceiling_metrics)

            for baseline in baselines_to_run:
                for eps in epsilon_values:
                    # Skip non-factory baselines when epsilon > 0
                    if eps > 0 and baseline in NON_FACTORY_BASELINES:
                        continue

                    display_name = _display_name_for(baseline, eps)
                    try:
                        grid_res = latent_config.get('acquisition', {}).get('grid_strategy_resolution', 5)

                        # Build epsilon override for factory baselines
                        eps_override = {'acquisition': {'epsilon': eps, 'epsilon_decay': args.epsilon_decay}}

                        # ---- Factory-based baselines (unified runner) ----
                        if baseline in FACTORY_BASELINES:
                            strategy, overrides = FACTORY_BASELINES[baseline]
                            combined_overrides = deepcopy(overrides) if overrides else {}
                            _deep_merge(combined_overrides, eps_override)
                            result = run_trial_latent(
                                seed, args.budget, latent_config, decoder,
                                strategy=strategy, posterior_method=posterior_method,
                                config_overrides=combined_overrides,
                                embeddings=embeddings
                            )

                        # ---- Special baselines: route through factory for non-vi ----
                        elif baseline == 'bald-grid' and posterior_method != 'vi':
                            # Use factory with grid strategy for non-vi posteriors
                            result = run_trial_latent(
                                seed, args.budget, latent_config, decoder,
                                strategy='grid', posterior_method=posterior_method,
                                embeddings=embeddings
                            )

                        # ---- Special baselines: vi-only special runners ----
                        elif baseline == 'bald-grid':
                            result = run_trial_latent_bald_grid(seed, args.budget, latent_config, decoder, grid_res, embeddings)

                        # ---- Baselines that ignore posterior method ----
                        elif baseline == 'gp':
                            result = run_trial_gp(seed, args.budget, latent_config, decoder, embeddings)
                        elif baseline == 'legacy':
                            result = run_trial_legacy(seed, args.budget, legacy_config, decoder,
                                                      gt_z, gt_checker, test_grid)
                        elif baseline == 'heuristic':
                            result = run_trial_heuristic(seed, args.budget, latent_config, decoder,
                                                         embeddings, args.heuristic_bank_size)
                        elif baseline == 'batchbald-direct':
                            result = run_trial_batchbald_direct(seed, args.budget, latent_config, decoder, embeddings)
                        else:
                            print(f"Unknown baseline: {baseline}")
                            continue

                        # If running singly, save immediately
                        if args.trial_idx is not None:
                            eps_tag = f"_eps{eps}" if multi_epsilon else ""
                            out_path = os.path.join(args.results_dir,
                                                    f"trial_{i}_{baseline}_{posterior_method}{eps_tag}.pkl")
                            with open(out_path, 'wb') as f:
                                pickle.dump({
                                    'baseline': baseline,
                                    'posterior_method': posterior_method,
                                    'epsilon': eps,
                                    'seed': seed,
                                    'result': result
                                }, f)
                            print(f"Saved {display_name} result to {out_path}")

                        results[display_name].append(result)

                    except Exception as e:
                        print(f"\n{display_name} failed for seed {seed}: {e}")
                        import traceback
                        traceback.print_exc()

            if args.trial_idx is not None:
                print("Single trial complete. Exiting.")
                return

    # Plot each metric
    if args.trial_idx is None or args.merge:
        # Compute average ceiling metrics across trials
        avg_ceiling = None
        if not args.merge and ceiling_metrics_list:
            avg_ceiling = {
                'ceiling_iou': np.mean([c['ceiling_iou'] for c in ceiling_metrics_list]),
                'ceiling_f1': np.mean([c['ceiling_f1'] for c in ceiling_metrics_list]),
                'ceiling_boundary_accuracy': np.mean([c['ceiling_boundary_accuracy'] for c in ceiling_metrics_list]),
            }
            print(f"\nAverage Ceiling Metrics across {len(ceiling_metrics_list)} trials:")
            print(f"  IoU: {avg_ceiling['ceiling_iou']:.4f}")
            print(f"  F1: {avg_ceiling['ceiling_f1']:.4f}")
            print(f"  Boundary Accuracy: {avg_ceiling['ceiling_boundary_accuracy']:.4f}")

        for metric in metrics_to_plot:
            plot_metric(results, metric, save_dir, args.trials, args.budget,
                       posterior_method=posterior_method, ceiling_metrics=avg_ceiling)
            print(f"Saved {metric}.png")

        # Print summary
        print(f"\nResults saved to {save_dir}")
        print(f"Posterior method: {posterior_method}")
        print("\nFinal metrics (mean +/- std):")
        for name, trials in results.items():
            if trials:
                print(f"\n  {name}:")
                for metric in metrics_to_plot:
                    final_vals = [t[metric][-1] for t in trials if not np.isnan(t[metric][-1])]
                    if final_vals:
                        print(f"    {metric}: {np.mean(final_vals):.4f} +/- {np.std(final_vals):.4f}")
                    else:
                        print(f"    {metric}: N/A")


if __name__ == "__main__":
    main()
