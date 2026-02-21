"""
Master Comparison: Legacy Pipeline vs All Baselines.

Compare BALD (gradient), BALD (grid), GP, and Random strategies.
Supports CLI-based baseline and metric selection.

Usage:
    # Run all baselines with all metrics
    python compare_all.py --baselines all --metrics all --trials 5 --budget 20

    # Run specific baselines
    python compare_all.py --baselines bald random --metrics iou accuracy

    # Available baselines: bald, bald-grid, gp, random, all
    # Available metrics: iou, accuracy, param_error, uncertainty, all
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

from active_learning.src.config import load_config, DEVICE
from active_learning.src.legacy.prior_generation import PriorGenerator
from active_learning.src.legacy.sample_user import UserGenerator
from active_learning.src.legacy.oracle import Oracle
from active_learning.src.legacy.active_learning_pipeline import ActiveLearner
from active_learning.src.legacy.variational_inference import VariationalInference
from active_learning.src.legacy.bald import BALD
from active_learning.src.metrics import compute_legacy_reachability_metrics
from active_learning.src.baselines.grid_strategy import GridStrategy
from active_learning.src.baselines.gp_strategy import GPStrategy
from active_learning.src.baselines.quasi_random_strategy import QuasiRandomStrategy
from active_learning.src.baselines.heuristic_strategy import HeuristicStrategy
from active_learning.src.legacy.feasibility_checker import FeasibilityChecker
from infer_params.training.level_set_torch import create_evaluation_grid


# Color scheme for baselines
COLORS = {
    'BALD': 'blue',
    'BALD (Grid)': 'cyan',
    'GP': 'green',
    'Random': 'red',
    'Quasi-Random': 'purple',
    'Weighted BALD': 'brown',
    'BatchBALD + Direct Opt': 'gold',
    'KL Annealed BALD': 'pink',
    'Tau Annealed BALD': 'orange',
    'Heuristic (Dense)': 'magenta'
}

# Metric labels for plots
METRIC_LABELS = {
    'iou': 'Reachability IoU',
    'accuracy': 'Accuracy',
    'f1': 'F1 Score',
    'param_error': 'Parameter Error (RMSE)',
    'uncertainty': 'Mean Posterior Std'
}


def compute_parameter_error(posterior, true_limits):
    """Compute RMSE between inferred and true joint limits."""
    squared_error = 0.0
    n_params = 0

    for joint in posterior.joint_names:
        p = posterior.params['joint_limits'][joint]
        pred_lower = p['lower_mean'].item()
        pred_upper = p['upper_mean'].item()
        true_lower, true_upper = true_limits[joint]

        squared_error += (pred_lower - true_lower) ** 2
        squared_error += (pred_upper - true_upper) ** 2
        n_params += 2

    return np.sqrt(squared_error / n_params)


def compute_uncertainty(posterior):
    """Compute mean posterior std across all parameters."""
    total_std = 0.0
    n_params = 0

    for joint in posterior.joint_names:
        p = posterior.params['joint_limits'][joint]
        total_std += np.exp(p['lower_log_std'].item())
        total_std += np.exp(p['upper_log_std'].item())
        n_params += 2

    return total_std / n_params


def setup_environment(seed, config):
    """Setup shared environment for a trial."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    prior_gen = PriorGenerator(config)
    prior = prior_gen.get_prior()

    user_gen = UserGenerator(config, prior_gen.joint_names,
                             prior_gen.anatomical_limits, prior_gen.pairs)
    true_limits, true_bumps, true_checker = user_gen.generate_user()

    # Setup evaluation grid
    eval_res = config.get('metrics', {}).get('grid_resolution', 12)
    def to_tensor(x): return torch.tensor(x, device=DEVICE, dtype=torch.float32)
    grid_lowers = [prior_gen.anatomical_limits[j][0] for j in prior_gen.joint_names]
    grid_uppers = [prior_gen.anatomical_limits[j][1] for j in prior_gen.joint_names]
    test_grid_tensor = create_evaluation_grid(to_tensor(grid_lowers), to_tensor(grid_uppers), eval_res, DEVICE)
    test_grid_np = test_grid_tensor.cpu().numpy()

    return prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np


def run_trial_bald(seed, budget, config):
    """Run BALD (gradient optimization) trial."""
    trial_config = deepcopy(config)
    trial_config['acquisition'] = trial_config.get('acquisition', {})
    trial_config['acquisition']['strategy'] = 'bald'

    # Explicitly disable weighted BALD for the regular BALD baseline
    trial_config['bald'] = trial_config.get('bald', {})
    trial_config['bald']['use_weighted_bald'] = False

    prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np = setup_environment(seed, trial_config)
    
    # Pass ground truth to prior generator to ensure noisy perturbation (consistent with latent pipeline)
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    
    oracle = Oracle(true_checker, prior_gen.joint_names)
    learner = ActiveLearner(prior, posterior, oracle, trial_config)

    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)

    log()
    for i in range(budget):
        if i % 1 == 0:
            print(f"[BALD] Query {i+1}/{budget}", flush=True)
        learner.step(verbose=False)
        log()

    return history


def run_trial_bald_grid(seed, budget, config, grid_resolution=5):
    """Run BALD with grid search for test selection."""
    trial_config = deepcopy(config)
    
    # Explicitly disable weighted BALD
    trial_config['bald'] = trial_config.get('bald', {})
    trial_config['bald']['use_weighted_bald'] = False

    prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np = setup_environment(seed, trial_config)
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    oracle = Oracle(true_checker, prior_gen.joint_names)

    vi = VariationalInference(prior, posterior, trial_config)
    bald = BALD(posterior, trial_config)
    grid_strat = GridStrategy(prior_gen.anatomical_limits, resolution=grid_resolution)

    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)

    log()
    log()
    for i in range(budget):
        if i % 1 == 0:
            print(f"[BALD (Grid)] Query {i+1}/{budget}", flush=True)
        def acq_fn(points):
            return bald.compute_score(points)

        test_point = grid_strat.select_test_point(acq_fn)
        oracle.query(test_point)
        vi.update_posterior(oracle.get_history())
        log()

    return history


def run_trial_gp(seed, budget, config):
    """Run GP baseline trial."""
    trial_config = deepcopy(config)
    
    prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np = setup_environment(seed, trial_config)
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    oracle = Oracle(true_checker, prior_gen.joint_names)
    vi = VariationalInference(prior, posterior, trial_config)

    gp_strat = GPStrategy(prior_gen.anatomical_limits)

    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)

    log()
    log()
    for i in range(budget):
        if i % 1 == 0:
            print(f"[GP] Query {i+1}/{budget}", flush=True)
        test_point = gp_strat.select_test_point()
        outcome = oracle.query(test_point)
        gp_strat.update(test_point, outcome)
        vi.update_posterior(oracle.get_history())
        log()

    return history


def run_trial_random(seed, budget, config):
    """Run Random baseline trial."""
    trial_config = deepcopy(config)
    trial_config['acquisition'] = trial_config.get('acquisition', {})
    trial_config['acquisition']['strategy'] = 'random'

    prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np = setup_environment(seed, trial_config)
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    oracle = Oracle(true_checker, prior_gen.joint_names)
    learner = ActiveLearner(prior, posterior, oracle, trial_config)

    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)

    log()
    for i in range(budget):
        if i % 1 == 0:
            print(f"[Random] Query {i+1}/{budget}", flush=True)
        learner.step(verbose=False)
        log()

    return history


def run_trial_quasi_random(seed, budget, config):
    """Run Quasi-Random baseline trial."""
    trial_config = deepcopy(config)
    
    prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np = setup_environment(seed, trial_config)
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    oracle = Oracle(true_checker, prior_gen.joint_names)
    
    # Instantiate QuasiRandomStrategy in standalone mode
    qr_strat = QuasiRandomStrategy(bald_strategy=None, n_quasi_random=budget, device=DEVICE, seed=seed)
    
    # Need VI to update posterior for metrics
    vi = VariationalInference(prior, posterior, trial_config)

    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)

    log()
    bounds = torch.tensor([prior_gen.anatomical_limits[j] for j in prior_gen.joint_names], device=DEVICE)
    
    for i in range(budget):
        if i % 1 == 0:
            print(f"[Quasi-Random] Query {i+1}/{budget}", flush=True)
        test_point, _ = qr_strat.select_test(bounds)
        outcome = oracle.query(test_point)
        # Update posterior even though selection was random/quasi-random
        vi.update_posterior(oracle.get_history())
        log()

    return history


def run_trial_weighted_bald(seed, budget, config):
    """Run Weighted BALD trial."""
    trial_config = deepcopy(config)
    trial_config['acquisition'] = trial_config.get('acquisition', {})
    trial_config['acquisition']['strategy'] = 'bald'
    
    # Enable Weighted BALD
    trial_config['bald'] = trial_config.get('bald', {})
    trial_config['bald']['use_weighted_bald'] = True
    if 'weighted_bald_sigma' not in trial_config['bald']:
        trial_config['bald']['weighted_bald_sigma'] = 0.1

    prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np = setup_environment(seed, trial_config)
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    oracle = Oracle(true_checker, prior_gen.joint_names)
    learner = ActiveLearner(prior, posterior, oracle, trial_config)

    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)

    log()
    for i in range(budget):
        if i % 1 == 0:
            print(f"[Weighted BALD] Query {i+1}/{budget}", flush=True)
        learner.step(verbose=False)
        log()

    return history


# =============================================================================
    return history


def run_trial_kl_annealed(seed, budget, config):
    """Run BALD with KL Annealing."""
    trial_config = deepcopy(config)
    
    # Enable KL Annealing
    trial_config['vi']['kl_annealing']['enabled'] = True
    trial_config['vi']['kl_annealing']['start_weight'] = 0.001
    trial_config['vi']['kl_annealing']['end_weight'] = 1.0 
    trial_config['vi']['kl_annealing']['duration'] = min(20, budget)
    trial_config['vi']['kl_annealing']['schedule'] = 'linear'

    prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np = setup_environment(seed, trial_config)
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    oracle = Oracle(true_checker, prior_gen.joint_names)
    learner = ActiveLearner(prior, posterior, oracle, trial_config)

    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)

    log()
    for i in range(budget):
        if i % 1 == 0:
            print(f"[KL Annealed] Query {i+1}/{budget}", flush=True)
        learner.step(verbose=False)
        log()

    return history


def run_trial_tau_annealed(seed, budget, config):
    """Run BALD with Temperature (Tau) Annealing."""
    trial_config = deepcopy(config)
    
    # Enable Tau Annealing
    trial_config['bald']['tau_schedule'] = {
        'start': 1.0,
        'end': 0.05,
        'duration': min(20, budget),
        'schedule': 'linear'
    }

    prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np = setup_environment(seed, trial_config)
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    oracle = Oracle(true_checker, prior_gen.joint_names)
    learner = ActiveLearner(prior, posterior, oracle, trial_config)

    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)

    log()
    for i in range(budget):
        if i % 1 == 0:
            print(f"[Tau Annealed] Query {i+1}/{budget}", flush=True)
        learner.step(verbose=False)
        log()

    return history


def run_trial_heuristic(seed, budget, config, bank_size=2000):
    """Run Heuristic Baseline (Dense Banking)."""
    trial_config = deepcopy(config)
    prior_gen, prior, true_limits, true_bumps, true_checker, test_grid_np = setup_environment(seed, trial_config)
    posterior = prior_gen.get_prior(true_limits=true_limits, true_bumps=true_bumps)
    oracle = Oracle(true_checker, prior_gen.joint_names)
    vi = VariationalInference(prior, posterior, trial_config)
    
    # 1. Generate Query Bank
    n_queries = bank_size
    bounds = torch.tensor([prior_gen.anatomical_limits[j] for j in prior_gen.joint_names], device=DEVICE)
    # Uniform random queries
    queries = bounds[:, 0] + torch.rand(n_queries, len(bounds), device=DEVICE) * (bounds[:, 1] - bounds[:, 0])

    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}

    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)

    log()
    
    # Loop
    for i in range(budget):
        if i % 1 == 0:
            print(f"[Heuristic] Query {i+1}/{budget}", flush=True)

        # 2. Sample Candidates from Posterior
        # Assuming posterior.sample returns dict compatible with Checker
        candidates = posterior.sample(200, temperature=1.0, return_list=False)
        
        # 3. Select Test Point
        strat = HeuristicStrategy(candidates, queries, FeasibilityChecker, device=DEVICE, joint_names=prior_gen.joint_names)
        test_point = strat.select_test_point()
        
        outcome = oracle.query(test_point)
        vi.update_posterior(oracle.get_history())
        log()

    return history


# BatchBALD + Direct Optimization Helpers (Legacy)
# =============================================================================

def binary_entropy(p, eps=1e-6):
    """Compute binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
    p = p.clamp(eps, 1 - eps)
    return -p * torch.log(p) - (1 - p) * torch.log(1 - p)

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

def select_batch_bald(prior, bounds, n_points, config, n_candidates=1000, seed=42):
    """Select K points utilizing information from the prior via BatchBALD (Legacy)."""
    torch.manual_seed(seed)
    
    # 1. Sample candidates
    n_joints = bounds.shape[0]
    candidates = bounds[:, 0] + torch.rand(n_candidates, n_joints, device=DEVICE) * (bounds[:, 1] - bounds[:, 0])
    
    # 2. Sample from prior (legacy dict structure)
    prior_samples = prior.sample(32, temperature=1.0, return_list=False) 
    # returns dict with batched tensors: {'joint_limits': (N, n_joints, 2), 'bumps': ...}
    
    # 3. Pre-compute probs: (n_samples, n_candidates)
    # FeasibilityChecker.compute_h_batched returns h (logit-like values)
    # h shape: (N_samples, N_candidates)
    h = FeasibilityChecker.compute_h_batched(
        q=candidates, # (n_candidates, n_joints)
        joint_limits=prior_samples['joint_limits'],
        pairwise_constraints=prior_samples['bumps'],
        joint_names=prior.joint_names,
        config=config
    )
    
    tau = config.get('bald', {}).get('tau', 1.0)
    probs = torch.sigmoid(h / tau)
    
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


def run_trial_batchbald_direct(seed, budget, config):
    """Run BatchBALD + Direct Optimization baseline (Legacy)."""
    trial_config = deepcopy(config)
    prior_gen, prior, true_limits, _, true_checker, test_grid_np = setup_environment(seed, trial_config)
    
    # 1. Select Batch of Points
    bounds = torch.tensor([prior_gen.anatomical_limits[j] for j in prior_gen.joint_names], device=DEVICE)
    batch_points = select_batch_bald(prior, bounds, n_points=budget, config=trial_config, seed=seed)
    
    # 2. Query All Points
    oracle = Oracle(true_checker, prior_gen.joint_names)
    outcomes = []
    for point in batch_points:
        outcomes.append(oracle.query(point))
    outcomes = torch.tensor(outcomes, device=DEVICE, dtype=torch.float32)
    
    history = {'iou': [], 'accuracy': [], 'f1': [], 'param_error': [], 'uncertainty': []}
    
    # We need a posterior object to update. 
    # For Direct Opt, we will clone the prior structure and update its parameters directly.
    posterior = prior_gen.get_prior() # Start fresh
    
    def log():
        iou, acc, f1 = compute_legacy_reachability_metrics(true_checker, posterior, test_grid_np)
        param_err = compute_parameter_error(posterior, true_limits)
        unc = compute_uncertainty(posterior)
        history['iou'].append(iou)
        history['accuracy'].append(acc)
        history['f1'].append(f1)
        history['param_error'].append(param_err)
        history['uncertainty'].append(unc)
        
    log() # Initial
    
    tau = trial_config.get('bald', {}).get('tau', 1.0)
    
    for k in range(1, budget + 1):
        if k % 1 == 0:
            print(f"[BatchBALD Direct] Query {k}/{budget}", flush=True)
        k_points = batch_points[:k]
        k_outcomes = outcomes[:k]
        
        # Reset posterior to prior mean before optimization
        # In legacy, params are in posterior.params['joint_limits'][joint]
        # We need to collect all trainable parameters
        trainable_params = []
        
        # Reset parameters to prior values
        # Be careful: prior object is static, posterior is mutable copy
        # We can just re-copy from prior_gen
        posterior = prior_gen.get_prior()
        
        for joint in posterior.joint_names:
            p = posterior.params['joint_limits'][joint]
            p['lower_mean'].requires_grad_(True)
            p['upper_mean'].requires_grad_(True)
            trainable_params.extend([p['lower_mean'], p['upper_mean']])
            
        optimizer = torch.optim.Adam(trainable_params, lr=0.05)
        
        # Optimization Loop
        for _ in range(100):
            optimizer.zero_grad()
            
            # Predict
            # To use compute_h_batched, we need to format current params as if they were samples
            # But here we have single point estimate (mean)
            # We can construct a "sample" dict with size 1
            
            # Construct 'joint_limits' tensor: (1, n_joints, 2)
            n_joints = len(posterior.joint_names)
            jl_tensor = torch.zeros(1, n_joints, 2, device=DEVICE)
            for i, joint in enumerate(posterior.joint_names):
                p = posterior.params['joint_limits'][joint]
                jl_tensor[0, i, 0] = p['lower_mean']
                jl_tensor[0, i, 1] = p['upper_mean']
            
            # Minimal bumps structure (using means)
            bumps_dict = {} # TODO: If we optimize bumps too? For now just limits.
            # If we ignore bumps optimization, result might be suboptimal but simpler.
            # The prior has bumps. We should probably use prior mean for bumps as fixed constraints?
            # Or just pass empty bumps if we assume checking limits only? 
            # Original prior has bumps. Let's use prior samples for bumps to be consistent?
            # No, we want the "current" bumps. But we aren't optimizing them.
            # Let's just use empty bumps for the direct limit optimization to avoid complexity.
            # Or better, use the bumps from the *prior mean* (which are usually 0 or random).
            # The prior generator initializes random bumps.
            # Let's extract current bump params (which are leaf tensors)
            # Actually, `FeasibilityChecker` expects the structure output by `UserDistribution.sample`
            # which matches the internal param structure.
            
            # Let's just use an empty bumps dict for now to focus on limits.
            # This matches the "Direct Param Inference" assumption (finding valid range).
            bumps_dict = {pair: {} for pair in prior_gen.pairs} 
            # We need to fill it with dummy tensors matching shape if required
            # But compute_h_batched handles empty or missing keys gracefully usually?
            # Let's check compute_h_batched signature in view file... 
            # It iterates over bumps. If dict is empty for a pair, it does nothing.
            
            h = FeasibilityChecker.compute_h_batched(
                q=k_points, # (k, n_joints)
                joint_limits=jl_tensor, # (1, n_joints, 2)
                pairwise_constraints=bumps_dict,
                joint_names=posterior.joint_names,
                config=trial_config
            )
            # h shape: (1, k)
            h = h.squeeze(0)
            
            probs = torch.sigmoid(h / tau)
            probs = torch.clamp(probs, 1e-6, 1-1e-6)
            
            loss = -(k_outcomes * torch.log(probs) + (1-k_outcomes) * torch.log(1-probs)).mean()
            loss.backward()
            optimizer.step()
            
        # Log
        log()
        
    return history


def plot_metric(results_dict, metric, save_dir, n_trials, budget):
    """Plot a single metric across all baselines."""
    # Infer max_len from actual data instead of using budget parameter
    max_len = 0
    for trials in results_dict.values():
        if trials:
            max_len = max(max_len, max(len(trial[metric]) for trial in trials))
    
    if max_len == 0:
        print(f"Warning: No data to plot for {metric}")
        return
    
    # Infer actual number of trials from data
    actual_n_trials = max(len(trials) for trials in results_dict.values() if trials)

    plt.figure(figsize=(10, 6))

    for name, trials in results_dict.items():
        if not trials:
            continue

        data = []
        for trial in trials:
            vals = trial[metric]
            if len(vals) < max_len:
                vals = vals + [vals[-1]] * (max_len - len(vals))
            data.append(vals[:max_len])

        arr = np.array(data)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = range(len(mean))

        plt.plot(x, mean, label=f"{name} (Final: {mean[-1]:.3f})",
                color=COLORS.get(name, 'black'), linewidth=2)
        plt.fill_between(x, mean - std, mean + std,
                        color=COLORS.get(name, 'black'), alpha=0.15)

    plt.title(f'{METRIC_LABELS.get(metric, metric)} - Legacy Pipeline ({actual_n_trials} trials)')
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


def main():
    parser = argparse.ArgumentParser(description="Master Comparison: Legacy Pipeline vs All Baselines")
    parser.add_argument("--baselines", nargs='+',
                       choices=['bald', 'bald-grid', 'gp', 'random', 'quasi-random', 'weighted-bald', 'batchbald-direct', 'kl-annealed', 'tau-annealed', 'heuristic', 'all'],
                       default=['all'],
                       help="Baselines to run")
    parser.add_argument("--metrics", nargs='+',
                       choices=['iou', 'accuracy', 'f1', 'param_error', 'uncertainty', 'all'],
                       default=['all'],
                       help="Metrics to plot")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--budget", type=int, default=20, help="Query budget")
    parser.add_argument("--heuristic-bank-size", type=int, default=2000,
                       help="Size of query bank for heuristic baseline")
    parser.add_argument("--seed", type=int, default=None,
                       help="Base random seed")
    parser.add_argument("--trial-idx", type=int, default=None,
                       help="Index of the trial to run (0 to trials-1). If set, runs only this trial.")
    parser.add_argument("--results-dir", type=str, default="active_learning/results/legacy_master_comparison",
                       help="Directory to save/load individual trial results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save output images (uses default if not specified)")
    parser.add_argument("--merge", action='store_true',
                       help="Merge existing results from results-dir and plot")
    args = parser.parse_args()

    # Expand 'all'
    if 'all' in args.baselines:
        baselines_to_run = ['bald', 'bald-grid', 'gp', 'random', 'quasi-random', 'weighted-bald', 'batchbald-direct', 'kl-annealed', 'tau-annealed', 'heuristic']
    else:
        baselines_to_run = args.baselines

    if 'all' in args.metrics:
        metrics_to_plot = ['iou', 'accuracy', 'f1', 'param_error', 'uncertainty']
    else:
        metrics_to_plot = args.metrics

    # Load config
    config = load_config(os.path.join(os.path.dirname(__file__), '../../configs/legacy.yaml'))
    config['stopping']['budget'] = args.budget

    # Setup output
    save_dir = args.output_dir if args.output_dir else "active_learning/images/legacy_master_comparison"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Get grid strategy resolution from config
    grid_res = config.get('acquisition', {}).get('grid_strategy_resolution', 5)
    
    # Map baseline names to display names and run functions
    baseline_map = {
        'bald': ('BALD', run_trial_bald),
        'bald-grid': ('BALD (Grid)', lambda s, b, c: run_trial_bald_grid(s, b, c, grid_res)),
        'gp': ('GP', run_trial_gp),
        'gp': ('GP', run_trial_gp),
        'random': ('Random', run_trial_random),
        'quasi-random': ('Quasi-Random', run_trial_quasi_random),
        'weighted-bald': ('Weighted BALD', run_trial_weighted_bald),
        'batchbald-direct': ('BatchBALD + Direct Opt', run_trial_batchbald_direct),
        'kl-annealed': ('KL Annealed BALD', run_trial_kl_annealed),
        'tau-annealed': ('Tau Annealed BALD', run_trial_tau_annealed),
        'heuristic': ('Heuristic (Dense)', lambda s, b, c: run_trial_heuristic(s, b, c, args.heuristic_bank_size))
    }

    # Mode 1: Merge Results
    if args.merge:
        print(f"Merging results from {args.results_dir}...")
        results = {baseline_map[b][0]: [] for b in baselines_to_run}
        
        # Load all .pkl files
        files = glob.glob(os.path.join(args.results_dir, "*.pkl"))
        print(f"Found {len(files)} result files.")
        
        for fpath in tqdm(files, desc="Loading results"):
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                    # data is {'baseline': name, 'seed': seed, 'result': history}
                    # We only care about baselines we asked to run/plot
                    b_name = data['baseline']
                    # Map strict keys back to display names if needed, or assume saved as display name?
                    # Let's save as key, map here.
                    if b_name in baseline_map:
                        disp_name = baseline_map[b_name][0]
                        if b_name in baselines_to_run:
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

        # If running a specific trial index
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
        print(f"Metrics to plot: {metrics_to_plot}")
        print(f"Budget: {args.budget}")

        results = {baseline_map[b][0]: [] for b in baselines_to_run}

        for i, seed in zip(indices, seeds_to_run):
            print(f"\n--- Trial {i} (Seed {seed}) ---")
            for baseline in baselines_to_run:
                display_name, run_fn = baseline_map[baseline]
                try:
                    result = run_fn(seed, args.budget, config)
                    
                    # If running singly, save immediately
                    if args.trial_idx is not None:
                        # Save format: { 'baseline': 'bald', 'seed': 123, 'result': history }
                        # Filename: trial_{i}_{baseline}.pkl
                        out_path = os.path.join(args.results_dir, f"trial_{i}_{baseline}.pkl")
                        with open(out_path, 'wb') as f:
                            pickle.dump({'baseline': baseline, 'seed': seed, 'result': result}, f)
                        print(f"Saved {baseline} result to {out_path}")
                    
                    results[display_name].append(result)
                except Exception as e:
                    print(f"\n{display_name} failed for seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
            
            if args.trial_idx is not None:
                print("Single trial complete. Exiting.")
                return

    # Plot each metric (Only if not single trial mode, or if merging)
    if args.trial_idx is None or args.merge:
        for metric in metrics_to_plot:
            plot_metric(results, metric, save_dir, args.trials, args.budget)
            print(f"Saved {metric}.png")

        # Print summary
        print(f"\nResults saved to {save_dir}")
        print("\nFinal metrics (mean +/- std):")
        for name, trials in results.items():
            if trials:
                print(f"\n  {name}:")
                for metric in metrics_to_plot:
                    final_vals = [t[metric][-1] for t in trials]
                    print(f"    {metric}: {np.mean(final_vals):.4f} +/- {np.std(final_vals):.4f}")


if __name__ == "__main__":
    main()
