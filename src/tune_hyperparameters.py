import sys
import os
import time
import numpy as np
import json
import copy
from pathlib import Path
import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.pruners import MedianPruner

if not torch.cuda.is_available():
    print("FATAL: CUDA not found on this node. Exiting to avoid recording a 'Failed' trial.")
    sys.exit(1) # Slurm will mark the job as failed, but Optuna won't see a 'broken' trial

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from active_learning.src.legacy.user_distribution import UserDistribution
from active_learning.src.legacy.feasibility_checker import FeasibilityChecker
from active_learning.src.legacy.oracle import Oracle
from active_learning.src.legacy.active_learning_pipeline import ActiveLearner
from active_learning.src.config import load_config, DEVICE
from active_learning.src.legacy.sample_user import UserGenerator
from active_learning.src.legacy.prior_generation import PriorGenerator


def compute_log_likelihood(posterior, true_limits):
    """
    Computes the Log Likelihood of the true parameters under the posterior distribution.
    Higher is better.
    """
    total_ll = 0.0
    count = 0

    # 1. Joint Limits LL
    for joint in posterior.joint_names:
        l_true, u_true = true_limits[joint]

        # Convert true limits to center/log_width parameterization
        true_center = (l_true + u_true) / 2.0
        val = (u_true - l_true) / 2.0
        if val <= 0: val = 1e-6
        true_log_hw = np.log(val)

        # Get Posterior Parameters
        stats = posterior.params['joint_limits'][joint]

        # Center Distribution
        center_dist = torch.distributions.Normal(stats['center_mean'], torch.exp(stats['center_log_std']))
        total_ll += center_dist.log_prob(torch.tensor(true_center, device=DEVICE)).item()

        # Width Distribution
        hw_dist = torch.distributions.Normal(stats['log_hw_mean'], torch.exp(stats['log_hw_log_std']))
        total_ll += hw_dist.log_prob(torch.tensor(true_log_hw, device=DEVICE)).item()

        count += 2

    return total_ll / count if count > 0 else 0.0

def evaluate_single_seed(config, seed):
    """
    Runs a single active learning run with a specific seed.
    Returns (ll_score, time_per_iter, total_iters)
    """
    # Force seed
    if 'user_generation' not in config:
        config['user_generation'] = {}

    config['user_generation']['seed'] = seed

    start_time = time.time()
    max_budget = config.get('stopping', {}).get('budget', 100)

    # Setup
    prior_gen = PriorGenerator(config)
    joint_names = prior_gen.joint_names
    pairs = prior_gen.pairs
    anatomical_limits = prior_gen.anatomical_limits

    # Generate Truth
    generator = UserGenerator(
        config,
        joint_names=joint_names,
        anatomical_limits=anatomical_limits,
        pairs=pairs
    )
    true_limits, true_bumps, true_checker = generator.generate_user()

    # Setup Prior
    prior = prior_gen.get_prior(true_limits, true_bumps)

    # Run Active Learning
    posterior = copy.deepcopy(prior)
    oracle = Oracle(ground_truth=true_checker, joint_names=joint_names)

    learner = ActiveLearner(
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        config=config
    )

    try:
        results = learner.run(verbose=False)
        total_iters = len(results)
    except Exception as e:
        print(f"    Seed {seed} failed: {e}")
        return -100.0, 0.0, max_budget # Penalty

    elapsed = time.time() - start_time
    avg_time = elapsed / total_iters if total_iters > 0 else 0

    # helper to compute metric
    ll_score = compute_log_likelihood(posterior, true_limits)

    return ll_score, avg_time, total_iters

def evaluate_pipeline(config):
    """
    Runs the active learning pipeline over multiple seeds and aggregates loss.
    Loss is defined as: -1 * Mean_LL + Time_Penalty
    """
    seeds = [100, 200, 300, 400, 500] # Fixed seeds for consistency across trials
    losses = []

    ll_scores = []
    times = []
    iters = []

    sys.stdout.flush()
    print(f"Starting Trial (Average over {len(seeds)} seeds)...")

    for seed in seeds:
        ll, t, n_iter = evaluate_single_seed(config, seed=seed)
        ll_scores.append(ll)
        times.append(t)
        iters.append(n_iter)

        # Loss = -LL + alpha * Time
        losses.append(-ll + (0.1 * t))
        print(f"  Seed {seed}: LL={ll:.4f}, Time/Iter={t:.4f}s, Iters={n_iter}")
        sys.stdout.flush()

    # Aggregate
    mean_ll = np.mean(ll_scores)
    mean_time = np.mean(times)
    mean_iters = np.mean(iters)

    loss = np.mean(losses)

    print(f"Trial Result: Mean LL: {mean_ll:.4f} | Loss: {loss:.4f}")
    return loss


def run_optuna(n_trials=50):
    """
    Performs hyperparameter optimization using Optuna.
    """
    def objective(trial):
        config = load_config()

        # BALD Acquisition
        config['bald']['tau'] = trial.suggest_float("bald_tau", 0.05, 1.0)
        config['bald']['sampling_temperature'] = trial.suggest_float("bald_sampling_temperature", 1.0, 2.0)
        config['bald']['n_mc_samples'] = trial.suggest_int("bald_n_mc_samples", 10, 50)
        config['bald_optimization']['n_restarts'] = trial.suggest_int("bald_n_restarts", 5, 20)
        config['bald_optimization']['n_iters_per_restart'] = trial.suggest_int("bald_n_iters_per_restart", 10, 100)
        config['bald_optimization']['lr_adam'] = trial.suggest_float("bald_lr_adam", 0.001, 0.1, log=True)
        config['bald_optimization']['lr_sgd'] = trial.suggest_float("bald_lr_sgd", 0.001, 0.1, log=True)
        config['bald_optimization']['switch_to_sgd_at'] = trial.suggest_float("bald_switch_to_sgd_at", 0.5, 1.0)
        config['bald_optimization']['plateau_patience'] = trial.suggest_int("bald_plateau_patience", 1, 5)
        config['bald_optimization']['plateau_threshold'] = trial.suggest_float("bald_plateau_threshold", 0.0001, 0.1, log=True)

        # VI Update
        config['vi']['n_mc_samples'] = trial.suggest_int("vi_n_mc_samples", 10, 50)
        config['vi']['learning_rate'] = trial.suggest_float("vi_lr", 0.005, 0.1, log=True)
        config['vi']['optimizer_type'] = trial.suggest_categorical("vi_optimizer_type", ["adam", "sgd"])
        config['vi']['max_iters'] = trial.suggest_int("vi_max_iters", 5, 200)
        config['vi']['convergence_tol'] = trial.suggest_float("vi_convergence_tol", 1e-6, 1e-2, log=True)
        config['vi']['patience'] = trial.suggest_int("vi_patience", 1, 5)
        config['vi']['kl_weight'] = trial.suggest_float("vi_kl_weight", 0.05, 2.0, log=True)
        config['vi']['grad_clip'] = trial.suggest_categorical("vi_grad_clip", [1.0, 3.0, 5.0, 10.0])

        # Stopping Criteria
        # config['stopping']['elbo_plateau_window'] = trial.suggest_int("stopping_elbo_plateau_patience", 0, 10)
        # config['stopping']['elbo_plateau_threshold'] = trial.suggest_float("stopping_elbo_plateau_threshold", 0.0001, 1.0, log=True)
        # config['stopping']['uncertainty_threshold'] = trial.suggest_float("stopping_uncertainty_threshold", 0.01, 0.5)
        # config['stopping']['bald_threshold'] = trial.suggest_float("stopping_bald_threshold", 0.01, 0.5)
        # config['stopping']['bald_patience'] = trial.suggest_int("stopping_bald_patience", 0, 10)

        return evaluate_pipeline(config)

    print("Starting Optuna Optimization...")

    storage_path = "./optuna_journal.log"
    storage = JournalStorage(JournalFileBackend(storage_path))

    study_name = sys.argv[1] if len(sys.argv) > 1 else "hyperparameter_optimization_study"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)
    return study.best_params

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    best_params = run_optuna()
    print("\nFinal Best Parameters:")
    print(json.dumps(best_params, indent=2))
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\nBest hyperparameters saved to 'best_hyperparameters.json'")
