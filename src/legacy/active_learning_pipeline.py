"""
Active Learning Pipeline for User Reachability.

Orchestrates: Test Acquisition (BALD/Random/Canonical/Quasi-Random) → Oracle Query → Posterior Update (VI)
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from active_learning.src.legacy.user_distribution import UserDistribution
from active_learning.src.legacy.feasibility_checker import FeasibilityChecker
from active_learning.src.legacy.oracle import Oracle
from active_learning.src.legacy.bald import BALD
from active_learning.src.legacy.variational_inference import VariationalInference
from active_learning.src.diagnostics import Diagnostics
from active_learning.src.config import DEVICE


@dataclass
class IterationResult:
    """Result from a single active learning iteration."""
    iteration: int
    test_point: torch.Tensor
    outcome: bool
    bald_score: float
    elbo: Optional[float] = None
    grad_norm: Optional[float] = None
    vi_converged: Optional[bool] = None
    vi_iterations: Optional[int] = None


class ActiveLearner:
    """
    Active learning loop for adaptive user characterization.

    Pipeline per iteration:
        1. Acquisition strategy selects test point (BALD/Random/Canonical/Quasi-Random)
        2. Oracle queries ground truth feasibility
        3. VI updates posterior given new observation
    """

    def __init__(
            self,
            prior: UserDistribution,
            posterior: UserDistribution,
            oracle: Oracle,
            config: dict = None
    ):
        """
        Args:
            prior: Prior distribution (fixed reference for KL)
            posterior: Posterior distribution (updated each iteration)
            oracle: Oracle for ground truth queries
            config: Configuration dictionary
        """
        self.prior = prior
        self.posterior = posterior
        self.oracle = oracle
        self.device = DEVICE
        self.config = config if config else {}

        # Get acquisition strategy from config
        acq_cfg = self.config.get('acquisition', {})
        self.acquisition_strategy = acq_cfg.get('strategy', 'bald')

        # Initialize BALD (used by 'bald' and 'quasi-random' strategies)
        self.bald = BALD(
            posterior=self.posterior,
            config=self.config
        )

        # Initialize VI
        self.vi = VariationalInference(
            prior=self.prior,
            posterior=self.posterior,
            config=self.config
        )

        # Initialize Diagnostics
        # Need true_limits from Oracle? Oracle is abstract. 
        # But we can access the underlying true_checker from oracle usually, or pass it in.
        # Since this pipeline is generic, we might need a way to get ground truth for diagnostics if available.
        # However, Diagnostics is only for debugging. 
        # Let's assume the user passes a true_checker if they want full diagnostics.
        # For now, we initialize it if possible.
        self.diagnostics = None
        if hasattr(self.oracle, 'ground_truth'):
            # This relies on Oracle having a 'ground_truth' attribute (FeasibilityChecker)
            # which is true for the standard Oracle implementation in legacy/oracle.py
            self.diagnostics = Diagnostics(
                joint_names=self.posterior.joint_names,
                true_limits=self.oracle.ground_truth.joint_limits # Assuming standard FeasibilityChecker
            )

        self.results: List[IterationResult] = []

        # KL Annealing setup
        self.annealing_config = self.config.get('vi', {}).get('kl_annealing', {})
        self.annealing_enabled = self.annealing_config.get('enabled', False)
        self.kl_start = self.annealing_config.get('start_weight', 0.0)
        self.kl_end = self.annealing_config.get('end_weight', self.vi.kl_weight)
        self.kl_duration = self.annealing_config.get('duration', 10)
        self.kl_schedule = self.annealing_config.get('schedule', 'linear')

    def _calculate_kl_weight(self, iteration: int) -> float:
        """Calculate KL weight for the current iteration based on annealing schedule."""
        if not self.annealing_enabled:
            return self.vi.kl_weight

        if iteration >= self.kl_duration:
            return self.kl_end

        # Schedules
        if self.kl_schedule == 'step':
            return self.kl_start
            
        elif self.kl_schedule == 'linear':
            progress = iteration / float(self.kl_duration)
            return self.kl_start + (self.kl_end - self.kl_start) * progress
            
        elif self.kl_schedule == 'cosine':
            progress = iteration / float(self.kl_duration)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress)) # 1 -> 0
            return self.kl_start * cosine_factor + self.kl_end * (1 - cosine_factor)
            
        elif self.kl_schedule == 'logistic':
             k = 10.0 / self.kl_duration
             x0 = self.kl_duration / 2.0
             sigmoid = 1 / (1 + np.exp(-k * (iteration - x0)))
             return self.kl_start + (self.kl_end - self.kl_start) * sigmoid

        return self.kl_end

        # Strategy-specific initialization
        if self.acquisition_strategy == 'canonical':
            self._init_canonical_strategy(acq_cfg)
        elif self.acquisition_strategy == 'quasi-random':
            self._init_quasi_random_strategy(acq_cfg)

    def _init_canonical_strategy(self, acq_cfg: dict):
        """Initialize canonical query strategy."""
        canonical_path = acq_cfg.get('canonical_path', 'models/canonical_queries.npz')
        try:
            data = np.load(canonical_path)
            self.canonical_queries = torch.tensor(data['queries'], device=self.device, dtype=torch.float32)
            self.canonical_idx = 0
        except FileNotFoundError:
            print(f"Warning: Canonical queries file not found at {canonical_path}. Falling back to BALD.")
            self.acquisition_strategy = 'bald'

    def _init_quasi_random_strategy(self, acq_cfg: dict):
        """Initialize quasi-random (Sobol) strategy."""
        from scipy.stats import qmc
        self.n_quasi_random = acq_cfg.get('n_quasi_random', 10)

        print(f"[ActiveLearner] Quasi-random strategy initialized with n_quasi_random={self.n_quasi_random}")

        # Generate Sobol sequence
        n_joints = len(self.posterior.joint_names)
        sampler = qmc.Sobol(d=n_joints, scramble=True)
        sobol_samples = sampler.random(n=self.n_quasi_random)  # In [0, 1]
        print(f"[ActiveLearner] Generated {len(sobol_samples)} Sobol points")

        # Scale to anatomical bounds (ensure all tensors on same device)
        lower = torch.tensor(
            [self.posterior.anatomical_limits[j][0] for j in self.posterior.joint_names],
            device=self.device,
            dtype=torch.float32
        )
        upper = torch.tensor(
            [self.posterior.anatomical_limits[j][1] for j in self.posterior.joint_names],
            device=self.device,
            dtype=torch.float32
        )

        self.quasi_random_queries = torch.tensor(sobol_samples, device=self.device, dtype=torch.float32)
        self.quasi_random_queries = lower + (upper - lower) * self.quasi_random_queries
        self.quasi_random_idx = 0

    def _select_test_point(self, iteration: int, verbose: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Select test point based on acquisition strategy.

        Returns:
            (test_point, acquisition_score) tuple
        """
        lower = torch.tensor([self.posterior.anatomical_limits[j][0] for j in self.posterior.joint_names])
        upper = torch.tensor([self.posterior.anatomical_limits[j][1] for j in self.posterior.joint_names])
        bounds = torch.stack([lower, upper], dim=1).detach().to(self.device)  # (n_joints, 2)

        if self.acquisition_strategy == 'bald':
            # Pass diagnostics (if available) to capture optimization stats
            test, score, stats = self.bald.select_test(bounds, verbose=verbose, iteration=iteration, diagnostics=self.diagnostics)
            self.bald_last_stats = stats
            return test, score

        elif self.acquisition_strategy == 'random':
            # Uniform random sampling from anatomical bounds
            rand_noise = torch.rand_like(lower)
            test_point = lower + (upper - lower) * rand_noise
            test_point = test_point.to(self.device)
            return test_point, 0.0  # No acquisition score for random

        elif self.acquisition_strategy == 'canonical':
            # Use pre-computed canonical queries
            if self.canonical_idx < len(self.canonical_queries):
                test_point = self.canonical_queries[self.canonical_idx]
                self.canonical_idx += 1
                return test_point, 0.0
            else:
                # Fallback to BALD after exhausting canonical queries
                test, score, stats = self.bald.select_test(bounds, verbose=verbose)
                self.bald_last_stats = stats
                return test, score

        elif self.acquisition_strategy == 'quasi-random':
            # Use Sobol sequence for initial queries, then switch to BALD
            if self.quasi_random_idx < self.n_quasi_random:
                test_point = self.quasi_random_queries[self.quasi_random_idx]
                self.quasi_random_idx += 1
                return test_point, 0.0
            else:
                # Switch to BALD after initial quasi-random phase
                if self.quasi_random_idx == self.n_quasi_random:
                    print(f"[ActiveLearner] Switching from quasi-random to BALD after {self.n_quasi_random} Sobol points")
                test, score, stats = self.bald.select_test(bounds, verbose=verbose, iteration=iteration, diagnostics=self.diagnostics)
                self.bald_last_stats = stats
                return test, score

        else:
            raise ValueError(f"Unknown acquisition strategy: {self.acquisition_strategy}")

    def step(self, verbose: bool = False) -> IterationResult:
        """
        Execute one iteration of active learning.

        Returns:
            IterationResult with test point, outcome, and scores
        """
        iteration = len(self.results)
        # print(f"[ActiveLearner] Step {iteration} - BEGIN", flush=True)

        # 1. Select test via acquisition strategy
        # print(f"[ActiveLearner] Step {iteration} - Selecting test via {self.acquisition_strategy}", flush=True)
        test_point, acquisition_score = self._select_test_point(iteration, verbose=verbose)
        # print(f"[ActiveLearner] Step {iteration} - Test selection DONE", flush=True)

        # 2. Query oracle
        # print(f"[ActiveLearner] Step {iteration} - Oracle.query()", flush=True)
        outcome = self.oracle.query(test_point)
        # print(f"[ActiveLearner] Step {iteration} - Oracle.query() DONE", flush=True)

        # 3. Update posterior via VI
        # print(f"[ActiveLearner] Step {iteration} - VI.update_posterior()", flush=True)
        current_kl_weight = self._calculate_kl_weight(iteration)
        history = self.oracle.get_history()
        vi_result = self.vi.update_posterior(history, diagnostics=self.diagnostics, iteration=iteration, kl_weight=current_kl_weight)
        # print(f"[ActiveLearner] Step {iteration} - VI.update_posterior() DONE", flush=True)
        
        # Log to diagnostics
        if self.diagnostics:
             # Extract bald_opt_stats if it was populated during select_test
             bald_stats = []
             if hasattr(self.diagnostics, 'bald_opt_stats') and self.diagnostics.bald_opt_stats:
                  # Move stats from temp storage to log (active_learning_pipeline logic handling)
                  pass 
             
             self.diagnostics.log_iteration(
                 iteration=iteration,
                 prior=self.prior, # This is technically the PREVIOUS posterior before update? No, VI updates in place.
                 # Actually, Diagnostics.log_iteration expects prior and posterior to track movement.
                 # In legacy pipeline, VI updates self.posterior IN PLACE. 
                 # Currently self.posterior IS the updated posterior.
                 # self.prior (the member) is the INITIAL prior, not the previous iteration's posterior.
                 # Diagnostics handles previous params internally.
                 posterior=self.posterior,
                 query=test_point,
                 true_checker=self.oracle.ground_truth if hasattr(self.oracle, 'ground_truth') else None,
                 grad_norm=vi_result.final_grad_norm,
                 vi_result=vi_result,
                 # We need to get bald_opt_stats used for THIS selection.
                 # The BALD.select_test returned them!
                 bald_opt_stats=self.bald_last_stats if hasattr(self, 'bald_last_stats') else [],
                 likelihood=self.vi.likelihood(history).item(),
                 kl_divergence=self.vi.regularizer(kl_weight=current_kl_weight).item()
             )

        # Compute final ELBO
        ll = self.vi.likelihood(history).item()
        kl = self.vi.regularizer(kl_weight=current_kl_weight).item()
        elbo = ll - kl

        result = IterationResult(
            iteration=iteration,
            test_point=test_point,
            outcome=outcome,
            bald_score=acquisition_score,  # Now generic acquisition score
            elbo=elbo,
            grad_norm=vi_result.final_grad_norm,
            vi_converged=vi_result.converged,
            vi_iterations=vi_result.n_iterations
        )
        self.results.append(result)

        if verbose:
            status = "feasible" if outcome else "infeasible"

            # Determine strategy name for logging
            if self.acquisition_strategy == 'quasi-random' and self.quasi_random_idx >= self.n_quasi_random:
                # Already switched to BALD
                strategy_info = "BALD"
            else:
                strategy_info = f"{self.acquisition_strategy.upper()}"

            # Add acquisition score if applicable
            if acquisition_score > 0:
                strategy_info += f"={acquisition_score:.4f}"

            print(f"[{iteration}] {strategy_info}, {status}, ELBO={elbo:.2f} (LL={ll:.2f}, KL={kl:.2f}, W={current_kl_weight:.4f}), ∇={vi_result.final_grad_norm:.4f}", flush=True)

        # print(f"[ActiveLearner] Step {iteration} - END", flush=True)
        return result

    def check_stopping_criteria(self) -> Tuple[bool, str]:
        """
        Check if any stopping criteria are met.
        Returns:
            (should_stop, reason)
        """
        stop_cfg = self.config.get('stopping', {})

        # 1. Budget
        budget = stop_cfg.get('budget', 100)
        if len(self.results) >= budget:
            return True, "Budget Reached"

        # 2. BALD Score Stopping
        if stop_cfg.get('bald_enabled', False):
            threshold = stop_cfg.get('bald_threshold', 0.05)
            patience = stop_cfg.get('bald_patience', 3)

            if len(self.results) >= patience:
                recent_scores = [r.bald_score for r in self.results[-patience:]]
                if all(s < threshold for s in recent_scores):
                    return True, f"BALD Score < {threshold} for {patience} iters"

        # 3. ELBO Plateau
        if stop_cfg.get('elbo_plateau_enabled', False):
            window = stop_cfg.get('elbo_plateau_window', 5)
            threshold = stop_cfg.get('elbo_plateau_threshold', 0.05)

            if len(self.results) >= window:
                recent_elbos = [r.elbo for r in self.results[-window:]]
                # Check if the range of ELBO values in the window is small
                if max(recent_elbos) - min(recent_elbos) < threshold:
                     return True, f"ELBO Plateau (range < {threshold} over {window} iters)"

        # 4. Posterior Uncertainty
        if stop_cfg.get('uncertainty_enabled', False):
            threshold = stop_cfg.get('uncertainty_threshold', 0.05)

            # Calculate mean std of joint limits (lower and upper bounds)
            total_std = 0.0
            count = 0
            for joint in self.posterior.joint_names:
                p = self.posterior.params['joint_limits'][joint]
                total_std += torch.exp(p['lower_log_std']).item()
                total_std += torch.exp(p['upper_log_std']).item()
                count += 2

            mean_std = total_std / count
            if mean_std < threshold:
                return True, f"Posterior Uncertainty {mean_std:.4f} < {threshold}"

        return False, ""

    def run(self, n_iterations: int = None, verbose: bool = True) -> List[IterationResult]:
        """
        Run active learning loop until stopping criteria met or n_iterations reached.

        Args:
            n_iterations: Optional override for max iterations (budget)
            verbose: Print progress

        Returns:
            List of IterationResult
        """
        budget = self.config.get('stopping', {}).get('budget', 100)
        if n_iterations is not None:
            budget = n_iterations

        print(f"Starting Active Learning (Max Budget: {budget})...")

        while len(self.results) < budget:
            self.step(verbose=verbose)

            should_stop, reason = self.check_stopping_criteria()
            if should_stop:
                if verbose:
                    print(f"Stopping Criteria Met: {reason}")
                break

        return self.results

    def get_posterior(self) -> UserDistribution:
        """Return current posterior."""
        return self.posterior

    def get_history(self):
        """Return test history from oracle."""
        return self.oracle.get_history()
