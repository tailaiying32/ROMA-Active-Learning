"""
Latent Active Learning Pipeline for User Reachability.

Orchestrates: Test Acquisition → Oracle Query → Posterior Update

Two orthogonal axes:
  - acquisition.strategy: test selection (bald, random, quasi_random, canonical, gp, grid, heuristic, version_space)
  - posterior.method: inference method (vi, ensemble, svgd)

Any strategy can be combined with any posterior method. BALD auto-detects the posterior type.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_bald import LatentBALD
from active_learning.src.latent_variational_inference import LatentVariationalInference
from active_learning.src.diagnostics import Diagnostics
from active_learning.src.config import DEVICE, create_generator
from active_learning.src.utils import calculate_kl_weight


@dataclass
class LatentIterationResult:
    """Result from a single active learning iteration."""
    iteration: int
    test_point: torch.Tensor
    outcome: float
    bald_score: float
    elbo: Optional[float] = None
    grad_norm: Optional[float] = None
    vi_converged: Optional[bool] = None
    vi_iterations: Optional[int] = None


class LatentActiveLearner:
    """
    Active learning loop for adaptive user characterization in latent space.

    Pipeline per iteration:
        1. Acquisition strategy selects most informative test point
        2. Oracle queries ground truth feasibility
        3. VI updates posterior given new observation

    Supports single posterior, ensemble of K posteriors, and particle-based posteriors.
    """

    def __init__(
            self,
            decoder,
            prior: LatentUserDistribution,
            posterior,  # LatentUserDistribution or List[LatentUserDistribution] for ensemble
            oracle: LatentOracle,
            bounds: torch.Tensor,
            config: dict = None,
            acquisition_strategy=None,
            vi=None  # Injected VI optimizer(s): single or List for ensemble
    ):
        """
        Args:
            decoder: Decoder model to map latent codes to outputs
            prior: Prior latent distribution (fixed reference for KL)
            posterior: Posterior latent distribution (or list for ensemble)
            oracle: Oracle for ground truth queries
            bounds: Tensor of shape (n_joints, 2) with [lower, upper] bounds for test points
            config: Configuration dictionary
            acquisition_strategy: Optional strategy object with select_test method.
                                  If None, defaults to LatentBALD.
            vi: Optional VI optimizer(s). If None, creates LatentVariationalInference.
                For ensemble, pass a list of VI optimizers.
        """
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior
        self.oracle = oracle
        self.bounds = bounds.to(DEVICE)
        self.device = DEVICE
        self.config = config if config else {}

        # Determine if this is an ensemble setup
        self._is_ensemble = isinstance(self.posterior, list)

        # Create BALD calculator for score computation
        # Auto-detect BALD variant from posterior type
        if acquisition_strategy is not None and hasattr(acquisition_strategy, 'compute_score'):
            self.bald_calculator = acquisition_strategy
        else:
            if isinstance(self.posterior, list):
                from active_learning.src.ensemble.ensemble_bald import EnsembleBALD
                self.bald_calculator = EnsembleBALD(
                    decoder=self.decoder, posteriors=self.posterior,
                    config=self.config, prior=self.prior)
            elif hasattr(self.posterior, 'get_particles'):
                from active_learning.src.svgd.particle_bald import ParticleBALD
                self.bald_calculator = ParticleBALD(
                    decoder=self.decoder, posterior=self.posterior,
                    config=self.config, prior=self.prior)
            else:
                self.bald_calculator = LatentBALD(
                    decoder=self.decoder, posterior=self.posterior,
                    prior=self.prior, config=self.config)

        # Initialize Diagnostics
        self.diagnostics = None
        if hasattr(self.oracle, 'ground_truth_z'):
            self.diagnostics = Diagnostics(
                joint_names=None,
                true_limits=None,
                true_z=self.oracle.ground_truth_z
            )
        elif hasattr(self.oracle, 'ground_truth_checker') and hasattr(self.oracle.ground_truth_checker, 'z'):
            self.diagnostics = Diagnostics(
                true_z=self.oracle.ground_truth_checker.z
            )

        # Initialize acquisition strategy
        if acquisition_strategy is not None:
            self.strategy = acquisition_strategy
        else:
            # Default: use BALD calculator
            self.strategy = self.bald_calculator

        # Initialize VI (auto-detect based on posterior type)
        if vi is not None:
            self.vi = vi
        else:
            if isinstance(self.posterior, list):
                # Ensemble: create K VI optimizers
                self.vi = [LatentVariationalInference(
                    decoder=self.decoder, prior=self.prior,
                    posterior=p, config=self.config
                ) for p in self.posterior]
            elif hasattr(self.posterior, 'get_particles'):
                from active_learning.src.svgd.svgd_vi import SVGDVariationalInference
                self.vi = SVGDVariationalInference(
                    decoder=self.decoder, prior=self.prior,
                    posterior=self.posterior, config=self.config)
            else:
                self.vi = LatentVariationalInference(
                    decoder=self.decoder, prior=self.prior,
                    posterior=self.posterior, config=self.config)

        self.results: List[LatentIterationResult] = []

        # Epsilon-greedy exploration parameters (acquisition section, with bald fallback)
        acq_cfg = self.config.get('acquisition', {})
        bald_cfg = self.config.get('bald', {})
        self.epsilon = acq_cfg.get('epsilon', bald_cfg.get('epsilon', 0.0))
        self.epsilon_decay = acq_cfg.get('epsilon_decay', bald_cfg.get('epsilon_decay', 1.0))
        self.epsilon_generator = create_generator(self.config, self.device)

        # KL Annealing setup — prefer method-specific config section, fallback to vi.kl_annealing
        vi_ref = self.vi[0] if isinstance(self.vi, list) else self.vi
        self.default_kl_weight = vi_ref.kl_weight
        method = self.config.get('posterior', {}).get('method', 'vi')
        if method in ('svgd', 'sliced_svgd', 'projected_svgd'):
            method_config = self.config.get(method, {})
            self.annealing_config = method_config.get(
                'kl_annealing', self.config.get('vi', {}).get('kl_annealing', {}))
        else:
            self.annealing_config = self.config.get('vi', {}).get('kl_annealing', {})

    def _calculate_kl_weight(self, iteration: int) -> float:
        """Calculate KL weight for the current iteration based on annealing schedule."""
        return calculate_kl_weight(iteration, self.annealing_config, self.default_kl_weight)

    def step(self, verbose: bool = False) -> LatentIterationResult:
        """
        Execute one iteration of active learning.

        Returns:
            LatentIterationResult with test point, outcome, and scores
        """
        iteration = len(self.results)

        # 1. Select test via acquisition strategy (with epsilon-greedy exploration)
        use_random = False
        if self.epsilon > 0:
            rand_val = torch.rand(1, device=self.device, generator=self.epsilon_generator).item()
            if rand_val < self.epsilon:
                use_random = True
                lower = self.bounds[:, 0]
                upper = self.bounds[:, 1]
                test_point = lower + torch.rand(lower.shape[0], device=self.device, generator=self.epsilon_generator) * (upper - lower)
                selection_score = 0.0
                if verbose:
                    print(f"  [Epsilon-greedy: random exploration (eps={self.epsilon:.3f})]")

        if not use_random:
            history = self.oracle.get_history()
            past_tests = [r.test_point for r in history.get_all()]

            select_args = {
                'bounds': self.bounds,
                'verbose': verbose,
                'test_history': past_tests,
                'iteration': iteration,
                'diagnostics': self.diagnostics
            }

            selection_result = self.strategy.select_test(**select_args)

            if len(selection_result) == 3:
                test_point, selection_score, self.bald_last_stats = selection_result
            else:
                test_point, selection_score = selection_result
                self.bald_last_stats = []

        # Ensure we have a valid BALD score for visualization
        if hasattr(self.strategy, 'compute_score'):
            bald_score = selection_score
        else:
            with torch.no_grad():
                # Ensure test_point is on the correct device for the bald_calculator
                tp_for_score = test_point
                if hasattr(self.bald_calculator, 'posteriors'):
                    tp_for_score = test_point.to(self.bald_calculator.posteriors[0].device)
                elif hasattr(self.bald_calculator, 'posterior'):
                    tp_for_score = test_point.to(self.bald_calculator.posterior.device)

                if hasattr(self.bald_calculator, '_sample_and_decode_all'):
                    # EnsembleBALD requires member_decoded_params
                    n_mc = self.config.get('bald', {}).get('n_mc_samples', 10)
                    member_decoded = self.bald_calculator._sample_and_decode_all(n_mc)
                    bald_score = self.bald_calculator.compute_score(tp_for_score, member_decoded).item()
                else:
                    bald_score = self.bald_calculator.compute_score(tp_for_score).item()

        # 2. Query oracle
        outcome = self.oracle.query(test_point)

        # 2b. Strategy-specific update (for GP, VersionSpace, etc.)
        if hasattr(self.strategy, 'post_query_update'):
            self.strategy.post_query_update(test_point, outcome, self.oracle.get_history())

        # 2c. Update acquisition strategy state (for G-BALD ellipsoid tracking, etc.)
        if hasattr(self.bald_calculator, 'update'):
            self.bald_calculator.update(test_point, outcome)

        # 3. Update posterior via VI
        current_kl_weight = self._calculate_kl_weight(iteration)
        history = self.oracle.get_history()

        best_idx = 0
        if isinstance(self.vi, list):
            # Ensemble: update each VI optimizer independently
            vi_results = []
            for vi_opt in self.vi:
                vr = vi_opt.update_posterior(history, kl_weight=current_kl_weight, diagnostics=None, iteration=iteration)
                vi_results.append(vr)
            best_idx = max(range(len(vi_results)), key=lambda i: vi_results[i].final_elbo)
            vi_result = vi_results[best_idx]
            self.last_vi_result = vi_result
        else:
            vi_result = self.vi.update_posterior(history, kl_weight=current_kl_weight, diagnostics=self.diagnostics, iteration=iteration)
            self.last_vi_result = vi_result

        # Log to diagnostics
        if self.diagnostics:
            if self._is_ensemble:
                diag_posterior = self.posterior[best_idx]
                diag_vi = self.vi[best_idx] if isinstance(self.vi, list) else self.vi
            else:
                diag_posterior = self.posterior
                diag_vi = self.vi

            self.diagnostics.log_iteration(
                iteration=iteration,
                prior=self.prior,
                posterior=diag_posterior,
                query=test_point,
                true_checker=self.oracle.ground_truth_checker if hasattr(self.oracle, 'ground_truth_checker') else None,
                grad_norm=vi_result.final_grad_norm,
                vi_result=vi_result,
                bald_opt_stats=self.bald_last_stats if hasattr(self, 'bald_last_stats') else [],
                likelihood=diag_vi.likelihood(history, iteration=iteration).item(),
                kl_divergence=diag_vi.regularizer(kl_weight=current_kl_weight).item()
            )

        # Compute final ELBO
        if isinstance(self.vi, list):
            mean_elbo = float(np.mean([vr.final_elbo for vr in vi_results]))
            elbo = mean_elbo
        else:
            ll = self.vi.likelihood(history, iteration=iteration).item()
            kl = self.vi.regularizer(kl_weight=current_kl_weight).item()
            elbo = ll - kl

        result = LatentIterationResult(
            iteration=iteration,
            test_point=test_point,
            outcome=outcome,
            bald_score=bald_score,
            elbo=elbo,
            grad_norm=vi_result.final_grad_norm,
            vi_converged=vi_result.converged if not isinstance(self.vi, list) else all(vr.converged for vr in vi_results),
            vi_iterations=vi_result.n_iterations if not isinstance(self.vi, list) else max(vr.n_iterations for vr in vi_results)
        )
        self.results.append(result)

        if verbose:
            status = "feasible" if outcome > 0.5 else "infeasible"
            if isinstance(self.vi, list):
                elbos = [vr.final_elbo for vr in vi_results]
                elbo_str = ", ".join(f"{e:.1f}" for e in elbos)
                print(
                    f"[{iteration}] BALD={bald_score:.4f}, {status}, "
                    f"ELBO(mean)={elbo:.2f} [{elbo_str}], K={len(self.vi)}"
                )
            else:
                ll_val = self.vi.likelihood(history, iteration=iteration).item()
                kl_val = self.vi.regularizer(kl_weight=current_kl_weight).item()
                print(f"[{iteration}] BALD={bald_score:.4f}, {status}, ELBO={elbo:.2f} (LL={ll_val:.2f}, KL={kl_val:.2f}, W={current_kl_weight:.4f}), grad={vi_result.final_grad_norm:.4f}", flush=True)

        # Apply epsilon decay for next iteration
        self.epsilon *= self.epsilon_decay

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
                if max(recent_elbos) - min(recent_elbos) < threshold:
                    return True, f"ELBO Plateau (range < {threshold} over {window} iters)"

        # 4. Posterior Uncertainty (latent space version)
        if stop_cfg.get('uncertainty_enabled', False):
            threshold = stop_cfg.get('uncertainty_threshold', 0.05)

            if self._is_ensemble:
                mean_std = float(np.mean([torch.exp(p.log_std).mean().item() for p in self.posterior]))
            else:
                mean_std = torch.exp(self.posterior.log_std).mean().item()
            if mean_std < threshold:
                return True, f"Posterior Uncertainty {mean_std:.4f} < {threshold}"

        return False, ""

    def run(self, n_iterations: int = None, verbose: bool = True) -> List[LatentIterationResult]:
        """
        Run active learning loop until stopping criteria met or n_iterations reached.

        Args:
            n_iterations: Optional override for max iterations (budget)
            verbose: Print progress

        Returns:
            List of LatentIterationResult
        """
        budget = self.config.get('stopping', {}).get('budget', 100)
        if n_iterations is not None:
            budget = n_iterations

        if self._is_ensemble:
            print(f"Starting Ensemble Active Learning (K={len(self.posterior)}, Budget: {budget})...")
        else:
            print(f"Starting Latent Active Learning (Max Budget: {budget})...")

        while len(self.results) < budget:
            self.step(verbose=verbose)

            should_stop, reason = self.check_stopping_criteria()
            if should_stop:
                if verbose:
                    print(f"Stopping Criteria Met: {reason}")
                break

        return self.results

    def get_posterior(self):
        """Return current posterior. For ensemble, returns best ELBO member."""
        if self._is_ensemble:
            if not self.results:
                return self.posterior[0]
            history = self.oracle.get_history()
            best_idx = 0
            best_elbo = -float('inf')
            current_kl = self._calculate_kl_weight(len(self.results))
            for i, vi in enumerate(self.vi):
                ll = vi.likelihood(history, iteration=len(self.results)).item()
                kl = vi.regularizer(kl_weight=current_kl).item()
                elbo_val = ll - kl
                if elbo_val > best_elbo:
                    best_elbo = elbo_val
                    best_idx = i
            return self.posterior[best_idx]
        return self.posterior

    def ensemble_predict_probs(self, test_points: torch.Tensor) -> torch.Tensor:
        """
        Compute mean predicted probability across ensemble members.
        Only valid for ensemble posteriors.

        Args:
            test_points: Tensor of shape (N, n_joints)

        Returns:
            Mean probabilities tensor of shape (N,)
        """
        if not self._is_ensemble:
            raise ValueError("ensemble_predict_probs requires ensemble posteriors")
        with torch.no_grad():
            all_probs = []
            for p in self.posterior:
                z = p.mean.unsqueeze(0)
                logits = LatentFeasibilityChecker.batched_logit_values(self.decoder, z, test_points)
                probs = torch.sigmoid(logits).squeeze(0)
                all_probs.append(probs)
            return torch.stack(all_probs).mean(dim=0)

    def get_history(self):
        """Return test history from oracle."""
        return self.oracle.get_history()
