"""
Ensemble BALD acquisition function for Deep Ensemble active learning.

Measures uncertainty via disagreement between K independent posterior members.
BALD_ensemble = H(E_k[p_k]) - E_k[H(p_k)]

Where p_k = E_{z~q_k}[sigmoid(f(z, x)/tau)] is the mean prediction of member k.
"""

import torch
import numpy as np
from typing import List, Optional

from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.config import create_generator
from active_learning.src.utils import binary_entropy, get_adaptive_param


class EnsembleBALD:
    """Ensemble BALD acquisition: selects points that maximize disagreement between ensemble members."""

    def __init__(
        self,
        decoder,
        posteriors: List[LatentUserDistribution],
        config: dict = None,
        prior: LatentUserDistribution = None
    ):
        """
        Args:
            decoder: LevelSetDecoder model
            posteriors: List of K posterior distributions (one per ensemble member)
            config: Configuration dictionary (expects 'bald' and 'bald_optimization' sections)
            prior: Prior distribution (unused here, kept for interface compatibility)
        """
        self.decoder = decoder
        self.posteriors = posteriors
        self.prior = prior
        self.config = config or {}

        # BALD settings
        bald_cfg = self.config.get('bald', {})
        self.tau = bald_cfg.get('tau', 1.0)
        self.n_samples = bald_cfg.get('n_mc_samples', 50)
        self.sampling_temperature = bald_cfg.get('sampling_temperature', 1.0)

        # Weighted BALD
        self.use_weighted_bald = bald_cfg.get('use_weighted_bald', False)
        self.weighted_bald_sigma = bald_cfg.get('weighted_bald_sigma', 0.1)

        # Adaptive schedules
        self.tau_schedule = bald_cfg.get('tau_schedule', None)
        self.weighted_bald_sigma_schedule = bald_cfg.get('weighted_bald_sigma_schedule', None)

        # Optimization settings
        opt_cfg = self.config.get('bald_optimization', {})
        self.opt_n_restarts = opt_cfg.get('n_restarts', 5)
        self.opt_n_iters = opt_cfg.get('n_iters_per_restart', 50)
        self.opt_lr_adam = opt_cfg.get('lr_adam', 0.05)
        self.opt_lr_sgd = opt_cfg.get('lr_sgd', 0.01)
        self.opt_switch_to_sgd_at = opt_cfg.get('switch_to_sgd_at', 0.75)

        # Generator for reproducibility
        device = posteriors[0].device if posteriors else 'cpu'
        self.generator = create_generator(self.config, device)

    def compute_score(
        self,
        test: torch.Tensor,
        member_decoded_params: List[tuple],
        iteration: int = None
    ) -> torch.Tensor:
        """
        Compute Ensemble BALD score for a test point.

        Ensemble BALD = H(E_k[p_k]) - E_k[H(p_k)]
        where p_k = E_{z~q_k}[sigmoid(f(z, x) / tau)] is member k's mean prediction.

        Args:
            test: Test point tensor of shape (n_joints,) or (n_points, n_joints)
            member_decoded_params: List of K decoded parameter tuples, one per ensemble member.
                                   Each tuple is (lower, upper, weights, pres_logits, blob_params).
            iteration: Current iteration for adaptive schedule lookups.

        Returns:
            Ensemble BALD score (scalar tensor, or tensor of shape (n_points,) for batched test)
        """
        current_tau = self._get_current_tau(iteration)

        member_mean_probs = []

        for decoded_params in member_decoded_params:
            # Evaluate logits for all samples of this member: shape (n_samples_k, n_points) or (n_samples_k,)
            logits = LatentFeasibilityChecker.evaluate_from_decoded(test, decoded_params) / current_tau
            probs = torch.sigmoid(logits)

            # Mean prediction for this member (average over samples)
            p_k = probs.mean(dim=0)  # (n_points,) or scalar
            member_mean_probs.append(p_k)

        member_mean_probs = torch.stack(member_mean_probs)  # (K, n_points) or (K,)

        # Ensemble BALD
        p_bar = member_mean_probs.mean(dim=0)  # E_k[p_k]
        entropy_of_mean = binary_entropy(p_bar)  # H(E_k[p_k])
        mean_of_entropies = binary_entropy(member_mean_probs).mean(dim=0)  # E_k[H(p_k)]

        bald_score = entropy_of_mean - mean_of_entropies

        # Weighted BALD gate
        if self.use_weighted_bald:
            w_sigma = self.weighted_bald_sigma
            if self.weighted_bald_sigma_schedule and iteration is not None:
                w_sigma = get_adaptive_param(
                    self.weighted_bald_sigma_schedule, iteration, w_sigma
                )
            gate = torch.exp(-(p_bar - 0.5) ** 2 / (2 * w_sigma ** 2))
            return bald_score * gate

        return bald_score

    def select_test(
        self,
        bounds: torch.Tensor,
        n_restarts: int = None,
        n_iters: int = None,
        lr_adam: float = None,
        lr_sgd: float = None,
        switch_to_sgd_at: float = None,
        verbose: bool = False,
        test_history: list = None,
        iteration: int = None,
        diagnostics=None
    ) -> tuple:
        """
        Find optimal test point via gradient ascent on Ensemble BALD score.

        Follows the same multi-restart Adam->SGD optimization pattern as LatentBALD,
        but samples from all K posteriors and uses ensemble disagreement scoring.

        Args:
            bounds: Tensor (n_joints, 2) with [lower, upper] bounds
            n_restarts: Number of random restarts
            n_iters: Iterations per restart
            lr_adam: Learning rate for Adam phase
            lr_sgd: Learning rate for SGD phase
            switch_to_sgd_at: Fraction of iterations before switching to SGD
            verbose: Print progress
            test_history: Past test point tensors for diversity bonus
            iteration: Current AL iteration index
            diagnostics: Diagnostics object

        Returns:
            (best_test, best_score, diagnostics_stats) tuple
        """
        n_restarts = n_restarts if n_restarts is not None else self.opt_n_restarts
        n_iters = n_iters if n_iters is not None else self.opt_n_iters
        lr_adam = lr_adam if lr_adam is not None else self.opt_lr_adam
        lr_sgd = lr_sgd if lr_sgd is not None else self.opt_lr_sgd
        switch_to_sgd_at = switch_to_sgd_at if switch_to_sgd_at is not None else self.opt_switch_to_sgd_at

        # Diversity settings
        diversity_weight = self.config.get('bald', {}).get('diversity_weight', 0.0)
        history_tensor = None
        if diversity_weight > 0 and test_history is not None and len(test_history) > 0:
            device = self.posteriors[0].device
            history_tensor = torch.stack([h.to(device) for h in test_history])

        device = self.posteriors[0].device
        n_joints = bounds.shape[0]
        K = len(self.posteriors)
        n_per_member = max(1, self.n_samples // K)

        best_test = None
        best_score = -float('inf')
        diag_stats_list = []

        for restart in range(n_restarts):
            # Random initialization within bounds
            lower = bounds[:, 0].detach()
            upper = bounds[:, 1].detach()
            t = lower + torch.rand(n_joints, device=device, generator=self.generator) * (upper - lower)
            t = t.clone().requires_grad_(True)

            optimizer = torch.optim.Adam([t], lr=lr_adam)
            use_sgd = False

            # Sample and pre-decode from each member (fixed per restart for stable optimization)
            member_decoded_params = self._sample_and_decode_all(n_per_member)

            for i in range(n_iters):
                if (not use_sgd) and (i >= int(switch_to_sgd_at * n_iters)):
                    optimizer = torch.optim.SGD([t], lr=lr_sgd)
                    use_sgd = True

                optimizer.zero_grad()

                score = self.compute_score(t, member_decoded_params, iteration=iteration)

                # Diversity bonus
                if history_tensor is not None:
                    distances = torch.norm(t.unsqueeze(0) - history_tensor, dim=1)
                    min_distance = distances.min()
                    score = score + diversity_weight * min_distance

                (-score).backward()
                optimizer.step()

                with torch.no_grad():
                    t.data = torch.clamp(t.data, lower, upper)

            # Final evaluation with fresh samples (unbiased estimate)
            fresh_member_decoded = self._sample_and_decode_all(n_per_member)

            with torch.no_grad():
                final_score = self.compute_score(t, fresh_member_decoded, iteration=iteration).item()

            if final_score > best_score:
                best_score = final_score
                best_test = t.detach().clone()

            # Diagnostics for this restart
            if diagnostics is not None:
                with torch.no_grad():
                    current_tau = self._get_current_tau(iteration)
                    member_p_means = []
                    eval_point = best_test if best_test is not None else t.detach()
                    for decoded in fresh_member_decoded:
                        logits = LatentFeasibilityChecker.evaluate_from_decoded(
                            eval_point, decoded
                        ) / current_tau
                        member_p_means.append(torch.sigmoid(logits).mean().item())

                    p_mean = float(np.mean(member_p_means))

                    gate_val = 1.0
                    if self.use_weighted_bald:
                        w_sigma = self.weighted_bald_sigma
                        if self.weighted_bald_sigma_schedule and iteration is not None:
                            w_sigma = get_adaptive_param(
                                self.weighted_bald_sigma_schedule, iteration, w_sigma
                            )
                        gate_val = float(np.exp(-(p_mean - 0.5) ** 2 / (2 * w_sigma ** 2)))

                    diag_stats_list.append({
                        'restart': restart,
                        'final_bald': final_score,
                        'p_mean': p_mean,
                        'gate': gate_val,
                        'member_disagreement': float(np.std(member_p_means))
                    })

        return best_test, best_score, diag_stats_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_and_decode_all(self, n_per_member: int) -> List[tuple]:
        """Sample from each posterior and pre-decode RBF parameters."""
        member_decoded = []
        for posterior in self.posteriors:
            zs = posterior.sample(
                n_per_member,
                temperature=self.sampling_temperature,
                generator=self.generator
            ).detach()
            with torch.no_grad():
                decoded = LatentFeasibilityChecker.decode_latent_params(self.decoder, zs)
                decoded = tuple(p.detach() for p in decoded)
            member_decoded.append(decoded)
        return member_decoded

    def _get_current_tau(self, iteration: Optional[int]) -> float:
        current_tau = self.tau
        if self.tau_schedule and iteration is not None:
            current_tau = get_adaptive_param(self.tau_schedule, iteration, self.tau)
        return current_tau

