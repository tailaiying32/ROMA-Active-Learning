"""
k-BALD: Greedy BatchBALD approximation for batch test selection.

Selects k tests per batch by greedily maximizing conditional BALD,
which accounts for redundancy between selected points.

Algorithm:
    1. Select x₁ = argmax BALD(x)
    2. Select x₂ = argmax [BALD(x) - λ·redundancy(x, x₁)]
    3. Select x₃ = argmax [BALD(x) - λ·redundancy(x, {x₁,x₂})]
    ... repeat for k points
    4. Query all k points, update posterior, repeat

The redundancy term measures correlation of predictions across posterior samples.
If two points have highly correlated predictions, querying both is redundant.

Reference: "BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian
           Active Learning" - Kirsch et al., 2019 (NeurIPS)
"""

import torch
from typing import Optional, List, Dict, Any, Tuple

from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.utils import binary_entropy, get_adaptive_param
from active_learning.src.config import DEVICE, create_generator


class KBaldStrategy:
    """
    k-BALD: Greedy BatchBALD approximation for batch test selection.

    Selects k tests per batch by greedily maximizing conditional BALD,
    ensuring diversity through information-theoretic redundancy penalties.
    """

    def __init__(self, decoder, posterior, config: dict):
        """
        Args:
            decoder: LevelSetDecoder model
            posterior: SVGDPosterior or LatentUserDistribution
            config: Configuration dict with optional 'kbald' section
        """
        self.decoder = decoder
        self.posterior = posterior
        self.config = config

        # k-BALD specific config
        kbald_config = config.get('kbald', {})
        self.batch_size = kbald_config.get('batch_size', 5)
        self.n_candidates = kbald_config.get('n_candidates', 5000)
        self.diversity_weight = kbald_config.get('diversity_weight', 0.5)
        self.tau = kbald_config.get('tau', 0.1)
        self.tau_schedule = kbald_config.get('tau_schedule', None)
        self.n_samples = kbald_config.get('n_samples', 32)

        # Device
        self.device = config.get('device', DEVICE)

        # Generator for reproducibility
        self.generator = create_generator(config, self.device)

        # Batch state management
        self.current_batch: List[torch.Tensor] = []
        self.current_batch_scores: List[float] = []

        # Optimization settings (for single-point fallback)
        opt_cfg = config.get('bald_optimization', {})
        self.opt_n_restarts = opt_cfg.get('n_restarts', 10)
        self.opt_n_iters = opt_cfg.get('n_iters_per_restart', 20)

    def _get_current_tau(self, iteration: int = None) -> float:
        """Get current tau value, applying schedule if configured."""
        if self.tau_schedule and iteration is not None:
            return get_adaptive_param(self.tau_schedule, iteration, self.tau)
        return self.tau

    def _generate_candidates(self, bounds: torch.Tensor) -> torch.Tensor:
        """Generate random candidates within bounds.

        Args:
            bounds: (n_joints, 2) tensor with [lower, upper] bounds

        Returns:
            candidates: (n_candidates, n_joints)
        """
        lower = bounds[:, 0].to(self.device)
        upper = bounds[:, 1].to(self.device)
        n_joints = bounds.shape[0]

        candidates = lower + torch.rand(
            self.n_candidates, n_joints,
            device=self.device, generator=self.generator
        ) * (upper - lower)

        return candidates

    def _get_posterior_samples(self) -> torch.Tensor:
        """Get samples from posterior distribution.

        Returns:
            zs: (K, latent_dim) posterior samples
        """
        if hasattr(self.posterior, 'get_particles'):
            # SVGD: use particles directly
            return self.posterior.get_particles().detach()
        else:
            # VI: sample from posterior
            return self.posterior.sample(self.n_samples).detach()

    def _compute_bald_scores(self, candidates: torch.Tensor, zs: torch.Tensor,
                              tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute BALD scores for all candidates.

        Args:
            candidates: (N, n_joints)
            zs: (K, latent_dim) posterior samples
            tau: temperature for sigmoid

        Returns:
            bald_scores: (N,)
            probs: (K, N) - predictions for redundancy computation
        """
        # Get predictions: (K, N)
        logits = LatentFeasibilityChecker.batched_logit_values(
            self.decoder, zs, candidates
        )
        probs = torch.sigmoid(logits / tau)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)

        # BALD = H(E[p]) - E[H(p)]
        mean_probs = probs.mean(dim=0)  # (N,)
        entropy_of_mean = binary_entropy(mean_probs)  # (N,)
        mean_of_entropies = binary_entropy(probs).mean(dim=0)  # (N,)
        bald_scores = entropy_of_mean - mean_of_entropies  # (N,)

        # Ensure non-negative
        bald_scores = torch.relu(bald_scores)

        return bald_scores, probs

    def _compute_redundancy(self, candidate_probs: torch.Tensor,
                            selected_probs_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute redundancy between candidates and already-selected points.

        Redundancy is measured by maximum absolute correlation of predictions
        across posterior samples. High correlation means the points provide
        similar information about the posterior.

        Args:
            candidate_probs: (K, N) predictions for candidates
            selected_probs_list: List of (K,) predictions for selected points

        Returns:
            redundancy: (N,) max correlation with any selected point
        """
        N = candidate_probs.shape[1]

        if not selected_probs_list:
            return torch.zeros(N, device=self.device)

        max_redundancy = torch.zeros(N, device=self.device)

        for selected_probs in selected_probs_list:
            # selected_probs: (K,) predictions for one selected point
            # candidate_probs: (K, N) predictions for all candidates

            # Normalize to zero mean
            cand_mean = candidate_probs.mean(dim=0, keepdim=True)  # (1, N)
            cand_centered = candidate_probs - cand_mean  # (K, N)

            sel_mean = selected_probs.mean()
            sel_centered = selected_probs - sel_mean  # (K,)

            # Compute correlation coefficient for each candidate
            # corr = E[(X - μ_X)(Y - μ_Y)] / (σ_X * σ_Y)
            cov = (cand_centered * sel_centered.unsqueeze(1)).mean(dim=0)  # (N,)

            cand_std = cand_centered.std(dim=0) + 1e-8  # (N,)
            sel_std = sel_centered.std() + 1e-8  # scalar

            corr = cov / (cand_std * sel_std)  # (N,)

            # Take absolute value (negative correlation also means redundancy)
            max_redundancy = torch.max(max_redundancy, corr.abs())

        return max_redundancy

    def select_batch(self, bounds: torch.Tensor, zs: torch.Tensor = None,
                     iteration: int = None) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Select k points using greedy conditional BALD.

        Args:
            bounds: (n_joints, 2) tensor with [lower, upper] bounds
            zs: Optional pre-sampled posterior samples
            iteration: Current iteration for tau scheduling

        Returns:
            selected_points: List of k test points
            selected_scores: List of k BALD scores
        """
        # Generate candidates
        candidates = self._generate_candidates(bounds)

        # Get posterior samples if not provided
        if zs is None:
            zs = self._get_posterior_samples()

        # Get current tau
        tau = self._get_current_tau(iteration)

        # Compute BALD scores and prediction probs for all candidates
        bald_scores, probs = self._compute_bald_scores(candidates, zs, tau)

        selected_indices = []
        selected_probs = []
        selected_scores = []

        # Greedy selection of k points
        for k in range(self.batch_size):
            # Compute conditional BALD: BALD - λ * redundancy
            redundancy = self._compute_redundancy(probs, selected_probs)
            conditional_bald = bald_scores - self.diversity_weight * redundancy

            # Mask already selected points
            for idx in selected_indices:
                conditional_bald[idx] = -float('inf')

            # Select best remaining candidate
            best_idx = conditional_bald.argmax().item()
            selected_indices.append(best_idx)
            selected_probs.append(probs[:, best_idx])
            selected_scores.append(bald_scores[best_idx].item())

        # Extract selected points
        selected_points = [candidates[i].clone() for i in selected_indices]

        return selected_points, selected_scores

    def compute_score(self, test_point: torch.Tensor, zs: torch.Tensor = None,
                      iteration: int = None) -> torch.Tensor:
        """
        Compute BALD score for a single test point.

        This is for compatibility with the LatentActiveLearner interface.

        Args:
            test_point: (n_joints,) or (N, n_joints)
            zs: Optional posterior samples
            iteration: Current iteration

        Returns:
            BALD score(s)
        """
        if test_point.dim() == 1:
            test_point = test_point.unsqueeze(0)

        if zs is None:
            zs = self._get_posterior_samples()

        tau = self._get_current_tau(iteration)
        bald_scores, _ = self._compute_bald_scores(test_point, zs, tau)

        return bald_scores.squeeze()

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
    ) -> Tuple[torch.Tensor, float, List[Dict]]:
        """
        Select next test point.

        If current batch is empty, selects a new batch of k points.
        Returns the next point from the batch.

        Args:
            bounds: (n_joints, 2) tensor with [lower, upper] bounds
            ... (other args for interface compatibility)

        Returns:
            (test_point, bald_score, diagnostics_list) tuple
        """
        diag_stats = []

        # If batch is empty, select new batch
        if not self.current_batch:
            if verbose:
                print(f"  [k-BALD] Selecting new batch of {self.batch_size} points...")

            self.current_batch, self.current_batch_scores = self.select_batch(
                bounds, iteration=iteration
            )

            if verbose:
                scores_str = ", ".join(f"{s:.4f}" for s in self.current_batch_scores)
                print(f"  [k-BALD] Batch BALD scores: [{scores_str}]")

            # Build diagnostics for the batch
            if diagnostics is not None:
                for i, (pt, score) in enumerate(zip(self.current_batch, self.current_batch_scores)):
                    diag_stats.append({
                        'restart': i,
                        'initial_bald': score,
                        'final_bald': score,
                        'p_mean': 0.5,  # Placeholder for compatibility
                        'gate': 1.0,    # Placeholder for compatibility
                        'batch_position': i,
                    })

        # Pop next point from batch
        test_point = self.current_batch.pop(0)
        test_score = self.current_batch_scores.pop(0)

        if verbose:
            remaining = len(self.current_batch)
            print(f"  [k-BALD] Returning point with BALD={test_score:.4f} ({remaining} remaining in batch)")

        return test_point, test_score, diag_stats

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about k-BALD state."""
        return {
            'batch_size': self.batch_size,
            'n_candidates': self.n_candidates,
            'diversity_weight': self.diversity_weight,
            'remaining_in_batch': len(self.current_batch),
        }

    def reset(self):
        """Reset k-BALD state (for new trial)."""
        self.current_batch = []
        self.current_batch_scores = []
