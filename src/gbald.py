"""
Geometric BALD (G-BALD) test selection strategy.

Combines:
1. BALD uncertainty (particle disagreement)
2. Ellipsoid-based core-set diversity
3. Boundary penalty to avoid outlier selections

Acquisition function:
    G-BALD(x) = BALD(x) · diversity(x)^λ_d · boundary(x)^λ_b

Where:
    - BALD(x) = H(p̄) - E[H(p)]  (standard mutual information)
    - diversity(x) = min_{s ∈ history} d_ellipsoid(x, s)  (core-set coverage)
    - boundary(x) = σ(-η · (d²_center - 1))  (penalize outliers)

Reference: "Bayesian Active Learning by Disagreements: A Geometric Perspective"
           Cao & Tsang, 2021 (arXiv:2105.02543)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple

from active_learning.src.ellipsoid import AdaptiveEllipsoid
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.utils import binary_entropy, get_adaptive_param
from active_learning.src.config import DEVICE, create_generator


class GeometricBALD:
    """
    G-BALD: Geometric Bayesian Active Learning by Disagreement.

    Extends standard BALD with geometric diversity via ellipsoid-based
    core-set selection. This encourages:
    - Spatial spread of queries (core-set diversity)
    - Avoidance of outlier/boundary points (ellipsoid penalty)
    - Adaptive geometry based on query history
    """

    def __init__(self, decoder, posterior, config: dict):
        """
        Args:
            decoder: LevelSetDecoder model
            posterior: SVGDPosterior or LatentUserDistribution
            config: Configuration dict with optional 'gbald' section
        """
        self.decoder = decoder
        self.posterior = posterior
        self.config = config

        # G-BALD specific config
        gbald_config = config.get('gbald', {})
        self.tau = gbald_config.get('tau', 0.1)
        self.tau_schedule = gbald_config.get('tau_schedule', None)
        self.n_samples = gbald_config.get('n_samples', 32)
        self.eta = gbald_config.get('eta', 2.0)
        self.diversity_weight = gbald_config.get('diversity_weight', 1.0)
        self.diversity_weight_schedule = gbald_config.get('diversity_weight_schedule', None)
        self.boundary_weight = gbald_config.get('boundary_weight', 1.0)
        self.use_log_scale = gbald_config.get('use_log_scale', True)
        self.diversity_normalization = gbald_config.get('diversity_normalization', 'max')

        # Determine dimensionality
        joint_names = config.get('prior', {}).get('joint_names', None)
        if joint_names:
            n_joints = len(joint_names)
        else:
            n_joints = config.get('prior', {}).get('n_joints', 4)

        # Ellipsoid for diversity (in test/joint space)
        # Require more points before enabling ellipsoid to get stable covariance estimate
        # Default: 3*dim instead of dim+1 to avoid poorly-conditioned covariance
        min_points = gbald_config.get('min_points_for_ellipsoid', 3 * n_joints)
        device = config.get('device', DEVICE)
        self.ellipsoid = AdaptiveEllipsoid(dim=n_joints, device=device, min_points_for_ellipsoid=min_points)

        # Warmup: gradually increase boundary weight as ellipsoid becomes more stable
        self.boundary_warmup_queries = gbald_config.get('boundary_warmup_queries', 2 * n_joints)

        # Query history (for core-set distance)
        self.query_history: List[torch.Tensor] = []

        # Device
        self.device = device

        # Optimization settings (from bald_optimization config)
        opt_cfg = config.get('bald_optimization', {})
        self.opt_n_restarts = opt_cfg.get('n_restarts', 20)
        self.opt_n_iters = opt_cfg.get('n_iters_per_restart', 30)
        self.opt_lr_adam = opt_cfg.get('lr_adam', 0.05)
        self.opt_lr_sgd = opt_cfg.get('lr_sgd', 0.01)
        self.opt_switch_to_sgd_at = opt_cfg.get('switch_to_sgd_at', 0.8)

        # Generator for reproducibility
        self.generator = create_generator(config, device)

    def compute_score(self, test_points: torch.Tensor,
                      zs: Optional[torch.Tensor] = None,
                      iteration: int = 0) -> torch.Tensor:
        """
        Compute G-BALD acquisition scores for candidate test points.

        G-BALD(x) = BALD(x) · diversity(x)^λ_d · boundary(x)^λ_b

        Args:
            test_points: Candidate points, shape (N, n_joints)
            zs: Optional pre-sampled latent codes, shape (K, latent_dim)
            iteration: Current iteration (for diagnostics/scheduling)

        Returns:
            scores: G-BALD scores, shape (N,)
        """
        test_points = test_points.to(self.device)

        # 1. Sample from posterior if not provided
        if zs is None:
            if hasattr(self.posterior, 'get_particles'):
                zs = self.posterior.get_particles()  # SVGD
            else:
                zs = self.posterior.sample(self.n_samples)  # VI

        # 2. Compute BALD scores (uncertainty component)
        bald_scores = self._compute_bald(test_points, zs, iteration)

        # 3. Compute diversity scores (core-set component)
        diversity_scores = self._compute_diversity(test_points)

        # 4. Compute boundary penalty
        boundary_scores = self._compute_boundary_penalty(test_points)

        # 5. Get current diversity weight (may be scheduled)
        current_diversity_weight = self.diversity_weight
        if self.diversity_weight_schedule is not None and iteration is not None:
            current_diversity_weight = get_adaptive_param(
                self.diversity_weight_schedule, iteration, self.diversity_weight
            )

        # 6. Combine scores
        if self.use_log_scale:
            # Log-scale combination (more numerically stable)
            log_bald = torch.log(bald_scores + 1e-10)
            log_div = torch.log(diversity_scores + 1e-10)
            log_bound = torch.log(boundary_scores + 1e-10)

            log_scores = (log_bald +
                         current_diversity_weight * log_div +
                         self.boundary_weight * log_bound)
            scores = torch.exp(log_scores)
        else:
            # Multiplicative combination
            scores = (bald_scores *
                     (diversity_scores ** current_diversity_weight) *
                     (boundary_scores ** self.boundary_weight))

        return scores

    def _compute_bald(self, test_points: torch.Tensor,
                      zs: torch.Tensor,
                      iteration: int = None) -> torch.Tensor:
        """
        Compute standard BALD scores: I(y; θ | x) = H(p̄) - E[H(p)]

        Args:
            test_points: (N, n_joints)
            zs: (K, latent_dim)
            iteration: Current iteration for tau scheduling

        Returns:
            bald_scores: (N,)
        """
        # Get current tau from schedule
        current_tau = self.tau
        if self.tau_schedule and iteration is not None:
            current_tau = get_adaptive_param(self.tau_schedule, iteration, self.tau)

        # Get feasibility probabilities for each particle
        # logits shape: (K, N)
        logits = LatentFeasibilityChecker.batched_logit_values(
            self.decoder, zs, test_points
        )
        probs = torch.sigmoid(logits / current_tau)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)

        # Mean probability across particles: p̄ = (1/K) Σ pₖ
        mean_probs = probs.mean(dim=0)  # (N,)

        # Entropy of mean: H(p̄)
        entropy_of_mean = binary_entropy(mean_probs)  # (N,)

        # Mean of entropies: (1/K) Σ H(pₖ)
        entropies = binary_entropy(probs)  # (K, N)
        mean_of_entropies = entropies.mean(dim=0)  # (N,)

        # BALD = H(p̄) - E[H(p)]
        bald_scores = entropy_of_mean - mean_of_entropies

        # Ensure non-negative (numerical stability)
        bald_scores = F.relu(bald_scores)

        return bald_scores

    def _compute_diversity(self, test_points: torch.Tensor) -> torch.Tensor:
        """
        Compute core-set diversity: min ellipsoid distance to query history.

        diversity(x) = min_{s ∈ S} d_ellipsoid(x, s)

        Points far from all historical queries get high diversity scores.

        Args:
            test_points: (N, n_joints)

        Returns:
            diversity_scores: (N,) - higher means more diverse
        """
        N = test_points.shape[0]
        device = test_points.device

        if len(self.query_history) == 0:
            # No history yet - all points equally diverse
            return torch.ones(N, device=device)

        # Stack history into tensor
        history = torch.stack(self.query_history).to(device)  # (H, n_joints)

        # Compute pairwise distances (using ellipsoid if valid, else Euclidean)
        # Shape: (N, H)
        dist_sq = self.ellipsoid.distance_sq(test_points, history)

        # Min distance to any historical point (core-set coverage)
        min_dist_sq = dist_sq.min(dim=1).values  # (N,)

        # Return sqrt for actual distance
        diversity = torch.sqrt(min_dist_sq + 1e-10)

        # Normalize for stability
        if self.diversity_normalization == 'max':
            max_div = diversity.max()
            if max_div > 1e-6:
                diversity = diversity / max_div
        elif self.diversity_normalization == 'softmax':
            diversity = F.softmax(diversity, dim=0) * N  # Scale back to ~1 mean

        return diversity

    def _compute_boundary_penalty(self, test_points: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary penalty: penalize points far from ellipsoid center.

        boundary_penalty(x) = σ(-η · (d²_center(x) - 1))

        - Points inside ellipsoid (d² < 1): penalty ≈ 1
        - Points on ellipsoid surface (d² = 1): penalty = 0.5
        - Points outside ellipsoid (d² > 1): penalty → 0

        Uses gradual warmup to avoid sudden collapse when ellipsoid first becomes valid.

        Args:
            test_points: (N, n_joints)

        Returns:
            boundary_penalty: (N,) - values in (0, 1)
        """
        N = test_points.shape[0]
        device = test_points.device

        if not self.ellipsoid.is_ellipsoid_valid:
            # Not enough points for ellipsoid - no penalty
            return torch.ones(N, device=device)

        # Squared Mahalanobis distance to center
        dist_sq_to_center = self.ellipsoid.mahalanobis_distance_sq(test_points)

        # Ensure it's the right shape
        if dist_sq_to_center.dim() == 0:
            dist_sq_to_center = dist_sq_to_center.unsqueeze(0)

        # Cap extreme Mahalanobis distances to prevent numerical issues
        # Values > 100 would give sigmoid(-η*99) ≈ 0 anyway
        dist_sq_to_center = torch.clamp(dist_sq_to_center, max=100.0)

        # Sigmoid penalty: σ(-η · (d² - 1))
        penalty = torch.sigmoid(-self.eta * (dist_sq_to_center - 1))

        # Gradual warmup: blend penalty with 1.0 based on how many queries beyond min_points
        n_queries = len(self.query_history)
        n_beyond_min = n_queries - self.ellipsoid.min_points
        if n_beyond_min < self.boundary_warmup_queries:
            # Linear warmup from 0 to 1
            warmup_factor = max(0.0, n_beyond_min / self.boundary_warmup_queries)
            # Blend: penalty = warmup_factor * penalty + (1 - warmup_factor) * 1.0
            penalty = warmup_factor * penalty + (1 - warmup_factor)

        return penalty

    def update(self, query_point: torch.Tensor, outcome: float = None):
        """
        Update G-BALD state after a query.

        This updates:
        1. Query history (for diversity computation)
        2. Ellipsoid statistics (for Mahalanobis distance)

        Args:
            query_point: The queried test point, shape (n_joints,)
            outcome: Query outcome (not used, but kept for interface compatibility)
        """
        query_point = query_point.to(self.device)
        if query_point.dim() > 1:
            query_point = query_point.squeeze()

        # Add to history
        self.query_history.append(query_point.detach().clone())

        # Update ellipsoid
        self.ellipsoid.update(query_point.detach())

    def get_diagnostics(self, iteration: int = None) -> Dict[str, Any]:
        """Return diagnostic information about G-BALD state."""
        # Get current diversity weight (may be scheduled)
        current_diversity_weight = self.diversity_weight
        if self.diversity_weight_schedule is not None and iteration is not None:
            current_diversity_weight = get_adaptive_param(
                self.diversity_weight_schedule, iteration, self.diversity_weight
            )

        diag = {
            'n_queries': len(self.query_history),
            'ellipsoid_valid': self.ellipsoid.is_ellipsoid_valid,
            'eta': self.eta,
            'diversity_weight': current_diversity_weight,
            'diversity_weight_base': self.diversity_weight,
            'boundary_weight': self.boundary_weight,
        }

        if self.ellipsoid.n > 0:
            diag['ellipsoid_center'] = self.ellipsoid.center.cpu().tolist()

        if self.ellipsoid.is_ellipsoid_valid:
            cov = self.ellipsoid.get_covariance_matrix()
            if cov is not None:
                # Eigenvalues show ellipsoid shape
                eigvals = torch.linalg.eigvalsh(cov)
                diag['ellipsoid_eigvals'] = eigvals.cpu().tolist()

        return diag

    def reset(self):
        """Reset G-BALD state (for new trial)."""
        self.query_history = []
        self.ellipsoid.reset()

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
        Find optimal test point via gradient ascent on G-BALD score.

        Uses batched multi-restart optimization similar to LatentBALD but
        with the G-BALD acquisition function (BALD * diversity * boundary).

        Args:
            bounds: Tensor of shape (n_joints, 2) with [lower, upper] bounds
            n_restarts: Number of random restarts
            n_iters: Iterations per restart
            lr_adam: Learning rate for Adam optimizer
            lr_sgd: Learning rate for SGD optimizer
            switch_to_sgd_at: Fraction of iterations at which to switch to SGD
            verbose: Print progress
            test_history: Not used (G-BALD tracks history internally)
            iteration: Current iteration index
            diagnostics: Optional diagnostics object

        Returns:
            (best_test, best_score, diagnostics_stats) tuple
        """
        # Use config values if not provided
        n_restarts = n_restarts if n_restarts is not None else self.opt_n_restarts
        n_iters = n_iters if n_iters is not None else self.opt_n_iters
        lr_adam = lr_adam if lr_adam is not None else self.opt_lr_adam
        lr_sgd = lr_sgd if lr_sgd is not None else self.opt_lr_sgd
        switch_to_sgd_at = switch_to_sgd_at if switch_to_sgd_at is not None else self.opt_switch_to_sgd_at

        n_joints = bounds.shape[0]
        lower = bounds[:, 0].to(self.device).detach()
        upper = bounds[:, 1].to(self.device).detach()

        diag_stats_list = []

        # Initialize ALL restarts at once: (n_restarts, n_joints)
        t = lower + torch.rand(n_restarts, n_joints, device=self.device, generator=self.generator) * (upper - lower)
        t = t.clone().requires_grad_(True)

        # Sample latent vectors ONCE, shared across all restarts
        if hasattr(self.posterior, 'get_particles'):
            zs = self.posterior.get_particles().detach()
        else:
            zs = self.posterior.sample(self.n_samples).detach()

        # Evaluate initial scores
        with torch.no_grad():
            initial_scores = self.compute_score(t, zs=zs, iteration=iteration)
            initial_best_score = initial_scores.max().item()

        # Optimizer setup
        optimizer = torch.optim.Adam([t], lr=lr_adam)
        use_sgd = False

        # Optimization loop
        for i in range(n_iters):
            # Switch to SGD after specified fraction
            if (not use_sgd) and (i >= int(switch_to_sgd_at * n_iters)):
                optimizer = torch.optim.SGD([t], lr=lr_sgd)
                use_sgd = True

            optimizer.zero_grad()

            # Compute G-BALD scores for all restarts
            scores = self.compute_score(t, zs=zs, iteration=iteration)

            # Maximize scores via gradient ascent
            (-scores.sum()).backward()

            optimizer.step()

            # Project to bounds
            with torch.no_grad():
                t.data = torch.clamp(t.data, lower, upper)

        # Evaluate final scores with fresh samples for unbiased estimate
        if hasattr(self.posterior, 'get_particles'):
            final_zs = self.posterior.get_particles().detach()
        else:
            final_zs = self.posterior.sample(self.n_samples).detach()

        with torch.no_grad():
            final_scores = self.compute_score(t, zs=final_zs, iteration=iteration)

        # Select best restart
        best_idx = final_scores.argmax()
        best_test = t[best_idx].detach().clone()
        best_score = final_scores[best_idx].item()

        # Diagnostics
        if diagnostics is not None:
            with torch.no_grad():
                # Compute component scores for all restarts
                bald_scores = self._compute_bald(t, final_zs, iteration)
                diversity_scores = self._compute_diversity(t)
                boundary_scores = self._compute_boundary_penalty(t)

                for r in range(n_restarts):
                    diag_stats_list.append({
                        'restart': r,
                        'initial_gbald': initial_scores[r].item(),
                        'final_gbald': final_scores[r].item(),
                        'bald': bald_scores[r].item(),
                        'diversity': diversity_scores[r].item(),
                        'boundary': boundary_scores[r].item(),
                    })

        if verbose:
            gain = best_score - initial_best_score
            print(f"  [G-BALD Opt] Initial Best: {initial_best_score:.4f} -> Final Best: {best_score:.4f} (Gain: {gain:+.4f})")

        return best_test, best_score, diag_stats_list
