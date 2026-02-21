"""
Prior Boundary acquisition strategy.

Selects initial queries at points on the prior mean's predicted feasibility
boundary, then switches to BALD.

Evaluates p(feasible | z_mean) for each candidate test point and keeps those
closest to the decision boundary (p ≈ 0.5). Farthest-point sampling then
spreads warmup queries evenly along this boundary.
"""

import torch
from typing import Tuple

from active_learning.src.config import DEVICE
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker


class PriorBoundaryStrategy:
    """
    Prior Boundary acquisition: target prior mean's decision boundary, then BALD.

    For the first n_warmup queries, returns pre-computed points that lie on the
    prior mean's feasibility boundary (where p(feasible) ≈ 0.5).
    After the warmup phase, delegates to the BALD strategy.
    """

    def __init__(
        self,
        decoder,
        prior,
        bald_strategy,
        n_warmup: int = 10,
        n_candidates: int = 5000,
        boundary_percentile: float = 0.05,
        device: str = DEVICE
    ):
        """
        Initialize prior boundary acquisition.

        Args:
            decoder: LevelSetDecoder model
            prior: Prior distribution (LatentUserDistribution)
            bald_strategy: Instance of LatentBALD/ParticleBALD for posterior-guided selection
            n_warmup: Number of boundary-targeted warmup queries
            n_candidates: Number of candidate test points to evaluate
            boundary_percentile: Fraction of candidates closest to p=0.5 to keep
            device: Torch device
        """
        self.decoder = decoder
        self.prior = prior
        self.bald = bald_strategy
        self.n_warmup = n_warmup
        self.n_candidates = n_candidates
        self.boundary_percentile = boundary_percentile
        self.device = device
        self.query_count = 0
        self.warmup_points = None

    def _compute_warmup_points(self, bounds: torch.Tensor):
        """
        Pre-compute warmup points on the prior mean's decision boundary.

        1. Generate uniform candidates across anatomical bounds
        2. Evaluate p(feasible | z_mean) for each candidate
        3. Keep the closest candidates to p = 0.5 (decision boundary)
        4. Farthest-point sample for spatial coverage along the boundary

        Args:
            bounds: (n_joints, 2) tensor of [min, max] bounds
        """
        n_dims = bounds.shape[0]

        # 1. Uniform candidates across anatomical bounds
        candidates = bounds[:, 0] + torch.rand(
            self.n_candidates, n_dims, device=self.device
        ) * (bounds[:, 1] - bounds[:, 0])

        # 2. Evaluate feasibility under the prior mean
        z_mean = self.prior.mean.unsqueeze(0)  # (1, latent_dim)
        with torch.no_grad():
            logits = LatentFeasibilityChecker.batched_logit_values(
                self.decoder, z_mean, candidates
            )
        probs = torch.sigmoid(logits.squeeze(0))  # (n_candidates,)

        # 3. Distance to decision boundary: |p - 0.5|  (lower = ON boundary)
        boundary_dist = (probs - 0.5).abs()

        # Keep the closest boundary_percentile fraction to p=0.5
        n_keep = max(int(self.n_candidates * self.boundary_percentile), self.n_warmup * 2)
        threshold = boundary_dist.quantile(self.boundary_percentile)
        boundary_mask = boundary_dist <= threshold
        boundary_candidates = candidates[boundary_mask]

        n_select = min(self.n_warmup, len(boundary_candidates))

        # 4. Farthest-point sampling for spatial coverage along the boundary
        #    Start from the candidate closest to p=0.5
        boundary_dists = boundary_dist[boundary_mask]
        selected = [boundary_dists.argmin().item()]

        for _ in range(n_select - 1):
            selected_pts = boundary_candidates[selected]              # (k, D)
            dists = torch.cdist(boundary_candidates.unsqueeze(0),
                                selected_pts.unsqueeze(0)).squeeze(0)  # (n_boundary, k)
            min_dists = dists.min(dim=1).values                        # (n_boundary,)
            # Exclude already-selected
            for s in selected:
                min_dists[s] = -1.0
            selected.append(min_dists.argmax().item())

        self.warmup_points = boundary_candidates[selected]
        print(f"Prior Boundary: selected {n_select} warmup points from "
              f"{len(boundary_candidates)} boundary candidates "
              f"(|p-0.5| threshold={threshold:.4f}, "
              f"prob range=[{probs[boundary_mask].min():.3f}, {probs[boundary_mask].max():.3f}])")

    def select_test(
        self,
        bounds: torch.Tensor,
        test_history=None,
        verbose: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, float]:
        """
        Select next test point.

        Args:
            bounds: (n_joints, 2) tensor of [min, max] bounds
            test_history: Optional history of test results (passed to BALD)
            verbose: Print progress (passed to BALD)
            **kwargs: Additional arguments (ignored for warmup, passed to BALD)

        Returns:
            test_point: (n_joints,) coordinate tensor
            score: Acquisition score (0 for warmup, BALD score otherwise)
        """
        # Phase 1: Prior boundary warmup
        if self.bald is None or self.query_count < self.n_warmup:
            if self.warmup_points is None:
                self._compute_warmup_points(bounds)
            if self.query_count < len(self.warmup_points):
                point = self.warmup_points[self.query_count]
                self.query_count += 1
                return point, 0.0

        # Phase 2: BALD
        self.query_count += 1
        return self.bald.select_test(
            bounds, test_history=test_history, verbose=verbose, **kwargs
        )

    def reset(self):
        """Reset query counter for a new trial."""
        self.query_count = 0

    @property
    def in_exploration_phase(self) -> bool:
        """True if still in prior boundary warmup phase."""
        if self.bald is None:
            return True
        return self.query_count < self.n_warmup

    @property
    def phase_name(self) -> str:
        """Current acquisition phase name."""
        return "Prior Boundary" if self.in_exploration_phase else "BALD"
