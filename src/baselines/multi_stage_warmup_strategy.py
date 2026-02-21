"""
Multi-Stage Warmup acquisition strategy.

Iteratively queries the CURRENT posterior's decision boundary, not just the prior's.
Each stage:
1. Samples candidates uniformly in anatomical bounds
2. Evaluates p(feasible) by averaging over posterior samples
3. Selects points where p_mean ≈ 0.5 (decision boundary)
4. Uses farthest-point sampling for spatial diversity

After warmup stages complete (or adaptive stopping triggered), switches to BALD.

Adaptive stopping: monitors rolling window entropy of query outcomes.
When entropy >= threshold (~50% feasible), the posterior boundary is approximately
correct, and we switch to BALD for refinement.
"""

import math
import torch
from typing import Tuple, Optional, List

from active_learning.src.config import DEVICE
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker


class MultiStageWarmupStrategy:
    """
    Multi-Stage Warmup: iteratively query posterior boundary, then optionally BALD.

    Unlike PriorBoundaryStrategy (which queries prior boundary once then BALD),
    this strategy queries the CURRENT posterior's boundary at each stage,
    allowing the posterior to refine its boundary estimate iteratively.

    Supports adaptive stopping based on rolling window entropy of outcomes.
    """

    def __init__(
        self,
        decoder,
        posterior,
        bald_strategy=None,
        n_stages: int = 5,
        queries_per_stage: int = 5,
        n_candidates: int = 5000,
        boundary_percentile: float = 0.05,
        use_farthest_point: bool = True,
        final_phase: str = 'bald',
        n_mc_samples: int = 50,
        adaptive_stopping: bool = True,
        entropy_threshold: float = 0.9,
        window_size: int = 15,
        min_warmup_queries: int = 10,
        device: str = DEVICE
    ):
        """
        Initialize multi-stage warmup acquisition.

        Args:
            decoder: LevelSetDecoder model
            posterior: Posterior distribution (ParticleUserDistribution for SVGD, or any with sample())
            bald_strategy: Optional BALD strategy to use after warmup
            n_stages: Number of boundary-targeting stages (acts as MAX if adaptive_stopping=True)
            queries_per_stage: Number of queries per stage
            n_candidates: Candidate pool size for boundary search
            boundary_percentile: Fraction of candidates closest to p=0.5 to consider
            use_farthest_point: Whether to use farthest-point sampling for diversity
            final_phase: What to do after warmup: 'bald', 'posterior_boundary', or 'stop'
            n_mc_samples: Number of MC samples for non-particle posteriors
            adaptive_stopping: Whether to use entropy-based adaptive stopping
            entropy_threshold: Entropy threshold to trigger switch to BALD (0.9 ≈ ratio in [0.35, 0.65])
            window_size: Rolling window size for entropy computation
            min_warmup_queries: Minimum queries before checking entropy
            device: Torch device
        """
        self.decoder = decoder
        self.posterior = posterior
        self.bald_strategy = bald_strategy
        self.n_stages = n_stages
        self.queries_per_stage = queries_per_stage
        self.n_candidates = n_candidates
        self.boundary_percentile = boundary_percentile
        self.use_farthest_point = use_farthest_point
        self.final_phase = final_phase
        self.n_mc_samples = n_mc_samples
        self.device = device

        # Adaptive stopping params
        self.adaptive_stopping = adaptive_stopping
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.min_warmup_queries = min_warmup_queries

        # State
        self.query_count = 0
        self.current_stage = 0
        self.stage_points = None
        self.stage_points_idx = 0

        # Adaptive stopping state
        self.outcome_history = []  # Track all warmup outcomes (updated via post_query_update)
        self.switched_to_final_phase = False

    def _get_posterior_samples(self) -> torch.Tensor:
        """
        Get samples from the posterior.
        For SVGD, returns all particles. For VI, samples n_mc_samples.

        Returns:
            z_samples: (n_samples, latent_dim) tensor on device
        """
        if hasattr(self.posterior, 'get_particles'):
            # SVGD: use all particles directly (no sampling overhead)
            return self.posterior.get_particles()
        else:
            # VI/other: sample
            return self.posterior.sample(self.n_mc_samples)

    def _compute_boundary_points(
        self,
        bounds: torch.Tensor,
        n_points: int,
        use_prior_mean: bool = False
    ) -> torch.Tensor:
        """
        Find points on the decision boundary.

        PERFORMANCE: All operations are batched on GPU.
        - Candidate generation: single vectorized op
        - Feasibility evaluation: batched decode + batched evaluate
        - Filtering: vectorized masking
        - Farthest-point: torch.cdist (GPU-accelerated)

        Args:
            bounds: (n_joints, 2) tensor of [min, max] bounds
            n_points: Number of boundary points to select
            use_prior_mean: If True, use prior mean only (stage 1). If False, use posterior samples.

        Returns:
            Selected boundary points: (n_points, n_joints)
        """
        n_dims = bounds.shape[0]
        bounds = bounds.to(self.device)

        # 1. Generate uniform candidates - VECTORIZED
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        candidates = lower + torch.rand(
            self.n_candidates, n_dims, device=self.device
        ) * (upper - lower)

        # 2. Get latent samples and evaluate feasibility
        with torch.no_grad():
            if use_prior_mean:
                # Stage 1: Use posterior mean only (like prior boundary)
                z_mean = self.posterior.mean.unsqueeze(0)  # (1, latent_dim)
                logits = LatentFeasibilityChecker.batched_logit_values(
                    self.decoder, z_mean, candidates
                )  # (1, n_candidates)
                p_mean = torch.sigmoid(logits).squeeze(0)  # (n_candidates,)
            else:
                # Stage 2+: Use all posterior samples
                z_samples = self._get_posterior_samples()  # (n_samples, latent_dim)

                # Decode all samples to RBF params ONCE
                decoded_params = LatentFeasibilityChecker.decode_latent_params(
                    self.decoder, z_samples
                )

                # Evaluate ALL candidates against ALL samples - BATCHED
                # logits: (n_samples, n_candidates)
                logits = LatentFeasibilityChecker.evaluate_from_decoded(
                    candidates, decoded_params
                )

                # Aggregate: mean probability across samples
                probs = torch.sigmoid(logits)  # (n_samples, n_candidates)
                p_mean = probs.mean(dim=0)     # (n_candidates,)

        # 3. Filter to boundary region - VECTORIZED
        boundary_dist = (p_mean - 0.5).abs()  # (n_candidates,)

        # Keep candidates closest to p=0.5
        n_keep = max(int(self.n_candidates * self.boundary_percentile), n_points * 3)
        n_keep = min(n_keep, len(candidates))

        # Use topk for efficiency (finds smallest k distances)
        _, top_indices = boundary_dist.topk(n_keep, largest=False)
        boundary_candidates = candidates[top_indices]
        boundary_dists = boundary_dist[top_indices]
        boundary_p_means = p_mean[top_indices]

        # Log boundary statistics
        p_min, p_max = boundary_p_means.min().item(), boundary_p_means.max().item()
        dist_threshold = boundary_dists.max().item()
        print(f"  [Boundary] Found {n_keep} candidates "
              f"(|p-0.5| <= {dist_threshold:.4f}, p range: [{p_min:.3f}, {p_max:.3f}])")

        # 4. Farthest-point sampling for spatial diversity - GPU
        if self.use_farthest_point and len(boundary_candidates) > n_points:
            selected_indices = self._farthest_point_sample_gpu(
                boundary_candidates, n_points, boundary_dists
            )
            return boundary_candidates[selected_indices]

        return boundary_candidates[:n_points]

    def _farthest_point_sample_gpu(
        self,
        candidates: torch.Tensor,
        n_select: int,
        boundary_dists: torch.Tensor
    ) -> List[int]:
        """
        Farthest-point sampling entirely on GPU.

        Uses torch.cdist for CUDA-accelerated pairwise distance computation.
        Starts from point closest to p=0.5, then greedily adds farthest points.

        Args:
            candidates: (N, D) candidate points on GPU
            n_select: Number of points to select
            boundary_dists: (N,) distance to p=0.5 for each candidate

        Returns:
            List of selected indices
        """
        N = len(candidates)
        device = candidates.device

        # Track selected points
        selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
        selected_indices = []

        # Start with point closest to p=0.5
        first_idx = boundary_dists.argmin().item()
        selected_indices.append(first_idx)
        selected_mask[first_idx] = True

        # Greedily add farthest points
        for _ in range(n_select - 1):
            # Get currently selected points
            selected_pts = candidates[selected_mask]  # (k, D)

            # Compute distances from all candidates to selected points
            # torch.cdist is CUDA-optimized
            dists = torch.cdist(candidates, selected_pts)  # (N, k)

            # Min distance to any selected point
            min_dists = dists.min(dim=1).values  # (N,)

            # Mask out already selected
            min_dists[selected_mask] = -float('inf')

            # Select farthest
            next_idx = min_dists.argmax().item()
            selected_indices.append(next_idx)
            selected_mask[next_idx] = True

        return selected_indices

    def _compute_stage_points(self, bounds: torch.Tensor):
        """Compute all points for the next stage."""
        self.current_stage += 1
        self.stage_points_idx = 0

        phase_name = f"Stage {self.current_stage}"
        if self.current_stage == 1:
            phase_name += " (Prior Mean)"
        else:
            phase_name += " (Posterior Boundary)"

        print(f"[MultiStageWarmup] Computing {phase_name} points...")

        # Stage 1 uses prior/posterior mean only, Stage 2+ uses full posterior
        use_prior_mean = (self.current_stage == 1)
        self.stage_points = self._compute_boundary_points(
            bounds, self.queries_per_stage, use_prior_mean=use_prior_mean
        )

    def post_query_update(self, test_point: torch.Tensor, outcome: float, history=None):
        """
        Callback after each query - used to track outcomes for adaptive stopping.

        Args:
            test_point: The queried test point
            outcome: Oracle outcome (1.0 = feasible, 0.0 = infeasible)
            history: Optional oracle history (not used)
        """
        # Track outcome for entropy computation
        self.outcome_history.append(float(outcome))

    def _compute_rolling_entropy(self) -> Tuple[float, float]:
        """
        Compute entropy of last window_size outcomes.

        Returns:
            (entropy, feasibility_ratio) tuple
        """
        if len(self.outcome_history) < self.min_warmup_queries:
            return 0.0, 0.5  # Not enough data

        # Get recent outcomes (up to window_size)
        recent = self.outcome_history[-self.window_size:]
        p = sum(recent) / len(recent)

        # Compute binary entropy
        if p == 0 or p == 1:
            entropy = 0.0
        else:
            entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)

        return entropy, p

    def _should_switch_to_final_phase(self) -> bool:
        """
        Check if entropy threshold is met for adaptive stopping.

        Returns:
            True if should switch to final phase (BALD)
        """
        if not self.adaptive_stopping:
            return False

        if len(self.outcome_history) < self.min_warmup_queries:
            return False

        entropy, ratio = self._compute_rolling_entropy()

        if entropy >= self.entropy_threshold:
            window_len = min(len(self.outcome_history), self.window_size)
            print(f"[Adaptive] Entropy threshold met! H={entropy:.3f}, "
                  f"ratio={ratio:.2f} (window={window_len})")
            return True

        return False

    def select_test(
        self,
        bounds: torch.Tensor,
        test_history: Optional[list] = None,
        verbose: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, float]:
        """
        Select next test point.

        Args:
            bounds: (n_joints, 2) tensor of [min, max] bounds
            test_history: Optional history of test results
            verbose: Print progress
            **kwargs: Additional arguments (passed to BALD in final phase)

        Returns:
            (test_point, score) tuple where score is 0 for warmup, BALD score otherwise
        """
        max_warmup = self.n_stages * self.queries_per_stage

        # 1. Check adaptive stopping criterion (if not already switched)
        #    (outcome_history is updated via post_query_update callback)
        if not self.switched_to_final_phase and self._should_switch_to_final_phase():
            self.switched_to_final_phase = True
            print(f"[Adaptive] Switching to {self.final_phase} after "
                  f"{len(self.outcome_history)} warmup queries.")

        # 2. Check if max warmup reached (safety cap)
        if not self.switched_to_final_phase and self.query_count >= max_warmup:
            self.switched_to_final_phase = True
            entropy, ratio = self._compute_rolling_entropy()
            print(f"[Adaptive] Max warmup ({max_warmup}) reached. H={entropy:.3f}, ratio={ratio:.2f}. "
                  f"Switching to {self.final_phase}.")

        # 3. If switched to final phase, use final phase strategy
        if self.switched_to_final_phase:
            if self.final_phase == 'bald' and self.bald_strategy is not None:
                self.query_count += 1
                return self.bald_strategy.select_test(
                    bounds, test_history=test_history, verbose=verbose, **kwargs
                )
            elif self.final_phase == 'posterior_boundary':
                # Continue with single posterior boundary query
                point = self._compute_boundary_points(bounds, n_points=1, use_prior_mean=False)[0]
                self.query_count += 1
                if verbose:
                    print(f"[Posterior Boundary] Query {self.query_count}")
                return point, 0.0
            else:
                # final_phase == 'stop' or bald_strategy is None
                raise StopIteration("Multi-stage warmup complete")

        # 4. Still in warmup - check if need new stage points
        if self.stage_points is None or self.stage_points_idx >= len(self.stage_points):
            self._compute_stage_points(bounds)

        # 5. Return next point from current stage
        point = self.stage_points[self.stage_points_idx]
        self.stage_points_idx += 1
        self.query_count += 1

        if verbose:
            stage_query = self.stage_points_idx
            entropy, ratio = self._compute_rolling_entropy()
            print(f"[Stage {self.current_stage}] Query {stage_query}/{self.queries_per_stage} "
                  f"(Total: {self.query_count}, H={entropy:.3f}, ratio={ratio:.2f})")

        return point, 0.0

    def reset(self):
        """Reset state for a new trial."""
        self.query_count = 0
        self.current_stage = 0
        self.stage_points = None
        self.stage_points_idx = 0

        # Reset adaptive stopping state
        self.outcome_history = []
        self.switched_to_final_phase = False

    @property
    def in_warmup_phase(self) -> bool:
        """True if still in multi-stage warmup (not switched to final phase)."""
        return not self.switched_to_final_phase

    @property
    def phase_name(self) -> str:
        """Current acquisition phase name."""
        if self.in_warmup_phase:
            if self.current_stage == 0:
                return "Multi-Stage Warmup (Starting)"
            elif self.current_stage == 1:
                return "Multi-Stage Warmup (Prior Mean)"
            else:
                return f"Multi-Stage Warmup (Stage {self.current_stage})"
        else:
            if self.final_phase == 'bald':
                return "BALD (post-warmup)"
            elif self.final_phase == 'posterior_boundary':
                return "Posterior Boundary (continuous)"
            else:
                return "Complete"

    @property
    def warmup_stats(self) -> dict:
        """Get current warmup statistics."""
        entropy, ratio = self._compute_rolling_entropy()
        return {
            'queries': len(self.outcome_history),
            'entropy': entropy,
            'feasibility_ratio': ratio,
            'switched': self.switched_to_final_phase,
            'current_stage': self.current_stage,
        }
