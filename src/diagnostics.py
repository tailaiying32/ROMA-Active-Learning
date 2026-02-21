"""
Diagnostics module for active learning debugging.

Tracks and analyzes:
1. Prior coverage: P(true_θ ∈ prior/posterior CI)
2. Query informativeness: Distance from query to true boundary
3. Gradient magnitude: ‖∇ELBO‖ during VI
4. Posterior movement: ‖θ_post - θ_prior‖ per iteration
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DiagnosticSnapshot:
    """Single iteration diagnostic snapshot."""
    iteration: int

    # Prior/Posterior coverage
    prior_coverage: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    posterior_coverage: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    # Latent Coverage (z-space)
    prior_z_coverage: float = 0.0
    posterior_z_coverage: float = 0.0

    # Query informativeness
    query_distance_to_boundary: float = 0.0
    query_is_near_boundary: bool = False

    # Gradient info (from VI)
    grad_norm: float = 0.0
    grad_norms_per_param: Dict[str, float] = field(default_factory=dict)

    # Posterior movement
    posterior_movement: float = 0.0
    posterior_movement_per_joint: Dict[str, float] = field(default_factory=dict)
    
    # Detailed Posterior Statistics
    posterior_stds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    latent_std_mean: float = 0.0

    # VI Convergence Stats
    vi_converged: bool = False
    vi_iterations: int = 0
    elbo_history: List[float] = field(default_factory=list)
    likelihood: float = 0.0
    kl_divergence: float = 0.0

    # BALD Optimization Stats
    bald_opt_stats: List[Dict[str, float]] = field(default_factory=list)


class Diagnostics:
    """
    Comprehensive diagnostics for active learning debugging.
    """

    def __init__(self, joint_names: List[str] = None, true_limits: Dict = None, true_z: torch.Tensor = None):
        """
        Args:
            joint_names: List of joint names (for legacy)
            true_limits: Dictionary of true joint limits (for legacy)
            true_z: True latent code (for latent pipeline)
        """
        self.joint_names = joint_names if joint_names else []
        self.true_limits = true_limits
        self.true_z = true_z
        self.history: List[DiagnosticSnapshot] = []

        # Store initial prior params for movement tracking
        self.initial_params: Optional[Dict] = None
        self.previous_params: Optional[Dict] = None

    def _extract_params(self, distribution) -> Dict:
        """Extract mean parameters for movement tracking."""
        # Check for Latent Distribution
        if hasattr(distribution, 'mean') and isinstance(distribution.mean, torch.Tensor):
            return {'z_mean': distribution.mean.detach().cpu().numpy().copy()}

        """Extract parameter values from a UserDistribution."""
        params = {}
        for joint in self.joint_names:
            p = distribution.params['joint_limits'][joint]

            # Extract lower params
            lower_mean = p['lower_mean'].detach().cpu().item()
            lower_std = torch.exp(p['lower_log_std']).detach().cpu().item()

            # Extract upper params
            upper_mean = p['upper_mean'].detach().cpu().item()
            upper_std = torch.exp(p['upper_log_std']).detach().cpu().item()

            # Calculate center/width stats for legacy compatibility if needed
            center_mean = (lower_mean + upper_mean) / 2
            half_width_mean = (upper_mean - lower_mean) / 2

            # Assuming independence for stds
            center_std = 0.5 * np.sqrt(lower_std**2 + upper_std**2)
            half_width_std = 0.5 * np.sqrt(lower_std**2 + upper_std**2)

            params[joint] = {
                'center_mean': center_mean,
                'center_std': center_std,
                'half_width_mean': half_width_mean,
                'half_width_std': half_width_std,
                'lower_mean': lower_mean,
                'lower_std': lower_std,
                'upper_mean': upper_mean,
                'upper_std': upper_std
            }
        return params

        return coverage

    def compute_coverage(self, distribution, n_sigma=2.0):
        """Check if ground truth is within n_sigma of the posterior mean."""
        # Latent Case
        if self.true_z is not None and hasattr(distribution, 'mean'):
            z_mean = distribution.mean.detach().cpu()
            z_std = torch.exp(distribution.log_std).detach().cpu()
            true_z = self.true_z.detach().cpu()

            covered = torch.abs(true_z - z_mean) <= n_sigma * z_std
            percent_covered = covered.float().mean().item()
            return percent_covered

        # Legacy Case
        if not self.joint_names or not self.true_limits:
             return {}

        params = self._extract_params(distribution)
        coverage = {}

        for joint in self.joint_names:
            true_lower, true_upper = self.true_limits[joint]
            p = params[joint]

            # Check if true value is within n_sigma of the mean
            lower_covered = abs(true_lower - p['lower_mean']) <= n_sigma * p['lower_std']
            upper_covered = abs(true_upper - p['upper_mean']) <= n_sigma * p['upper_std']

            coverage[joint] = {
                'lower': lower_covered,
                'upper': upper_covered
            }

        return coverage

    def compute_query_boundary_distance(
        self,
        query: torch.Tensor,
        true_checker
    ) -> Tuple[float, bool]:
        """
        Compute distance from query to true feasibility boundary.

        A query is "near boundary" if |h_value| is small.

        Returns:
            (distance_to_boundary, is_near_boundary)
        """
        if isinstance(query, torch.Tensor):
            query_np = query.detach().cpu().numpy()
        else:
            query_np = np.array(query)

        h_value = true_checker.h_value(query_np)

        if isinstance(h_value, torch.Tensor):
            h_value = h_value.item()
        elif isinstance(h_value, np.ndarray):
            h_value = h_value.item() if h_value.ndim == 0 else h_value[0]

        distance = abs(h_value)
        # Consider "near boundary" if within 0.1 radians (~6 degrees)
        is_near = distance < 0.1

        return distance, is_near

    def compute_posterior_movement(
        self,
        posterior,
        from_initial: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute how much the posterior has moved from prior/previous iteration.

        Args:
            posterior: Current posterior distribution
            from_initial: If True, compare to initial prior. If False, compare to previous iteration.

        Returns:
            (total_movement, {joint: movement})
        """
        current_params = self._extract_params(posterior)

        if from_initial:
            reference = self.initial_params
        else:
            reference = self.previous_params

        if reference is None:
             return 0.0, {}

        if 'z_mean' in current_params:
            curr = current_params['z_mean']
            prev = reference['z_mean']
            dist = float(np.linalg.norm(curr - prev))
            return dist, {}

        total_sq_movement = 0.0
        per_joint = {}

        for joint in self.joint_names:
            curr = current_params[joint]
            ref = reference[joint]

            # Movement in lower and upper means
            d_lower = (curr['lower_mean'] - ref['lower_mean'])**2
            d_upper = (curr['upper_mean'] - ref['upper_mean'])**2

            joint_movement = np.sqrt(d_lower + d_upper)
            per_joint[joint] = joint_movement
            total_sq_movement += d_lower + d_upper

        return np.sqrt(total_sq_movement), per_joint

    def log_iteration(
        self,
        iteration: int,
        prior,
        posterior,
        query: torch.Tensor,
        true_checker,
        grad_norm: float = 0.0,
        grad_norms_per_param: Dict[str, float] = None,
        # New args
        vi_result=None,
        bald_opt_stats: List[Dict] = None,
        likelihood: float = None,
        kl_divergence: float = None
    ) -> DiagnosticSnapshot:
        """
        Log diagnostics for a single iteration.
        """
        # Initialize reference params on first call
        if self.initial_params is None:
            self.initial_params = self._extract_params(prior)
            self.previous_params = self._extract_params(prior)

        # Compute all diagnostics
        coverage = self.compute_coverage(prior)
        post_coverage = self.compute_coverage(posterior)
        
        # Handle latent vs legacy return types
        prior_cov_legacy = coverage if isinstance(coverage, dict) else {}
        post_cov_legacy = post_coverage if isinstance(post_coverage, dict) else {}
        prior_cov_z = coverage if isinstance(coverage, float) else 0.0
        post_cov_z = post_coverage if isinstance(post_coverage, float) else 0.0

        query_dist, query_near = self.compute_query_boundary_distance(query, true_checker)

        movement_total, movement_per_joint = self.compute_posterior_movement(posterior, from_initial=False)

        snapshot = DiagnosticSnapshot(
            iteration=iteration,
            prior_coverage=prior_cov_legacy,
            posterior_coverage=post_cov_legacy,
            prior_z_coverage=prior_cov_z,
            posterior_z_coverage=post_cov_z,
            query_distance_to_boundary=query_dist,
            query_is_near_boundary=query_near,
            grad_norm=grad_norm,
            grad_norms_per_param=grad_norms_per_param or {},
            posterior_movement=movement_total,
            posterior_movement_per_joint=movement_per_joint,
            # New fields
            posterior_stds=self._extract_stds(posterior) if isinstance(self._extract_stds(posterior), dict) else {},
            latent_std_mean=self._extract_stds(posterior) if isinstance(self._extract_stds(posterior), float) else 0.0,
            vi_converged=vi_result.converged if vi_result else False,
            vi_iterations=vi_result.n_iterations if vi_result else 0,
            elbo_history=vi_result.elbo_history if vi_result else [],
            likelihood=likelihood if likelihood is not None else 0.0,
            kl_divergence=kl_divergence if kl_divergence is not None else 0.0,
            bald_opt_stats=bald_opt_stats if bald_opt_stats else []
        )


        self.history.append(snapshot)

        # Update previous params for next iteration
        self.previous_params = self._extract_params(posterior)

        return snapshot

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all iterations."""
        if not self.history:
            return {}

        n_iters = len(self.history)

        # Coverage over time
        prior_coverage_rate = {
            'lower': [],
            'upper': []
        }
        posterior_coverage_rate = {
            'lower': [],
            'upper': []
        }

        for snap in self.history:
            # Average coverage across joints
            prior_lower = np.mean([snap.prior_coverage[j]['lower'] for j in self.joint_names])
            prior_upper = np.mean([snap.prior_coverage[j]['upper'] for j in self.joint_names])
            post_lower = np.mean([snap.posterior_coverage[j]['lower'] for j in self.joint_names])
            post_upper = np.mean([snap.posterior_coverage[j]['upper'] for j in self.joint_names])

            prior_coverage_rate['lower'].append(prior_lower)
            prior_coverage_rate['upper'].append(prior_upper)
            posterior_coverage_rate['lower'].append(post_lower)
            posterior_coverage_rate['upper'].append(post_upper)

        return {
            'n_iterations': n_iters,
            'prior_coverage_rate': prior_coverage_rate,
            'posterior_coverage_rate': posterior_coverage_rate,
            'query_distances': [s.query_distance_to_boundary for s in self.history],
            'queries_near_boundary': [s.query_is_near_boundary for s in self.history],
            'grad_norms': [s.grad_norm for s in self.history],
            'posterior_movements': [s.posterior_movement for s in self.history],
            'final_posterior_coverage': self.history[-1].posterior_coverage if self.history else {}
        }

    def print_iteration_summary(self, snapshot: DiagnosticSnapshot):
        """Print a concise summary of diagnostics for one iteration."""
        # Coverage summary
        post_cov = snapshot.posterior_coverage
        n_covered = sum(
            1 for j in self.joint_names
            for bound in ['lower', 'upper']
            if post_cov[j][bound]
        )
        total_bounds = len(self.joint_names) * 2

        # Query info
        query_status = "NEAR" if snapshot.query_is_near_boundary else "far"

        # print(f"  [Diag] Coverage: {n_covered}/{total_bounds} | "
        #       f"Query->Boundary: {snapshot.query_distance_to_boundary:.3f} ({query_status}) | "
        #       f"∇ELBO: {snapshot.grad_norm:.4f} | "
        #       f"Δθ: {snapshot.posterior_movement:.4f}")

    def _extract_stds(self, distribution):
        """Extract just the standard deviations for easy logging."""
        # Latent Case
        if hasattr(distribution, 'log_std'):
             return torch.exp(distribution.log_std).mean().item()
             
        stds = {}
        for joint in self.joint_names:
            p = distribution.params['joint_limits'][joint]
            stds[joint] = {
                'lower': torch.exp(p['lower_log_std']).detach().cpu().item(),
                'upper': torch.exp(p['upper_log_std']).detach().cpu().item()
            }
        return stds

    def print_final_report(self):
        """Print a comprehensive final diagnostic report."""
        stats = self.get_summary_stats()

        print("\n" + "="*60)
        print("DIAGNOSTIC REPORT")
        print("="*60)

        # 1. Coverage Analysis
        print("\n1. COVERAGE ANALYSIS (Is truth within 2-sigma of posterior?)")
        print("-" * 40)
        final_cov = stats.get('final_posterior_coverage', {})
        for joint in self.joint_names:
            if joint in final_cov:
                l_status = "Y" if final_cov[joint]['lower'] else "N"
                u_status = "Y" if final_cov[joint]['upper'] else "N"
                print(f"  {joint}: lower={l_status}, upper={u_status}")

        # Overall coverage trajectory
        post_cov = stats.get('posterior_coverage_rate', {})
        if post_cov:
            initial_cov = (post_cov['lower'][0] + post_cov['upper'][0]) / 2
            final_cov_rate = (post_cov['lower'][-1] + post_cov['upper'][-1]) / 2
            print(f"\n  Coverage: {initial_cov:.1%} (initial) -> {final_cov_rate:.1%} (final)")

        # 2. Query Informativeness
        print("\n2. QUERY INFORMATIVENESS")
        print("-" * 40)
        distances = stats.get('query_distances', [])
        near_boundary = stats.get('queries_near_boundary', [])
        if distances:
            print(f"  Mean distance to boundary: {np.mean(distances):.3f}")
            print(f"  Queries near boundary: {sum(near_boundary)}/{len(near_boundary)} ({100*np.mean(near_boundary):.1f}%)")
            print(f"  Distance trend: {distances[0]:.3f} (first) -> {distances[-1]:.3f} (last)")

        # 3. Gradient Health
        print("\n3. GRADIENT HEALTH")
        print("-" * 40)
        grad_norms = stats.get('grad_norms', [])
        if grad_norms and any(g > 0 for g in grad_norms):
            nonzero_grads = [g for g in grad_norms if g > 0]
            print(f"  Mean ‖∇ELBO‖: {np.mean(nonzero_grads):.4f}")
            print(f"  Min/Max: {np.min(nonzero_grads):.4f} / {np.max(nonzero_grads):.4f}")
            if np.mean(nonzero_grads) < 0.01:
                print("  ⚠ WARNING: Gradients are very small - consider increasing tau")
        else:
            print("  (No gradient data logged)")

        # 4. Posterior Movement
        print("\n4. POSTERIOR MOVEMENT")
        print("-" * 40)
        movements = stats.get('posterior_movements', [])
        if movements:
            total_movement = sum(movements)
            print(f"  Total movement: {total_movement:.4f} rad")
            print(f"  Mean per iteration: {np.mean(movements):.4f} rad")
            if total_movement < 0.1:
                print("  ⚠ WARNING: Posterior barely moved - check prior/likelihood settings")

        # 5. Recommendations
        print("\n5. RECOMMENDATIONS")
        print("-" * 40)
        recommendations = []

        if distances and np.mean(distances) > 0.3:
            recommendations.append("• Queries not reaching boundaries - consider increasing tau for broader gradients")

        if grad_norms and np.mean([g for g in grad_norms if g > 0] or [0]) < 0.01:
            recommendations.append("• Vanishing gradients - increase tau or check likelihood model")

        if movements and sum(movements) < 0.1:
            recommendations.append("• Posterior stuck - try: increase init_std, decrease kl_weight, or increase VI iterations")

        if post_cov and (post_cov['lower'][-1] + post_cov['upper'][-1]) / 2 < 0.5:
            recommendations.append("• Poor coverage - prior may be too narrow (increase init_std)")

        if not recommendations:
            recommendations.append("• No major issues detected")

        for rec in recommendations:
            print(f"  {rec}")

        print("\n" + "="*60)
