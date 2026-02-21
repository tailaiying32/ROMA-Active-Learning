"""
Projected Stein Variational Newton (pSVN) Variational Inference.

Wraps the ProjectedSVN optimizer with the standard VI interface.
Inherits likelihood/prior computation from SVGDVariationalInference.
"""

from active_learning.src.svgd.svgd_vi import SVGDVariationalInference, SVGDVIResult, _reset_csvs
from active_learning.src.svgd.svgd_vi import _write_outer_row, _write_inner_row
from active_learning.src.svgd.svgd_vi import _INNER_LOG_OUTER_ITERS, _INNER_LOG_INTERVAL
from active_learning.src.svgd.projected_svn_optimizer import ProjectedSVN
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.svgd.particle_user_distribution import ParticleUserDistribution
from active_learning.src.test_history import TestHistory
from active_learning.src.utils import get_adaptive_param

import torch


class ProjectedSVNVariationalInference(SVGDVariationalInference):
    """
    Projected Stein Variational Newton Inference.

    Inherits log_likelihood, log_prior, likelihood, regularizer,
    and diagnostic logging from SVGDVariationalInference.

    Overrides:
      - __init__: Read from config['projected_svn'], create ProjectedSVN optimizer
      - update_posterior: Pass prior precision to optimizer for Hessian computation
    """

    def __init__(
        self,
        decoder,
        prior: LatentUserDistribution,
        posterior: ParticleUserDistribution,
        config: dict = None
    ):
        """
        Initialize Projected SVN VI.

        Args:
            decoder: Decoder model
            prior: Prior distribution (used for log_prior and Hessian)
            posterior: Particle posterior to be updated
            config: Config dict with 'projected_svn' section
        """
        # Don't call super().__init__() - it reads config['svgd'] and creates vanilla SVGD
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior
        self.config = config or {}

        # Read from 'projected_svn' config section
        svn_config = self.config.get('projected_svn', {})

        # Use projected_svgd config as fallback for shared params
        psvgd_config = self.config.get('projected_svgd', {})

        self.max_iters = svn_config.get('max_iters', psvgd_config.get('max_iters', 50))
        self.n_particles = self.posterior.n_particles
        self.kernel_width = svn_config.get('kernel_width', psvgd_config.get('kernel_width', None))

        # Step size decay configuration (can use larger steps due to Newton preconditioning)
        step_decay_cfg = svn_config.get('step_decay', psvgd_config.get('step_decay', {}))
        self.step_decay_enabled = step_decay_cfg.get('enabled', True)
        self.step_decay_schedule = step_decay_cfg.get('schedule', 'linear')
        self.step_decay_start_lr = step_decay_cfg.get('start_lr', 0.0448)
        self.step_decay_end_lr = step_decay_cfg.get('end_lr', 0.0112)
        self.step_decay_duration = step_decay_cfg.get('duration', None)
        self.step_decay_power = step_decay_cfg.get('power', 0.5)
        self.lr = self.step_decay_start_lr

        # Gradient clipping
        vi_config = self.config.get('vi', {})
        self.grad_clip = vi_config.get('grad_clip', 1.0)

        # Likelihood temperature (tau)
        self.tau = self.config.get('bald', {}).get('tau', 1.0)
        self.tau_schedule = self.config.get('bald', {}).get('tau_schedule', None)

        # KL annealing config (shared with projected_svgd)
        kl_cfg = svn_config.get('kl_annealing', psvgd_config.get('kl_annealing', {}))
        self.kl_annealing_enabled = kl_cfg.get('enabled', False)
        self.kl_annealing_config = kl_cfg

        # Prior weight
        self.kl_weight = svn_config.get('prior_weight', psvgd_config.get('prior_weight', 0.003))

        # pSVN-specific parameters
        n_eigenvectors = svn_config.get('n_eigenvectors', 10)
        n_random_slices = svn_config.get('n_random_slices', 10)
        kernel_type = svn_config.get('kernel_type', psvgd_config.get('kernel_type', 'imq'))
        use_gauss_newton = svn_config.get('use_gauss_newton', True)
        hessian_update_freq = svn_config.get('hessian_update_freq', 5)
        regularization = float(svn_config.get('regularization', 1e-4))
        include_prior_hessian = svn_config.get('include_prior_hessian', True)

        # Create ProjectedSVN optimizer
        self.optimizer = ProjectedSVN(
            n_eigenvectors=n_eigenvectors,
            n_random_slices=n_random_slices,
            kernel_width=self.kernel_width,
            kernel_type=kernel_type,
            use_gauss_newton=use_gauss_newton,
            hessian_update_freq=hessian_update_freq,
            regularization=regularization,
            include_prior_hessian=include_prior_hessian,
        )

        # Repulsive scaling not used in pSVN (handled by Newton preconditioning)
        self.repulsive_scaling = 0.0

        # Reset diagnostic CSVs
        _reset_csvs()
        self._first_particles = None

    def _get_prior_precision(self) -> torch.Tensor:
        """
        Get the prior precision matrix for Hessian computation.

        Returns:
            precision: (D, D) precision matrix
        """
        if getattr(self.prior, 'precision_matrix', None) is not None:
            return self.prior.precision_matrix
        else:
            # Diagonal prior: precision = 1/σ²
            std = torch.exp(self.prior.log_std)
            return torch.diag(1.0 / (std ** 2))

    def _get_kl_weight_for_iter(self, iteration: int, n_data: int) -> float:
        """Get KL weight for current iteration, applying annealing if enabled."""
        if self.kl_annealing_enabled and self.kl_annealing_config:
            return get_adaptive_param(self.kl_annealing_config, n_data, self.kl_weight)
        return self.kl_weight

    def update_posterior(
        self,
        test_history: TestHistory,
        kl_weight: float = None,
        diagnostics=None,
        iteration: int = None
    ) -> SVGDVIResult:
        """
        Update the particle posterior using Projected SVN.

        Args:
            test_history: History of test results
            kl_weight: Optional override for KL weight
            diagnostics: Diagnostics object (optional)
            iteration: Outer iteration index (for logging)

        Returns:
            SVGDVIResult with convergence info
        """
        if iteration == 0:
            _reset_csvs()

        # Reset optimizer cache at start of each outer iteration
        self.optimizer.reset_cache()

        particles = self.posterior.get_particles()
        particles_before = particles.detach().clone()

        if self._first_particles is None:
            self._first_particles = particles.detach().clone()

        grad_norm_history = []

        # Resolve KL weight
        n_data = len(test_history.get_all())
        kw = kl_weight if kl_weight is not None else self._get_kl_weight_for_iter(iteration, n_data)

        current_tau = self._get_current_tau(iteration)
        log_inner = (iteration is not None and
                     (_INNER_LOG_OUTER_ITERS is None or iteration in _INNER_LOG_OUTER_ITERS))

        # Get prior precision for Hessian computation
        prior_precision = self._get_prior_precision()

        # Pre-update stats
        with torch.no_grad():
            ll_before = self.log_likelihood(test_history, particles, iteration=iteration)
            lp_before = self.log_prior(particles)

        n_clipped = 0
        total_inner = 0

        for i in range(self.max_iters):
            # 1. Compute gradients
            p_in = particles.detach().requires_grad_(True)

            ll = self.log_likelihood(test_history, p_in, iteration=iteration)
            lp = self.log_prior(p_in)

            ll_grad = torch.autograd.grad(ll.sum(), p_in, retain_graph=True)[0]
            lp_grad = torch.autograd.grad(lp.sum(), p_in)[0]
            log_prob_grad = ll_grad + kw * lp_grad  # (K, D)

            score_grad_per_particle = log_prob_grad.norm(dim=-1)

            # 2. Projected SVN step
            want_diag = log_inner and (i % _INNER_LOG_INTERVAL == 0 or i == self.max_iters - 1)

            if want_diag:
                phi, svn_diag = self.optimizer.step(
                    particles,
                    log_prob_grad,
                    prior_precision=prior_precision,
                    kl_weight=kw,
                    iteration=i,
                    return_diagnostics=True
                )
            else:
                phi = self.optimizer.step(
                    particles,
                    log_prob_grad,
                    prior_precision=prior_precision,
                    kl_weight=kw,
                    iteration=i,
                    return_diagnostics=False
                )

            # 3. Gradient clipping
            phi_pre_clip_norms = phi.norm(p=2, dim=-1)
            if self.grad_clip is not None and self.grad_clip > 0:
                phi_norm = phi_pre_clip_norms.unsqueeze(-1)
                clip_coef = torch.clamp(self.grad_clip / (phi_norm + 1e-6), max=1.0)
                phi = phi * clip_coef
                n_clipped += int((phi_pre_clip_norms > self.grad_clip).sum().item())
                total_inner += phi_pre_clip_norms.shape[0]

            # 4. Compute step size
            if self.step_decay_enabled:
                if self.step_decay_schedule in ('linear', 'cosine', 'exponential'):
                    schedule = {
                        'start': self.step_decay_start_lr,
                        'end': self.step_decay_end_lr,
                        'duration': self.step_decay_duration or self.max_iters,
                        'schedule': self.step_decay_schedule
                    }
                    effective_lr = get_adaptive_param(schedule, i, self.lr)
                else:
                    effective_lr = self.lr / max(n_data ** self.step_decay_power, 1.0)
            else:
                effective_lr = self.lr

            # 5. Update particles
            with torch.no_grad():
                particles += effective_lr * phi

            grad_norm = phi.norm().item()
            grad_norm_history.append(grad_norm)

            # Inner-loop logging
            if want_diag:
                _write_inner_row({
                    'outer_iter': iteration,
                    'inner_iter': i,
                    'effective_lr': effective_lr,
                    'phi_norm': grad_norm,
                    'phi_per_particle_mean': phi.norm(dim=-1).mean().item(),
                    'phi_per_particle_max': phi.norm(dim=-1).max().item(),
                    'score_grad_mean': score_grad_per_particle.mean().item(),
                    'score_grad_max': score_grad_per_particle.max().item(),
                    'term1_norm': svn_diag['term1_norm'],
                    'term2_norm': svn_diag['term2_norm'],
                    'term1_1d_mean': svn_diag.get('term1_proj_norm', 0.0),
                    'term2_1d_mean': svn_diag.get('term2_proj_norm', 0.0),
                    'term1_per_particle': svn_diag['term1_per_particle'],
                    'term2_per_particle': svn_diag['term2_per_particle'],
                    'term1_term2_ratio': 0.0,
                    'h': svn_diag['h'],
                    'mean_pairwise_dist': svn_diag['mean_pairwise_dist'],
                    'min_pairwise_dist': svn_diag['min_pairwise_dist_offdiag'],
                    'kernel_mean': svn_diag['kernel_mean'],
                    'clip_frac': float((phi_pre_clip_norms > self.grad_clip).sum().item()) / phi_pre_clip_norms.shape[0] if self.grad_clip else 0.0,
                    'll_mean': ll.mean().item(),
                    'll_std': ll.std().item(),
                    'lp_mean': lp.mean().item(),
                    'lp_std': lp.std().item(),
                    # pSVN-specific
                    'n_projected_dirs': svn_diag.get('n_projected_dirs', 0),
                    'variance_explained': 0.0,  # Not computed for pSVN
                    'eigenvalue_max': svn_diag.get('eigenvalue_max', 0.0),
                    'eigenvalue_min': svn_diag.get('eigenvalue_min', 0.0),
                    'eigenvalue_condition': svn_diag.get('eigenvalue_condition', 0.0),
                    'hessian_updated': svn_diag.get('hessian_updated', False),
                })

        # Post-update outer-iteration logging
        with torch.no_grad():
            ll_after = self.log_likelihood(test_history, particles, iteration=iteration)
            lp_after = self.log_prior(particles)

            particle_mean = particles.mean(dim=0)
            particle_std = particles.std(dim=0)

            K = particles.shape[0]
            diffs = particles.unsqueeze(1) - particles.unsqueeze(0)
            pw_dist = diffs.norm(dim=-1)
            mask = ~torch.eye(K, dtype=bool, device=particles.device)
            pw_offdiag = pw_dist[mask]

            movement = (particles - particles_before).norm(dim=-1)
            prior_dist = (particles - self.prior.mean.unsqueeze(0)).norm(dim=-1)

        clip_frac = n_clipped / max(total_inner, 1)

        _write_outer_row({
            'iteration': iteration,
            'n_data': n_data,
            'effective_lr': effective_lr,
            'tau': current_tau,
            'kl_weight': kw,
            'particle_std_mean': particle_std.mean().item(),
            'particle_std_min': particle_std.min().item(),
            'particle_std_max': particle_std.max().item(),
            'pw_dist_mean': pw_offdiag.mean().item(),
            'pw_dist_min': pw_offdiag.min().item(),
            'pw_dist_max': pw_offdiag.max().item(),
            'pw_dist_median': pw_offdiag.median().item(),
            'movement_mean': movement.mean().item(),
            'movement_min': movement.min().item(),
            'movement_max': movement.max().item(),
            'movement_std': movement.std().item(),
            'prior_dist_mean': prior_dist.mean().item(),
            'prior_dist_min': prior_dist.min().item(),
            'prior_dist_max': prior_dist.max().item(),
            'll_before_mean': ll_before.mean().item(),
            'll_before_std': ll_before.std().item(),
            'll_before_min': ll_before.min().item(),
            'll_before_max': ll_before.max().item(),
            'll_after_mean': ll_after.mean().item(),
            'll_after_std': ll_after.std().item(),
            'll_after_min': ll_after.min().item(),
            'll_after_max': ll_after.max().item(),
            'll_per_datapoint': ll_after.mean().item() / max(n_data, 1),
            'lp_before_mean': lp_before.mean().item(),
            'lp_after_mean': lp_after.mean().item(),
            'phi_norm_first': grad_norm_history[0] if grad_norm_history else 0,
            'phi_norm_last': grad_norm_history[-1] if grad_norm_history else 0,
            'phi_norm_max': max(grad_norm_history) if grad_norm_history else 0,
            'clip_fraction': clip_frac,
        })

        return SVGDVIResult(
            converged=True,
            n_iterations=self.max_iters,
            final_grad_norm=grad_norm_history[-1] if grad_norm_history else 0.0,
            grad_norm_history=grad_norm_history,
            kl_weight=kw,
            kl_divergence=0.0
        )
