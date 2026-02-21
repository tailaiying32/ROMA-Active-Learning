import torch
import numpy as np
import csv
import os
from dataclasses import dataclass
from typing import Optional

from active_learning.src.test_history import TestHistory
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.svgd.svgd_optimizer import SVGD
from active_learning.src.svgd.particle_user_distribution import ParticleUserDistribution
from active_learning.src.utils import get_adaptive_param

@dataclass
class SVGDVIResult:
    """Result from SVGD posterior update."""
    converged: bool
    n_iterations: int
    final_grad_norm: float
    grad_norm_history: list
    # For compatibility with diagnostics
    final_elbo: float = 0.0 # ELBO not defined for SVGD in the same way
    elbo_history: list = None
    mean_history: Optional[np.ndarray] = None
    log_std_history: Optional[np.ndarray] = None
    kl_divergence: float = 0.0
    kl_weight: float = 0.0


# ── Diagnostic CSV logging ──────────────────────────────────────────────────

DIAG_DIR = os.path.join("active_learning", "diagnostics", "latent")
OUTER_CSV = os.path.join(DIAG_DIR, "svgd_debug.csv")
INNER_CSV = os.path.join(DIAG_DIR, "svgd_inner_debug.csv")

# Which outer iterations get full inner-loop logging
_INNER_LOG_OUTER_ITERS = None  # None = log ALL outer iterations
# Which inner iterations to sample within those outer iterations
_INNER_LOG_INTERVAL = 5  # every 5th inner step

_outer_header_written = False
_inner_header_written = False


def _ensure_dir():
    os.makedirs(DIAG_DIR, exist_ok=True)


def _write_outer_row(row: dict):
    global _outer_header_written
    _ensure_dir()
    write_header = not _outer_header_written and not os.path.exists(OUTER_CSV)
    with open(OUTER_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    _outer_header_written = True


def _write_inner_row(row: dict):
    global _inner_header_written
    _ensure_dir()
    write_header = not _inner_header_written and not os.path.exists(INNER_CSV)
    with open(INNER_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    _inner_header_written = True


def _reset_csvs():
    """Delete old CSV files at the start of a new run."""
    global _outer_header_written, _inner_header_written
    _ensure_dir()
    for path in (OUTER_CSV, INNER_CSV):
        if os.path.exists(path):
            os.remove(path)
    _outer_header_written = False
    _inner_header_written = False

# ─────────────────────────────────────────────────────────────────────────────


class SVGDVariationalInference:
    '''
    SVGD Inference class.
    Adapts the SVGD optimizer to the LatentActiveLearner interface.
    Updates the particle distribution using Stein Variational Gradient Descent.
    '''

    def __init__(
        self,
        decoder,
        prior: LatentUserDistribution,
        posterior: ParticleUserDistribution,
        config: dict = None
    ):
        '''
        Initialize SVGD VI.
        Args:
            decoder: Decoder model
            prior: Prior distribution (used for log_prior gradient)
            posterior: Particle posterior to be updated
            config: Config dict
        '''
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior
        self.config = config or {}

        # Config extraction
        svgd_config = self.config.get('svgd', {})
        self.lr = svgd_config.get('step_size', 0.1)
        self.max_iters = svgd_config.get('max_iters', 100)
        self.n_particles = self.posterior.n_particles
        self.kernel_width = svgd_config.get('kernel_width', None)

        # Gradient clipping
        vi_config = self.config.get('vi', {})
        self.grad_clip = vi_config.get('grad_clip', 1.0) # Default clip to 1.0 in latent space

        # Likelihood temperature (tau)
        self.tau = self.config.get('bald', {}).get('tau', 1.0)
        self.tau_schedule = self.config.get('bald', {}).get('tau_schedule', None)

        # Step size decay configuration
        step_decay_cfg = svgd_config.get('step_decay', {})
        self.step_decay_enabled = step_decay_cfg.get('enabled', False)
        self.step_decay_schedule = step_decay_cfg.get('schedule', 'power')  # 'linear', 'cosine', 'exponential', 'power'
        self.step_decay_start_lr = step_decay_cfg.get('start_lr', self.lr)  # Defaults to step_size for backward compat
        self.step_decay_end_lr = step_decay_cfg.get('end_lr', self.step_decay_start_lr * 0.25)
        self.step_decay_duration = step_decay_cfg.get('duration', None)  # None = use max_iters
        self.step_decay_power = step_decay_cfg.get('power', 0.5)  # Legacy: lr / n_data^power
        # If start_lr is specified in step_decay, use it as the base LR
        if 'start_lr' in step_decay_cfg:
            self.lr = self.step_decay_start_lr

        # Repulsive force scaling (D^alpha) to counteract variance collapse in high dimensions
        self.repulsive_scaling = svgd_config.get('repulsive_scaling', 0.0)

        # Optimizer
        self.optimizer = SVGD(kernel_width=self.kernel_width, repulsive_scaling=self.repulsive_scaling)

        # Prior weight: controls strength of log_prior pull on particles
        # Read from svgd.prior_weight config (default 1.0 for backward compatibility)
        self.kl_weight = svgd_config.get('prior_weight', 1.0)

        # Reset diagnostic CSVs at init (new run)
        _reset_csvs()
        self._first_particles = None  # track initial particles for total movement


    def _get_current_tau(self, iteration=None):
        """Get tau for the current iteration, applying schedule if configured."""
        if self.tau_schedule and iteration is not None:
            return get_adaptive_param(self.tau_schedule, iteration, self.tau)
        return self.tau

    def get_kl_weight(self, iteration=None):
        """For compatibility with LatentActiveLearner."""
        return self.kl_weight

    def log_likelihood(self, test_history: TestHistory, particles: torch.Tensor, iteration=None) -> torch.Tensor:
        """
        Compute log p(D|z) for specific particles.
        Sum over data points.
        Returns tensor of shape (n_particles,)
        """
        results = test_history.get_all()
        if not results:
            return torch.zeros(particles.shape[0], device=particles.device)

        # Stack test points and outcomes (binary 0.0/1.0)
        test_points = torch.stack([r.test_point for r in results]).to(particles.device)
        outcomes = torch.tensor([r.outcome for r in results], device=particles.device).unsqueeze(0) # (1, N_data)

        # Compute logits for all particles
        # particles: (K, latent_dim)
        # logits: (K, N_data)
        pred_logits = LatentFeasibilityChecker.batched_logit_values(self.decoder, particles, test_points)

        # Scaled logits for BCE
        current_tau = self._get_current_tau(iteration)
        scaled_logits = pred_logits / current_tau

        # Expand targets to (K, N_data)
        targets_expanded = outcomes.expand_as(scaled_logits)

        # BCE (negative log likelihood per point)
        # We want sum log p(y|z).
        # BCE = - [y log p + (1-y) log (1-p)] = - log p(y|z)
        # So log p(y|z) = -BCE
        neg_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            scaled_logits,
            targets_expanded,
            reduction='none'
        )

        # Sum over data points (dim 1) -> (K,)
        log_lik = -neg_bce.sum(dim=1)

        return log_lik

    def likelihood(self, test_history: TestHistory, iteration=None) -> torch.Tensor:
        """
        Compute scalar log likelihood for the current posterior (for diagnostics compatibility).
        Wraps log_likelihood using current particles.
        Returns scalar tensor (mean log-likelihood).
        """
        particles = self.posterior.get_particles()
        log_lik = self.log_likelihood(test_history, particles, iteration=iteration)
        # log_lik is (K,), representing sum log p(D|z_k).
        # We want expected log likelihood E_q [ log p(D|z) ].
        # So we mean over K.
        return log_lik.mean()

    def regularizer(self, kl_weight: float = None) -> torch.Tensor:
        """
        SVGD does not compute explicit KL divergence during optimization.
        Returns 0.0 for API compatibility with LatentActiveLearner diagnostics.
        """
        return torch.tensor(0.0, device=self.posterior.device)

    def log_prior(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z) under the Gaussian prior.
        Supports full covariance (precision matrix) or diagonal.
        Returns (K,)
        """
        mu = self.prior.mean

        if getattr(self.prior, 'precision_matrix', None) is not None:
            # Full covariance: -0.5 (z-μ)ᵀ Σ⁻¹ (z-μ)
            diff = particles - mu.unsqueeze(0)                        # (K, D)
            log_p = -0.5 * torch.sum((diff @ self.prior.precision_matrix) * diff, dim=1)
        else:
            # Diagonal: -0.5 sum((z-μ)²/σ²)
            std = torch.exp(self.prior.log_std)
            z_norm = (particles - mu.unsqueeze(0)) / std.unsqueeze(0)
            log_p = -0.5 * torch.sum(z_norm**2, dim=1)

        return log_p

    def update_posterior(self, test_history: TestHistory, kl_weight: float = None, diagnostics=None, iteration: int = None) -> SVGDVIResult:
        '''
        Update the particle posterior using SVGD.
        '''
        # Reset CSV logs at the start of a new run
        if iteration == 0:
            _reset_csvs()

        particles = self.posterior.get_particles()
        particles_before = particles.detach().clone()

        if self._first_particles is None:
            self._first_particles = particles.detach().clone()

        grad_norm_history = []

        # Resolve KL weight
        kw = kl_weight if kl_weight is not None else self.kl_weight

        n_data = len(test_history.get_all())
        current_tau = self._get_current_tau(iteration)
        log_inner = (iteration is not None and
                     (_INNER_LOG_OUTER_ITERS is None or iteration in _INNER_LOG_OUTER_ITERS))

        # Pre-update particle stats
        with torch.no_grad():
            ll_before = self.log_likelihood(test_history, particles, iteration=iteration)
            lp_before = self.log_prior(particles)

        n_clipped = 0
        total_inner = 0

        for i in range(self.max_iters):
            # 1. Detach particles and enable grad to compute d(log_prob)/d(particles)
            p_in = particles.detach().requires_grad_(True)

            # 2. Compute Log Joint
            ll = self.log_likelihood(test_history, p_in, iteration=iteration)
            lp = self.log_prior(p_in)

            # Compute gradients separately: use total (un-normalized) likelihood
            # so that more data = stronger attractive signal, enabling natural
            # Bayesian posterior concentration. Use step_decay to keep stable.
            ll_grad = torch.autograd.grad(ll.sum(), p_in, retain_graph=True)[0]
            lp_grad = torch.autograd.grad(lp.sum(), p_in)[0]
            log_prob_grad = ll_grad + kw * lp_grad  # (K, D)

            # Score gradient stats (before SVGD kernel)
            score_grad_per_particle = log_prob_grad.norm(dim=-1)  # (K,)

            # 3. SVGD Step
            # phi: (K, D)
            want_diag = log_inner and (i % _INNER_LOG_INTERVAL == 0 or i == self.max_iters - 1)
            if want_diag:
                phi, svgd_diag = self.optimizer.step(particles, log_prob_grad, return_diagnostics=True)
            else:
                phi = self.optimizer.step(particles, log_prob_grad)

            # Clip phi to prevent exploding updates at low tau
            phi_pre_clip_norms = phi.norm(p=2, dim=-1)  # (K,) per-particle norms
            if self.grad_clip is not None and self.grad_clip > 0:
                # Clip per-particle norm
                phi_norm = phi_pre_clip_norms.unsqueeze(-1)  # (K, 1)
                clip_coef = torch.clamp(self.grad_clip / (phi_norm + 1e-6), max=1.0)
                phi = phi * clip_coef
                n_clipped += int((phi_pre_clip_norms > self.grad_clip).sum().item())
                total_inner += phi_pre_clip_norms.shape[0]

            # 4. Compute per-iteration step size
            if self.step_decay_enabled:
                if self.step_decay_schedule in ('linear', 'cosine', 'exponential'):
                    # Iteration-based decay using get_adaptive_param
                    schedule = {
                        'start': self.step_decay_start_lr,
                        'end': self.step_decay_end_lr,
                        'duration': self.step_decay_duration or self.max_iters,
                        'schedule': self.step_decay_schedule
                    }
                    effective_lr = get_adaptive_param(schedule, i, self.lr)
                else:
                    # Legacy power decay: lr / n_data^power
                    effective_lr = self.lr / max(n_data ** self.step_decay_power, 1.0)
            else:
                effective_lr = self.lr

            # 5. Update
            with torch.no_grad():
                particles += effective_lr * phi

            # Diagnostics
            grad_norm = phi.norm().item()
            grad_norm_history.append(grad_norm)

            # Inner-loop CSV logging
            if want_diag:
                # For sliced/projected SVGD, use term2_1d_norm_mean (actual repulsion in 1D projections)
                # For standard SVGD, use term2_norm
                term2_effective = svgd_diag.get('term2_1d_norm_mean', svgd_diag['term2_norm'])
                term1_effective = svgd_diag.get('term1_1d_norm_mean', svgd_diag['term1_norm'])

                _write_inner_row({
                    'outer_iter': iteration,
                    'inner_iter': i,
                    'effective_lr': effective_lr,
                    'phi_norm': grad_norm,
                    'phi_per_particle_mean': phi.norm(dim=-1).mean().item(),
                    'phi_per_particle_max': phi.norm(dim=-1).max().item(),
                    'score_grad_mean': score_grad_per_particle.mean().item(),
                    'score_grad_max': score_grad_per_particle.max().item(),
                    'term1_norm': svgd_diag['term1_norm'],
                    'term2_norm': svgd_diag['term2_norm'],
                    'term1_1d_mean': term1_effective,
                    'term2_1d_mean': term2_effective,
                    'term1_per_particle': svgd_diag['term1_per_particle'],
                    'term2_per_particle': svgd_diag['term2_per_particle'],
                    'term1_term2_ratio': term1_effective / max(term2_effective, 1e-10),
                    'h': svgd_diag['h'],
                    'mean_pairwise_dist': svgd_diag['mean_pairwise_dist'],
                    'min_pairwise_dist': svgd_diag['min_pairwise_dist_offdiag'],
                    'kernel_mean': svgd_diag['kernel_mean'],
                    'clip_frac': float((phi_pre_clip_norms > self.grad_clip).sum().item()) / phi_pre_clip_norms.shape[0] if self.grad_clip else 0.0,
                    'll_mean': ll.mean().item(),
                    'll_std': ll.std().item(),
                    'lp_mean': lp.mean().item(),
                    'lp_std': lp.std().item(),
                    # Projected SVGD specific (0 for other methods)
                    'n_projected_dirs': svgd_diag.get('n_projected_dirs', 0),
                    'variance_explained': svgd_diag.get('variance_explained', 0.0),
                })

        # ── Post-update outer-iteration logging ──
        with torch.no_grad():
            ll_after = self.log_likelihood(test_history, particles, iteration=iteration)
            lp_after = self.log_prior(particles)

            # Particle health
            particle_mean = particles.mean(dim=0)    # (D,)
            particle_std = particles.std(dim=0)      # (D,)

            # Pairwise distances
            K = particles.shape[0]
            diffs = particles.unsqueeze(1) - particles.unsqueeze(0)
            pw_dist = diffs.norm(dim=-1)  # (K, K)
            mask = ~torch.eye(K, dtype=bool, device=particles.device)
            pw_offdiag = pw_dist[mask]

            # Movement from previous iteration
            movement = (particles - particles_before).norm(dim=-1)  # (K,)

            # Distance from prior mean
            prior_dist = (particles - self.prior.mean.unsqueeze(0)).norm(dim=-1)

        clip_frac = n_clipped / max(total_inner, 1)

        _write_outer_row({
            'iteration': iteration,
            'n_data': n_data,
            'effective_lr': effective_lr,
            'tau': current_tau,
            'kl_weight': kw,
            # Particle health
            'particle_std_mean': particle_std.mean().item(),
            'particle_std_min': particle_std.min().item(),
            'particle_std_max': particle_std.max().item(),
            # Pairwise distances
            'pw_dist_mean': pw_offdiag.mean().item(),
            'pw_dist_min': pw_offdiag.min().item(),
            'pw_dist_max': pw_offdiag.max().item(),
            'pw_dist_median': pw_offdiag.median().item(),
            # Movement
            'movement_mean': movement.mean().item(),
            'movement_min': movement.min().item(),
            'movement_max': movement.max().item(),
            'movement_std': movement.std().item(),
            # Distance from prior
            'prior_dist_mean': prior_dist.mean().item(),
            'prior_dist_min': prior_dist.min().item(),
            'prior_dist_max': prior_dist.max().item(),
            # Log-likelihood (per particle)
            'll_before_mean': ll_before.mean().item(),
            'll_before_std': ll_before.std().item(),
            'll_before_min': ll_before.min().item(),
            'll_before_max': ll_before.max().item(),
            'll_after_mean': ll_after.mean().item(),
            'll_after_std': ll_after.std().item(),
            'll_after_min': ll_after.min().item(),
            'll_after_max': ll_after.max().item(),
            'll_per_datapoint': ll_after.mean().item() / max(n_data, 1),
            # Log-prior (per particle)
            'lp_before_mean': lp_before.mean().item(),
            'lp_after_mean': lp_after.mean().item(),
            # Gradient health
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
            kl_weight=kl_weight if kl_weight is not None else 0.0,
            kl_divergence=0.0
        )
