import torch
import numpy as np
from dataclasses import dataclass
from active_learning.src.test_history import TestHistory
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.utils import get_adaptive_param
from typing import Optional


@dataclass
class LatentVIResult:
    """Result from VI posterior update."""
    converged: bool
    n_iterations: int
    final_elbo: float
    final_grad_norm: float
    grad_norm_history: list
    elbo_history: list
    # Detailed diagnostics
    mean_history: Optional[np.ndarray] = None # (n_iters, latent_dim)
    log_std_history: Optional[np.ndarray] = None # (n_iters, latent_dim)
    mean_grad_history: Optional[np.ndarray] = None # (n_iters, latent_dim)
    log_std_grad_history: Optional[np.ndarray] = None # (n_iters, latent_dim)


class LatentVariationalInference:
    '''
    Latent Variational Inference class for approximating posterior updates in latent space.

    Updates the posterior via ELBO maximization (AKA minimizing KL divergence).

    ELBO = E_q[log p(H|z)] - KL(q(z)||p(z))
    Where:
        - H are the observed data (feasibility labels)
        - z are the latent codes representing user parameters
        - p(H|z) is the likelihood of observing H given z
        - p(z) is the prior over z
        - q(z) is the variational distribution approximating the posterior

    We use the reparameterization trick to sample from q(z) during optimization.
    ELBO(m, s) =
        1/N sum(log p(y_i|z_j))  [Gaussian Likelihood]
      - KL(q(z; m, s)||p(z))
    where z_j ~ q(z)
    '''

    def __init__(
        self,
        decoder,
        prior: LatentUserDistribution,
        posterior: LatentUserDistribution,
        config: dict = None
    ):
        '''
        Initialize Latent Variational Inference.
        Args:
            decoder: Decoder model to map latent codes to outputs
            prior: Prior latent user distribution
            posterior: Posterior latent user distribution to be updated
            config: Configuration dictionary (expects 'vi' and 'bald' sections)
        '''
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior
        self.config = config or {}

        # Load from config (with defaults)
        vi_config = self.config.get('vi', {})
        bald_config = self.config.get('bald', {})
        kl_annealing_config = vi_config.get('kl_annealing', {})

        self.n_samples = vi_config.get('n_mc_samples', 50)
        self.noise_std = vi_config.get('noise_std', 1.0)  # Kept for backward compatibility
        self.lr = vi_config.get('learning_rate', 1e-3)
        self.max_iters = vi_config.get('max_iters', 1000)
        self.convergence_tol = vi_config.get('convergence_tol', 1e-4)
        self.patience = vi_config.get('patience', 10)
        # KL weight comes from kl_annealing.end_weight (favored) or legacy kl_weight
        self.kl_weight = kl_annealing_config.get('end_weight', vi_config.get('kl_weight', 1.0))
        self.grad_clip = vi_config.get('grad_clip', None)

        # Tau temperature for sigmoid (matches BALD for consistency)
        self.tau = bald_config.get('tau', 1.0)
        self.tau_schedule = bald_config.get('tau_schedule', None)

        # Minimum standard deviation to prevent posterior collapse
        self.min_std = vi_config.get('min_std', 1e-4)
        self.min_log_std = np.log(self.min_std)

        # Detect full-covariance mode (posterior has cov_cholesky set)
        self._is_full_cov = getattr(self.posterior, 'cov_cholesky', None) is not None

    def _get_current_tau(self, iteration=None):
        """Get tau for the current iteration, applying schedule if configured."""
        if self.tau_schedule and iteration is not None:
            return get_adaptive_param(self.tau_schedule, iteration, self.tau)
        return self.tau

    def likelihood(self, test_history: TestHistory, iteration=None) -> torch.Tensor:
        """
        Computes sum log p(y | t, z) averaged over samples.

        Uses Binary Cross-Entropy (BCE) likelihood with tau-scaled probabilities.
        This matches the probabilistic interpretation used in BALD acquisition.

        Uses binary 0/1 targets from the oracle, matching the binary observation
        model assumed by BALD acquisition.

        Uses batched decoder calls for efficiency.
        """
        results = test_history.get_all()
        if not results:
            return torch.tensor(0.0, device=self.posterior.device)

        # Stack test points and outcomes (binary 0.0/1.0)
        test_points = torch.stack([r.test_point for r in results]).to(self.posterior.device)
        outcomes = torch.tensor([r.outcome for r in results], device=self.posterior.device).unsqueeze(0)  # (1, n_points)

        # Get N samples using reparameterization
        samples = self.posterior.sample(self.n_samples)

        # Ensure batched tensor for efficient computation
        if torch.is_tensor(samples):
            z_batch = samples  # (n_samples, latent_dim)
        else:
            z_batch = torch.stack(samples)  # (n_samples, latent_dim)

        # Compute logits for all samples in one batched forward pass
        # logits shape: (n_samples, n_points)
        pred_logits = LatentFeasibilityChecker.batched_logit_values(self.decoder, z_batch, test_points)

        # Use BCEWithLogits for numerical stability (avoids gradient killing from clamp)
        # Note: we want Log Likelihood, which is -BCE
        # scaled_logits matches the tau scaling used for target_probs
        current_tau = self._get_current_tau(iteration)
        scaled_logits = pred_logits / current_tau
        
        # Expand targets to match samples (n_samples, n_points)
        targets_expanded = outcomes.expand_as(scaled_logits)

        neg_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            scaled_logits,
            targets_expanded,
            reduction='none'
        )
        
        log_probs = -neg_bce

        # Sum over test points, then average over samples
        total_log_likelihood = log_probs.sum(dim=1).mean()

        return total_log_likelihood

    @staticmethod
    def _build_cholesky(chol_raw):
        """Convert unconstrained raw matrix to valid lower-triangular Cholesky factor.

        Uses log-Cholesky parameterization: diagonal stored in log-space,
        off-diagonal unconstrained. Ensures positive diagonal.
        """
        return torch.tril(chol_raw, diagonal=-1) + torch.diag(torch.exp(chol_raw.diagonal()))

    def _full_cov_kl(self):
        """Closed-form KL(q||p) for full-covariance posterior vs Gaussian prior.

        KL(N(μ_q, Σ_q) || N(μ_p, Σ_p)) where Σ_q = L_q L_q^T.
        Supports both full-covariance and diagonal priors.
        """
        L_q = self.posterior.cov_cholesky  # (D, D) — set during optimization loop
        mu_q = self.posterior.mean
        mu_p = self.prior.mean
        D = mu_q.shape[0]
        diff = mu_p - mu_q

        if getattr(self.prior, 'precision_matrix', None) is not None:
            # Full-covariance prior: use precision matrix for efficiency
            P_p = self.prior.precision_matrix
            Sigma_q = L_q @ L_q.T
            trace_term = (P_p * Sigma_q).sum()            # tr(Σ_p^{-1} Σ_q)
            quad_term = diff @ P_p @ diff                  # (μ_p-μ_q)^T Σ_p^{-1} (μ_p-μ_q)
            if getattr(self.prior, 'cov_cholesky', None) is not None:
                log_det_p = 2.0 * self.prior.cov_cholesky.diagonal().log().sum()
            else:
                log_det_p = 2.0 * self.prior.log_std.sum()
        else:
            # Diagonal prior fallback
            var_p = torch.exp(2 * self.prior.log_std)
            Sigma_q = L_q @ L_q.T
            trace_term = (Sigma_q.diagonal() / var_p).sum()
            quad_term = (diff ** 2 / var_p).sum()
            log_det_p = 2.0 * self.prior.log_std.sum()

        log_det_q = 2.0 * L_q.diagonal().log().sum()
        log_det_ratio = log_det_p - log_det_q

        return 0.5 * (trace_term + quad_term - D + log_det_ratio)

    def regularizer(self, kl_weight: float = None) -> torch.Tensor:
        '''
        Computes KL(q(z)||p(z)) between posterior and prior.

        Dispatches to full-covariance or diagonal KL based on posterior type.
        '''
        # Use provided weight or default to instance variable
        weight = kl_weight if kl_weight is not None else self.kl_weight

        if self._is_full_cov:
            return weight * self._full_cov_kl()

        # Diagonal Gaussians:
        # KL(q||p) = sum_i [ log(std_p_i / std_q_i) + (var_q_i + (mu_q_i - mu_p_i)^2) / (2 * var_p_i) - 0.5 ]
        mu_q = self.posterior.mean
        log_std_q = self.posterior.log_std
        mu_p = self.prior.mean
        log_std_p = self.prior.log_std

        var_q = torch.exp(2 * log_std_q)
        var_p = torch.exp(2 * log_std_p)

        kl_per_dim = (log_std_p - log_std_q) + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5
        kl_divergence = kl_per_dim.sum()

        return weight * kl_divergence

    def update_posterior(self, test_history: TestHistory, kl_weight: float = None, diagnostics=None, iteration: int = None) -> LatentVIResult:
        '''
        Update the posterior distribution via ELBO maximization.

        Args:
            test_history: History of test points and outcomes
            kl_weight: Optional dynamic KL weight for this update (overrides config)

        Returns:
            LatentVIResult with convergence info and gradient diagnostics
        '''
        if self._is_full_cov:
            return self._update_posterior_full_cov(test_history, kl_weight, diagnostics, iteration)

        # Collect parameters to optimize (mean and log_std of the posterior)
        self.posterior.mean.requires_grad_(True)
        self.posterior.log_std.requires_grad_(True)
        params = [self.posterior.mean, self.posterior.log_std]

        optimizer = torch.optim.Adam(params, lr=self.lr)
        epochs_without_improvement = 0
        best_elbo = -float('inf')

        # Track diagnostics
        grad_norm_history = []
        elbo_history = []
        
        # Detailed diagnostics
        mean_history = []
        log_std_history = []
        mean_grad_history = []
        log_std_grad_history = []
        
        final_grad_norm = 0.0

        for i in range(self.max_iters):
            optimizer.zero_grad()

            # Compute ELBO
            log_likelihood = self.likelihood(test_history, iteration=iteration)
            kl_div = self.regularizer(kl_weight=kl_weight)
            elbo = log_likelihood - kl_div

            # We minimize negative ELBO
            loss = -elbo
            loss.backward()
            
            # Capture gradients and parameters for diagnostics
            with torch.no_grad():
                mean_history.append(self.posterior.mean.detach().cpu().numpy().copy())
                log_std_history.append(self.posterior.log_std.detach().cpu().numpy().copy())
                
                if self.posterior.mean.grad is not None:
                     mean_grad_history.append(self.posterior.mean.grad.detach().cpu().numpy().copy())
                else:
                     mean_grad_history.append(np.zeros_like(mean_history[-1]))
                     
                if self.posterior.log_std.grad is not None:
                     log_std_grad_history.append(self.posterior.log_std.grad.detach().cpu().numpy().copy())
                else:
                     log_std_grad_history.append(np.zeros_like(log_std_history[-1]))

            # Compute gradient norm before clipping (for diagnostics)
            total_grad_norm = 0.0
            for p in params:
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = np.sqrt(total_grad_norm)
            grad_norm_history.append(total_grad_norm)
            final_grad_norm = total_grad_norm

            # Clip gradients to prevent exploding gradients
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

            optimizer.step()

            # Clamp log_std parameters to prevent variance collapse
            with torch.no_grad():
                self._clamp_params()

            current_elbo = elbo.item()
            elbo_history.append(current_elbo)

            if current_elbo > best_elbo + self.convergence_tol:
                best_elbo = current_elbo
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                return LatentVIResult(
                    converged=True,
                    n_iterations=i + 1,
                    final_elbo=current_elbo,
                    final_grad_norm=final_grad_norm,
                    grad_norm_history=grad_norm_history,
                    elbo_history=elbo_history,
                    mean_history=np.array(mean_history),
                    log_std_history=np.array(log_std_history),
                    mean_grad_history=np.array(mean_grad_history),
                    log_std_grad_history=np.array(log_std_grad_history)
                )

        return LatentVIResult(
            converged=False,
            n_iterations=self.max_iters,
            final_elbo=elbo_history[-1] if elbo_history else 0.0,
            final_grad_norm=final_grad_norm,
            grad_norm_history=grad_norm_history,
            elbo_history=elbo_history,
            mean_history=np.array(mean_history),
            log_std_history=np.array(log_std_history),
            mean_grad_history=np.array(mean_grad_history),
            log_std_grad_history=np.array(log_std_grad_history)
        )

    def _update_posterior_full_cov(self, test_history, kl_weight, diagnostics, iteration):
        """Update posterior with full-covariance Cholesky parameterization.

        Optimizes mean and a raw Cholesky factor (log-diagonal parameterization)
        to maximize ELBO with closed-form KL.
        """
        # 1. Convert current Cholesky L to raw parameterization (log-diagonal)
        L_init = self.posterior.cov_cholesky.detach().clone()
        chol_raw = L_init.clone()
        chol_raw.diagonal().copy_(torch.log(L_init.diagonal()))
        chol_raw.requires_grad_(True)

        self.posterior.mean.requires_grad_(True)
        params = [self.posterior.mean, chol_raw]

        optimizer = torch.optim.Adam(params, lr=self.lr)
        epochs_without_improvement = 0
        best_elbo = -float('inf')

        grad_norm_history = []
        elbo_history = []
        mean_history = []
        log_std_history = []
        mean_grad_history = []
        log_std_grad_history = []
        final_grad_norm = 0.0

        for i in range(self.max_iters):
            optimizer.zero_grad()

            # Build valid Cholesky from raw params (positive diagonal via exp)
            L = self._build_cholesky(chol_raw)
            self.posterior.cov_cholesky = L  # autograd flows through

            # Compute ELBO
            log_likelihood = self.likelihood(test_history, iteration=iteration)
            kl_div = self.regularizer(kl_weight=kl_weight)
            elbo = log_likelihood - kl_div

            loss = -elbo
            loss.backward()

            # Diagnostics: marginal std from L
            with torch.no_grad():
                mean_history.append(self.posterior.mean.detach().cpu().numpy().copy())
                marginal_log_std = torch.log(L.detach().pow(2).sum(dim=1).sqrt() + 1e-8)
                log_std_history.append(marginal_log_std.cpu().numpy().copy())

                if self.posterior.mean.grad is not None:
                    mean_grad_history.append(self.posterior.mean.grad.detach().cpu().numpy().copy())
                else:
                    mean_grad_history.append(np.zeros(self.posterior.latent_dim))

                if chol_raw.grad is not None:
                    # Summarize chol_raw grad as diagonal for compatibility
                    log_std_grad_history.append(chol_raw.grad.diagonal().detach().cpu().numpy().copy())
                else:
                    log_std_grad_history.append(np.zeros(self.posterior.latent_dim))

            # Gradient norm
            total_grad_norm = 0.0
            for p in params:
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = np.sqrt(total_grad_norm)
            grad_norm_history.append(total_grad_norm)
            final_grad_norm = total_grad_norm

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

            optimizer.step()

            # Clamp raw diagonal to prevent variance collapse
            with torch.no_grad():
                chol_raw.diagonal().clamp_(min=self.min_log_std)

            current_elbo = elbo.item()
            elbo_history.append(current_elbo)

            if current_elbo > best_elbo + self.convergence_tol:
                best_elbo = current_elbo
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        # Finalize: store converged Cholesky and update log_std for compatibility
        with torch.no_grad():
            L_final = self._build_cholesky(chol_raw)
            self.posterior.cov_cholesky = L_final.detach()
            self.posterior.log_std = torch.log(L_final.detach().pow(2).sum(dim=1).sqrt() + 1e-8)
            self.posterior.mean.requires_grad_(False)

        converged = epochs_without_improvement >= self.patience
        n_iters = len(elbo_history)

        return LatentVIResult(
            converged=converged,
            n_iterations=n_iters,
            final_elbo=elbo_history[-1] if elbo_history else 0.0,
            final_grad_norm=final_grad_norm,
            grad_norm_history=grad_norm_history,
            elbo_history=elbo_history,
            mean_history=np.array(mean_history),
            log_std_history=np.array(log_std_history),
            mean_grad_history=np.array(mean_grad_history),
            log_std_grad_history=np.array(log_std_grad_history)
        )

    def _clamp_params(self):
        """Clamp log_std parameters to enforce minimum bandwidth."""
        self.posterior.log_std.clamp_(min=self.min_log_std)
