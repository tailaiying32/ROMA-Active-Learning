import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from active_learning.src.test_history import TestHistory
from active_learning.src.legacy.user_distribution import UserDistribution
from active_learning.src.legacy.feasibility_checker import FeasibilityChecker


@dataclass
class VIResult:
    """Result from VI posterior update."""
    converged: bool
    n_iterations: int
    final_elbo: float
    final_grad_norm: float
    grad_norm_history: list
    elbo_history: list


class VariationalInference:
    '''
    Variational Inference class for approximating posterior updates.

    Updates the posterior via ELBO maximization (AKA minimizing KL divergence).

    ELBO = E_q[log p(H|theta)] - KL(q(theta)||p(theta))
    Where:
        - H are the observed data (feasibility labels)
        - theta are the user parameters (joint limits and pairwise constraints)
        - p(H|theta) is the likelihood of observing H given theta
        - p(theta) is the prior over theta
        - q(theta) is the variational distribution approximating the posterior

    We use the reparameterization trick to sample from q(theta) during optimization.
    ELBO(m, s) =
        1/N sum(log p(H|theta_i))
      - KL(q(theta; m, s)||p(theta))
    where theta_i = m + s * epsilon, epsilon ~ N(0, I)
    and we assume theta ~ N(m, diag(s^2)) (Gaussian variational distribution).
    '''

    def __init__(
        self,
        prior: UserDistribution,
        posterior: UserDistribution,
        config: dict = None
    ):
            '''
            Initialize Variational Inference.
            Args:
                prior: Prior user distribution
                posterior: Posterior user distribution to be updated
                config: Configuration dictionary (expects 'vi' and 'bald' sections)
            '''
            self.prior = prior
            self.posterior = posterior
            self.config = config or {}

            # Load from config (with defaults)
            vi_config = self.config.get('vi', {})
            bald_config = self.config.get('bald', {})
            
            self.n_samples = vi_config.get('n_mc_samples', 50)
            self.tau = bald_config.get('tau', 1.0) # Use same tau as BALD usually
            self.lr = vi_config.get('learning_rate', 1e-3)
            self.max_iters = vi_config.get('max_iters', 1000)
            self.convergence_tol = vi_config.get('convergence_tol', 1e-4)
            self.patience = vi_config.get('patience', 10)
            self.kl_weight = vi_config.get('kl_weight', 1.0)
            self.grad_clip = vi_config.get('grad_clip', None)

            # Minimum standard deviation to prevent posterior collapse
            self.min_std = vi_config.get('min_std', 1e-4)
            self.min_log_std = np.log(self.min_std)


    def likelihood(self, test_history: TestHistory) -> torch.Tensor:
        """
        Computes sum log p(y | t, theta) averaged over samples using vectorized operations.
        """
        results = test_history.get_all()
        if not results:
            return torch.tensor(0.0, device=self.posterior.device)

        # stack test points: (n_history, n_joints)
        test_points = torch.stack([r.test_point for r in results]).to(self.posterior.device)
        # outcomes: (n_history,)
        outcomes = torch.tensor([1.0 if r.outcome else 0.0 for r in results], device=self.posterior.device)

        # Sample from posterior (batched)
        samples = self.posterior.sample(self.n_samples, return_list=False)

        # Compute batched h-values: (n_samples, n_history)
        h_values = FeasibilityChecker.compute_h_batched(
            q=test_points,
            joint_limits=samples['joint_limits'],
            pairwise_constraints=samples['bumps'],
            joint_names=self.posterior.joint_names,
            config=self.config
        )

        # convert h values to probabilities via sigmoid
        probs = torch.sigmoid(h_values / self.tau) # (N, H)

        # Log likelihood per user sample
        # outcomes broadcast to (1, H)
        # sum over history (dim=1)
        log_probs = outcomes * torch.log(probs + 1e-8) + (1 - outcomes) * torch.log(1 - probs + 1e-8)
        total_log_likelihood = log_probs.sum(dim=1).sum(dim=0) # Sum over history, then sum (actually we want mean over samples)
        
        # We need sum over log_probs for each history point, averaged over samples
        # E_q [ sum_i log p(y_i | theta) ] = mean_samples ( sum_i log p(y_i | theta_sample) )
        
        log_likelihood_per_sample = log_probs.sum(dim=1) # (N,) sum over history
        mean_log_likelihood = log_likelihood_per_sample.mean() # Scalar
        
        return mean_log_likelihood


    def regularizer(self, kl_weight: float = None) -> torch.Tensor:
        '''
        Computes KL(q(theta)||p(theta)) between posterior and prior.
        '''
        weight = kl_weight if kl_weight is not None else self.kl_weight

        kl_divergence = torch.tensor(0.0, device=self.posterior.device)

        # Helper for KL between two Gaussians
        def gaussian_kl(mu_q, log_std_q, mu_p, log_std_p):
            var_q = torch.exp(2 * log_std_q)
            var_p = torch.exp(2 * log_std_p)

            # log(std_p / std_q) is simply (log_std_p - log_std_q)
            return (log_std_p - log_std_q) + (var_q + (mu_q - mu_p)**2) / (2 * var_p) - 0.5

        # 1. Joint Limits (independent lower and upper bounds)
        for joint in self.posterior.joint_names:
            p_post = self.posterior.params['joint_limits'][joint]
            p_prior = self.prior.params['joint_limits'][joint]

            # Lower Bound KL
            kl_divergence += gaussian_kl(p_post['lower_mean'], p_post['lower_log_std'],
                                         p_prior['lower_mean'], p_prior['lower_log_std'])
            # Upper Bound KL
            kl_divergence += gaussian_kl(p_post['upper_mean'], p_post['upper_log_std'],
                                         p_prior['upper_mean'], p_prior['upper_log_std'])

        # 2. Bumps
        for pair in self.posterior.params['bumps']:
            post_bumps = self.posterior.params['bumps'][pair]
            prior_bumps = self.prior.params['bumps'][pair]

            # Assuming same number of bumps and matched by index
            for i in range(min(len(post_bumps), len(prior_bumps))):
                b_post = post_bumps[i]
                b_prior = prior_bumps[i]

                # mu (2D)
                kl_divergence += gaussian_kl(b_post['mu']['mean'], b_post['mu']['log_std'],
                                             b_prior['mu']['mean'], b_prior['mu']['log_std']).sum()
                # ls (2D)
                kl_divergence += gaussian_kl(b_post['ls']['mean'], b_post['ls']['log_std'],
                                             b_prior['ls']['mean'], b_prior['ls']['log_std']).sum()
                # alpha (scalar)
                kl_divergence += gaussian_kl(b_post['alpha']['mean'], b_post['alpha']['log_std'],
                                             b_prior['alpha']['mean'], b_prior['alpha']['log_std'])
                # theta (scalar)
                kl_divergence += gaussian_kl(b_post['theta']['mean'], b_post['theta']['log_std'],
                                             b_prior['theta']['mean'], b_prior['theta']['log_std'])
        return weight * kl_divergence

    def update_posterior(self, test_history: TestHistory, diagnostics=None, iteration: int = None, kl_weight: float = None) -> VIResult:
        '''
        Update the posterior distribution via ELBO maximization.

        Returns:
            VIResult with convergence info and gradient diagnostics
        '''

        # collect parameters to optimize
        params = []
        param_names = []
        for joint in self.posterior.joint_names:
            p = self.posterior.params['joint_limits'][joint]
            for key in ['lower_mean', 'lower_log_std', 'upper_mean', 'upper_log_std']:
                p[key].requires_grad_(True)
                params.append(p[key])
                param_names.append(f"{joint}/{key}")

        # Bump parameters
        for pair in self.posterior.params['bumps']:
            for bump_idx, bump in enumerate(self.posterior.params['bumps'][pair]):
                for param_group in ['mu', 'ls', 'alpha', 'theta']:
                    for key in ['mean', 'log_std']:
                        bump[param_group][key].requires_grad_(True)
                        params.append(bump[param_group][key])
                        param_names.append(f"{pair}/bump{bump_idx}/{param_group}/{key}")

        optimizer = torch.optim.Adam(params, lr=self.lr)
        epochs_without_improvement = 0
        best_elbo = -float('inf')

        # Track diagnostics
        grad_norm_history = []
        elbo_history = []
        final_grad_norm = 0.0

        for i in range(self.max_iters):
            optimizer.zero_grad()

            # Compute ELBO
            log_likelihood = self.likelihood(test_history)
            kl_div = self.regularizer(kl_weight=kl_weight)
            elbo = log_likelihood - kl_div

            # We minimize negative ELBO
            loss = -elbo
            loss.backward()

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
                return VIResult(
                    converged=True,
                    n_iterations=i + 1,
                    final_elbo=current_elbo,
                    final_grad_norm=final_grad_norm,
                    grad_norm_history=grad_norm_history,
                    elbo_history=elbo_history
                )

        return VIResult(
            converged=False,
            n_iterations=self.max_iters,
            final_elbo=elbo_history[-1] if elbo_history else 0.0,
            final_grad_norm=final_grad_norm,
            grad_norm_history=grad_norm_history,
            elbo_history=elbo_history
        )

    def _clamp_params(self):
        """Clamp log_std parameters to enforce minimum bandwidth."""
        # 1. Joint Limits
        for joint in self.posterior.joint_names:
            p = self.posterior.params['joint_limits'][joint]
            p['lower_log_std'].clamp_(min=self.min_log_std)
            p['upper_log_std'].clamp_(min=self.min_log_std)

        # 2. Bumps
        for pair in self.posterior.params['bumps']:
            for bump in self.posterior.params['bumps'][pair]:
                bump['mu']['log_std'].clamp_(min=self.min_log_std)
                bump['ls']['log_std'].clamp_(min=self.min_log_std)
                bump['alpha']['log_std'].clamp_(min=self.min_log_std)
                bump['theta']['log_std'].clamp_(min=self.min_log_std)



