"""
Latent BALD (Bayesian Active Learning by Disagreement) acquisition function.

Selects test points that maximize information gain about user reachability
by finding points with maximum disagreement among posterior samples.
"""

import torch
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.config import create_generator
from active_learning.src.utils import binary_entropy, get_adaptive_param


class LatentBALD:
    """Latent BALD acquisition function for selecting informative test points."""

    def __init__(
            self,
            decoder,
            posterior: LatentUserDistribution,
            config: dict = None,
            prior: LatentUserDistribution = None
        ):
        """
        Args:
            decoder: Decoder model to map latent codes to outputs
            posterior: Posterior distribution over latent user parameters
            config: Configuration dictionary (expects 'bald' and 'bald_optimization' sections)
        """
        self.decoder = decoder
        self.posterior = posterior
        self.prior = prior
        self.config = config or {}

        # BALD settings
        bald_cfg = self.config.get('bald', {})
        self.tau = bald_cfg.get('tau', 1.0)
        self.n_samples = bald_cfg.get('n_mc_samples', 50)
        self.sampling_temperature = bald_cfg.get('sampling_temperature', 1.0)
        
        # Weighted BALD settings
        self.use_weighted_bald = bald_cfg.get('use_weighted_bald', False)
        self.weighted_bald_sigma = bald_cfg.get('weighted_bald_sigma', 0.1)
        
        # Legacy Boundary Weight check
        boundary_weight = bald_cfg.get('boundary_weight', 0.0)
        if self.use_weighted_bald and boundary_weight > 0:
            raise ValueError(
                "Configuration Error: 'use_weighted_bald' is True, but deprecated 'boundary_weight' is > 0. "
                "Weighted BALD and the legacy KL-divergence term are mutually exclusive. "
                "Please set 'boundary_weight: 0.0' in your config."
            )

        # Optimization settings
        opt_cfg = self.config.get('bald_optimization', {})
        self.opt_n_restarts = opt_cfg.get('n_restarts', 5)
        
        # Create generator for reproducibility (None = truly random)
        self.generator = create_generator(self.config, self.posterior.device)
        self.opt_n_iters = opt_cfg.get('n_iters_per_restart', 50)
        self.opt_lr_adam = opt_cfg.get('lr_adam', 0.05)
        self.opt_lr_sgd = opt_cfg.get('lr_sgd', 0.01)
        self.opt_switch_to_sgd_at = opt_cfg.get('switch_to_sgd_at', 0.75)
        
        # Adaptive schedules (nested under bald: in config)
        self.weighted_bald_sigma_schedule = bald_cfg.get('weighted_bald_sigma_schedule', None)
        self.tau_schedule = bald_cfg.get('tau_schedule', None)

    def compute_score(self, test: torch.Tensor, zs: list = None, decoded_params: tuple = None, iteration: int = None) -> torch.Tensor:
        """
        Compute BALD score for a test point.

        BALD = H(E[p]) - E[H(p)]  (mutual information)

        Args:
            test: Test point tensor of shape (n_joints,)
            zs: Optional pre-sampled latent vectors. If None, samples fresh.
            decoded_params: Optional pre-decoded RBF parameters (tuple).
            iteration: Current iteration index for adaptive schedules.

        Returns:
            BALD score (scalar tensor)
        """
        # Sample latent codes from posterior if not provided (and not pre-decoded)
        if zs is None and decoded_params is None:
            zs = self.posterior.sample(self.n_samples, temperature=self.sampling_temperature)
            
        current_tau = self.tau
        if self.tau_schedule and iteration is not None:
             current_tau = get_adaptive_param(self.tau_schedule, iteration, self.tau)

        if decoded_params is not None:
            # FAST PATH: Use pre-decoded parameters
            logits = LatentFeasibilityChecker.evaluate_from_decoded(test, decoded_params) / current_tau
        else:
            # SLOW PATH: Decode then evaluate
            # Ensure batched tensor (n_samples, latent_dim)
            if torch.is_tensor(zs):
                z_batch = zs
            else:
                z_batch = torch.stack(zs)
            logits = LatentFeasibilityChecker.batched_logit_values(self.decoder, z_batch, test) / current_tau  # (n_samples,)
            
        probs = torch.sigmoid(logits)  # (n_samples,)

        # BALD = Entropy of mean - Mean of entropies
        p_mean = probs.mean(dim=0)
        entropy_of_mean = binary_entropy(p_mean)
        mean_of_entropies = binary_entropy(probs).mean(dim=0)
        bald_score = entropy_of_mean - mean_of_entropies

        # Weighted BALD: Apply Gaussian Gate centered at p=0.5
        # Gates the score to focus only on regions near the decision boundary
        if self.use_weighted_bald:
            # w = exp( - (p - 0.5)^2 / (2 * sigma^2) )
            diff = p_mean - 0.5
            
            w_sigma = self.weighted_bald_sigma
            if self.weighted_bald_sigma_schedule and iteration is not None:
                w_sigma = get_adaptive_param(self.weighted_bald_sigma_schedule, iteration, w_sigma)

            gate = torch.exp(- (diff**2) / (2 * w_sigma**2))
            return bald_score * gate

        # Boundary-Aware Exploration Bonus (Legacy KL Term)
        # Targeted exploration of regions where the posterior belief differs significantly from the prior
        # Score += w * KL(P_posterior || P_prior)
        boundary_weight = self.config.get('bald', {}).get('boundary_weight', 0.0)
        if boundary_weight > 0 and self.prior is not None:
            # Sample from prior (using same generator/seed logic if needed, but simple sampling is fine for comparison)
            # We reuse the same number of samples as posterior
            z_prior = self.prior.sample(self.n_samples)
            if not torch.is_tensor(z_prior):
                z_prior = torch.stack(z_prior)
            
            # Predict with prior samples
            prior_logits = LatentFeasibilityChecker.batched_logit_values(self.decoder, z_prior, test) / self.tau
            prior_probs = torch.sigmoid(prior_logits)
            p_prior_mean = prior_probs.mean(dim=0)
            
            # Compute KL(p_mean || p_prior_mean) for Bernoulli distributions
            # KL(p||q) = p log(p/q) + (1-p) log((1-p)/(1-q))
            eps = 1e-6
            p = torch.clamp(p_mean, eps, 1-eps)
            q = torch.clamp(p_prior_mean, eps, 1-eps)
            
            kl = p * torch.log(p/q) + (1-p) * torch.log((1-p)/(1-q))
            
            return bald_score + boundary_weight * kl

        return bald_score

    def _detach_samples(self, zs: torch.Tensor) -> torch.Tensor:
        """
        Detach latent tensor batch to prevent gradient accumulation.

        This is needed when reusing samples across multiple backward passes,
        since we only need gradients w.r.t. the test point, not the samples.

        Args:
            zs: Tensor of shape (n_samples, latent_dim)

        Returns:
            Detached tensor of same shape
        """
        if torch.is_tensor(zs):
            return zs.detach()
        return torch.stack(zs).detach()

    def compute_score_batched(
        self,
        tests: torch.Tensor,
        decoded_params: tuple = None,
        iteration: int = None
    ) -> torch.Tensor:
        """
        Compute BALD scores for multiple test points simultaneously.

        This is the batched version of compute_score() for parallel optimization
        of multiple restarts.

        Args:
            tests: Test points tensor of shape (n_tests, n_joints)
            decoded_params: Pre-decoded RBF parameters (n_samples, ...), shared across all tests
            iteration: Current iteration index for adaptive schedules

        Returns:
            BALD scores tensor of shape (n_tests,)
        """
        # Get current tau from schedule
        current_tau = self.tau
        if self.tau_schedule and iteration is not None:
            current_tau = get_adaptive_param(self.tau_schedule, iteration, self.tau)

        # Evaluate all test points against all samples: (n_samples, n_tests)
        logits = LatentFeasibilityChecker.evaluate_from_decoded(tests, decoded_params) / current_tau
        probs = torch.sigmoid(logits)  # (n_samples, n_tests)

        # BALD = H(E[p]) - E[H(p)]
        # Mean over samples (dim=0), keep test points (dim=1)
        p_mean = probs.mean(dim=0)  # (n_tests,)
        entropy_of_mean = binary_entropy(p_mean)  # (n_tests,)
        mean_of_entropies = binary_entropy(probs).mean(dim=0)  # (n_tests,)
        bald_scores = entropy_of_mean - mean_of_entropies  # (n_tests,)

        # Weighted BALD: Apply Gaussian Gate centered at p=0.5
        if self.use_weighted_bald:
            w_sigma = self.weighted_bald_sigma
            if self.weighted_bald_sigma_schedule and iteration is not None:
                w_sigma = get_adaptive_param(self.weighted_bald_sigma_schedule, iteration, w_sigma)
            diff = p_mean - 0.5
            gate = torch.exp(-(diff**2) / (2 * w_sigma**2))  # (n_tests,)
            bald_scores = bald_scores * gate

        return bald_scores

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
        diagnostics = None
    ) -> tuple:
        """
        Find optimal test point via gradient ascent on BALD score.

        Args:
            bounds: Tensor of shape (n_joints, 2) with [lower, upper] bounds
            n_restarts: Number of random restarts
            n_iters: Iterations per restart
            lr_adam: Learning rate for Adam optimizer
            lr_sgd: Learning rate for SGD optimizer
            switch_to_sgd_at: Fraction of iterations at which to switch from Adam to SGD
            verbose: Print progress
            test_history: Optional list of past test points (tensors) for diversity bonus

        Returns:
            (best_test, best_score, diagnostics_stats) tuple
        """
        # Use config values if not provided
        n_restarts = n_restarts if n_restarts is not None else self.opt_n_restarts
        n_iters = n_iters if n_iters is not None else self.opt_n_iters
        lr_adam = lr_adam if lr_adam is not None else self.opt_lr_adam
        lr_sgd = lr_sgd if lr_sgd is not None else self.opt_lr_sgd
        switch_to_sgd_at = switch_to_sgd_at if switch_to_sgd_at is not None else self.opt_switch_to_sgd_at

        # Sequential diversity settings
        diversity_weight = self.config.get('bald', {}).get('diversity_weight', 0.0)
        history_tensor = None
        if diversity_weight > 0 and test_history is not None and len(test_history) > 0:
            # Stack history tensors for efficient distance computation
            history_tensor = torch.stack([h.to(self.posterior.device) for h in test_history])

        device = self.posterior.device
        n_joints = bounds.shape[0]

        diag_stats_list = []

        # === BATCHED OPTIMIZATION ===
        # Initialize ALL restarts at once: (n_restarts, n_joints)
        lower = bounds[:, 0].detach()  # (n_joints,)
        upper = bounds[:, 1].detach()  # (n_joints,)
        t = lower + torch.rand(n_restarts, n_joints, device=device, generator=self.generator) * (upper - lower)
        t = t.clone().requires_grad_(True)

        # Sample latent vectors ONCE, shared across all restarts
        # This is valid because restarts differ in starting point, not in the objective
        zs = self._detach_samples(self.posterior.sample(
            self.n_samples, temperature=self.sampling_temperature, generator=self.generator
        ))

        # PRE-DECODE SAMPLES ONCE (shared across all restarts)
        with torch.no_grad():
            decoded_params = LatentFeasibilityChecker.decode_latent_params(self.decoder, zs)
            decoded_params = tuple(p.detach() for p in decoded_params)

        # Evaluate initial scores for all restarts
        with torch.no_grad():
            initial_scores = self.compute_score_batched(t, decoded_params=decoded_params, iteration=iteration)
            initial_best_score = initial_scores.max().item()

        # Single optimizer for all restarts (PyTorch handles batched params naturally)
        optimizer = torch.optim.Adam([t], lr=lr_adam)
        use_sgd = False

        # Optimization loop (single loop for ALL restarts in parallel)
        for i in range(n_iters):
            # Switch to SGD after specified fraction of iterations
            if (not use_sgd) and (i >= int(switch_to_sgd_at * n_iters)):
                optimizer = torch.optim.SGD([t], lr=lr_sgd)
                use_sgd = True

            optimizer.zero_grad()

            # Compute scores for ALL restarts at once: (n_restarts,)
            scores = self.compute_score_batched(t, decoded_params=decoded_params, iteration=iteration)

            # Add Sequential Diversity Bonus (batched)
            if history_tensor is not None:
                # t: (n_restarts, n_joints), history_tensor: (n_history, n_joints)
                # Compute pairwise distances: (n_restarts, n_history)
                distances = torch.cdist(t, history_tensor)
                min_distances = distances.min(dim=1).values  # (n_restarts,)
                scores = scores + diversity_weight * min_distances

            # Sum and backprop (gradients are independent per restart)
            (-scores.sum()).backward()

            optimizer.step()

            # Project all test points to bounds
            with torch.no_grad():
                t.data = torch.clamp(t.data, lower, upper)

        # Evaluate final optimized scores on the SAME samples
        with torch.no_grad():
            optimized_scores = self.compute_score_batched(t, decoded_params=decoded_params, iteration=iteration)
            optimized_best_score = optimized_scores.max().item()

        # Use fresh samples for final unbiased score evaluation
        final_zs = self._detach_samples(self.posterior.sample(
            self.n_samples, temperature=self.sampling_temperature, generator=self.generator
        ))
        with torch.no_grad():
            final_decoded = LatentFeasibilityChecker.decode_latent_params(self.decoder, final_zs)

        # Compute final scores for all restarts
        with torch.no_grad():
            final_scores = self.compute_score_batched(t, decoded_params=final_decoded, iteration=iteration)

        # Select best restart
        best_idx = final_scores.argmax()
        best_test = t[best_idx].detach().clone()
        best_score = final_scores[best_idx].item()

        # Diagnostics logging (batched)
        if diagnostics is not None:
            with torch.no_grad():
                current_tau = self.tau
                if self.tau_schedule and iteration is not None:
                    current_tau = get_adaptive_param(self.tau_schedule, iteration, self.tau)

                # Evaluate all restarts for diagnostics: (n_samples, n_restarts)
                logits = LatentFeasibilityChecker.evaluate_from_decoded(t, final_decoded) / current_tau
                probs = torch.sigmoid(logits)
                p_means = probs.mean(dim=0)  # (n_restarts,)

                # Compute gate values if using weighted BALD
                if self.use_weighted_bald:
                    w_sigma = self.weighted_bald_sigma
                    if self.weighted_bald_sigma_schedule and iteration is not None:
                        w_sigma = get_adaptive_param(self.weighted_bald_sigma_schedule, iteration, w_sigma)
                    gate_vals = torch.exp(-((p_means - 0.5)**2) / (2 * w_sigma**2))
                else:
                    gate_vals = torch.ones_like(p_means)

                for r in range(n_restarts):
                    diag_stats_list.append({
                        'restart': r,
                        'initial_bald': initial_scores[r].item(),
                        'final_bald': final_scores[r].item(),
                        'p_mean': p_means[r].item(),
                        'gate': gate_vals[r].item()
                    })

        if verbose:
            gain = optimized_best_score - initial_best_score
            # print(f"  [BALD Opt] Random Best: {initial_best_score:.4f} -> Opt Best: {optimized_best_score:.4f} (Gain: {gain:+.4f})")

        return best_test, best_score, diag_stats_list
