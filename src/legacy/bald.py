"""
BALD (Bayesian Active Learning by Disagreement) acquisition function.

Selects test points that maximize information gain about user reachability
by finding points with maximum disagreement among posterior samples.
"""

import torch
from active_learning.src.legacy.user_distribution import UserDistribution
from active_learning.src.legacy.feasibility_checker import FeasibilityChecker
from active_learning.src.utils import binary_entropy, get_adaptive_param


class BALD:
    """BALD acquisition function for selecting informative test points."""

    def __init__(
            self,
            posterior: UserDistribution,
            config: dict = None
        ):
        """
        Args:
            posterior: Posterior distribution over user parameters
            config: Configuration dictionary (expects 'bald' and 'bald_optimization' sections)
        """
        self.posterior = posterior
        self.config = config or {}

        # 1. BALD settings (Priority: Config > Default)
        bald_cfg = self.config.get('bald', {})
        self.tau = bald_cfg.get('tau', 1.0)
        self.n_samples = bald_cfg.get('n_mc_samples', 50)
        self.sampling_temperature = bald_cfg.get('sampling_temperature', 1.0)

        # 2. Optimization settings
        opt_cfg = self.config.get('bald_optimization', {})
        self.opt_n_restarts = opt_cfg.get('n_restarts', 5)
        self.opt_n_iters = opt_cfg.get('n_iters_per_restart', 50)
        self.opt_lr_adam = opt_cfg.get('lr_adam', 0.05)
        self.opt_lr_sgd = opt_cfg.get('lr_sgd', 0.01)
        self.opt_switch_to_sgd_at = opt_cfg.get('switch_to_sgd_at', 0.75)
        
        # Adaptive schedules (if provided)
        self.weighted_bald_sigma_schedule = self.config.get('weighted_bald_sigma_schedule', None)
        self.tau_schedule = self.config.get('tau_schedule', None)
        
    def compute_score(self, test: torch.Tensor, thetas: dict = None, iteration: int = None) -> torch.Tensor:
        """
        Compute BALD score for a test point using vectorized operations.

        BALD = H(E[p]) - E[H(p)]  (mutual information)

        Args:
            test: Test point tensor of shape (n_joints,) or (B, n_joints)
            thetas: Optional pre-sampled virtual users (dict of tensors).
                    If None, samples fresh.

        Returns:
            BALD score (scalar if single test point, (B,) if batch)
        """
        # Sample virtual users from posterior if not provided
        if thetas is None:
            # return_list=False gives us the vectorized structure
            thetas = self.posterior.sample(self.n_samples, temperature=self.sampling_temperature, return_list=False)

        joint_limits = thetas['joint_limits'] # (N, n_joints, 2)
        bumps = thetas['bumps'] # Dict of batched params

        # Use Batched Feasibility Checker
        # h: (N, B) where N=samples, B=batch size (or 1)
        # Note: compute_h_batched handles single point (n_joints,) correctly too
        h = FeasibilityChecker.compute_h_batched(
            q=test,
            joint_limits=joint_limits,
            pairwise_constraints=bumps,
            joint_names=self.posterior.joint_names,
            config=self.config
        )

        # Adaptive Tau
        current_tau = self.tau
        if self.tau_schedule and iteration is not None:
             current_tau = get_adaptive_param(self.tau_schedule, iteration, self.tau)

        logits = h / current_tau # (N, B)
        probs = torch.sigmoid(logits) # (N, B)

        # BALD = Entropy of mean - Mean of entropies
        p_mean = probs.mean(dim=0) # (B,)
        entropy_of_mean = binary_entropy(p_mean)
        mean_of_entropies = binary_entropy(probs).mean(dim=0) # (B,)

        score = entropy_of_mean - mean_of_entropies

        # Weighted BALD: Apply Gaussian Gate centered at p=0.5
        use_weighted_bald = self.config.get('bald', {}).get('use_weighted_bald', False)
        if use_weighted_bald:
            default_sigma = self.config.get('bald', {}).get('weighted_bald_sigma', 0.1)
            weighted_bald_sigma = get_adaptive_param(self.weighted_bald_sigma_schedule, iteration, default_sigma)

            # w = exp( - (p - 0.5)^2 / (2 * sigma^2) )
            diff = p_mean - 0.5
            gate = torch.exp(- (diff**2) / (2 * weighted_bald_sigma**2))
            score = score * gate
        
        # If input was single point, return scalar
        if test.dim() == 1:
            return score.squeeze()
            
        return score # (B,)

    def _detach_samples(self, thetas: dict) -> dict:
        """
        Detach all tensors in sampled thetas dict to prevent gradient accumulation.
        Handles the batched tensor format.
        """
        detached = {
            'joint_limits': thetas['joint_limits'].detach(),
            'bumps': {}
        }
        
        for pair, bump_params in thetas['bumps'].items():
            detached['bumps'][pair] = {}
            for k, v in bump_params.items():
                if isinstance(v, torch.Tensor):
                    detached['bumps'][pair][k] = v.detach()
                else:
                    detached['bumps'][pair][k] = v
                    
        return detached

    def select_test(
        self,
        bounds: torch.Tensor,
        n_restarts: int = None,
        n_iters: int = None,
        lr_adam: float = None,
        lr_sgd: float = None,
        switch_to_sgd_at: float = None,
        verbose: bool = False,
        iteration: int = None,
        diagnostics = None
    ) -> tuple:
        """
        Find optimal test point via gradient ascent on BALD score.

        Args:
            bounds: Tensor of shape (n_joints, 2) with [lower, upper] bounds
            n_restarts: Number of random restarts
            n_iters: Iterations per restart
            lr: Learning rate
            verbose: Print progress

        Returns:
            (best_test, best_score) tuple
        """
        # Use config values if not provided
        n_restarts = n_restarts if n_restarts is not None else self.opt_n_restarts
        n_iters = n_iters if n_iters is not None else self.opt_n_iters
        lr_adam = lr_adam if lr_adam is not None else self.opt_lr_adam
        lr_sgd = lr_sgd if lr_sgd is not None else self.opt_lr_sgd
        switch_to_sgd_at = switch_to_sgd_at if switch_to_sgd_at is not None else self.opt_switch_to_sgd_at

        device = self.posterior.device
        n_joints = bounds.shape[0]

        best_test = None
        best_score = -float('inf')

        import sys
        
        # Diagnostic tracking
        diag_stats_list = []

        # print(f"[BALD] select_test: {n_restarts} restarts, {n_iters} iters per restart", flush=True)
        for restart in range(n_restarts):
            # print(f"[BALD] Restart {restart+1}/{n_restarts} - initializing", flush=True)
            # Random initialization within bounds
            lower = bounds[:, 0].detach()
            upper = bounds[:, 1].detach()
            t = lower + torch.rand(n_joints, device=device) * (upper - lower)
            t = t.clone().requires_grad_(True)

            # Start with Adam optimizer
            optimizer = torch.optim.Adam([t], lr=lr_adam)
            use_sgd = False

            # Sample virtual users ONCE per restart for stable optimization
            # This prevents the objective landscape from changing each iteration
            # Detach samples so gradients only flow through test point, not samples
            thetas = self._detach_samples(self.posterior.sample(self.n_samples, temperature=self.sampling_temperature))
            
            initial_score_val = self.compute_score(t, thetas=thetas, iteration=iteration).item()

            for i in range(n_iters):
                # Switch to SGD after specified fraction of iterations
                if (not use_sgd) and (i >= int(switch_to_sgd_at * n_iters)):
                    optimizer = torch.optim.SGD([t], lr=lr_sgd)
                    use_sgd = True
                optimizer.zero_grad()
                score = self.compute_score(t, thetas=thetas, iteration=iteration)  # Use fixed samples
                (-score).backward()  # Maximize score
                optimizer.step()

                # Project to bounds
                with torch.no_grad():
                    t.data = torch.clamp(t.data, lower, upper)

            # Use fresh samples for final unbiased score evaluation
            final_score = self.compute_score(t, iteration=iteration).item()
            if final_score > best_score:
                best_score = final_score
                best_test = t.detach().clone()
            
            # Diagnostics: Compute p_mean and gate for logging
            if diagnostics is not None:
                with torch.no_grad():
                     # Re-compute to get p_mean
                     h = FeasibilityChecker.compute_h_batched(
                        q=t,
                        joint_limits=thetas['joint_limits'],
                        pairwise_constraints=thetas['bumps'],
                        joint_names=self.posterior.joint_names,
                        config=self.config
                     )
                     current_tau = self.tau
                     if self.tau_schedule and iteration is not None:
                         current_tau = get_adaptive_param(self.tau_schedule, iteration, self.tau)
                     
                     probs = torch.sigmoid(h / current_tau)
                     p_mean = probs.mean().item()
                     gate_val = 1.0
                     
                     use_weighted_bald = self.config.get('bald', {}).get('use_weighted_bald', False)
                     if use_weighted_bald:
                         default_sigma = self.config.get('bald', {}).get('weighted_bald_sigma', 0.1)
                         weighted_bald_sigma = get_adaptive_param(self.weighted_bald_sigma_schedule, iteration, default_sigma)
                         gate_val = torch.exp(-torch.tensor((p_mean - 0.5)**2) / (2 * weighted_bald_sigma**2)).item()

                     diag_stats_list.append({
                         'restart': restart,
                         'initial_bald': initial_score_val,
                         'final_bald': final_score,
                         'p_mean': p_mean,
                         'gate': gate_val
                     })

            # print(f"[BALD] Restart {restart + 1}/{n_restarts}: score={final_score:.4f}", flush=True)

        # Pass diagnostics back via the list object if provided (it's mutable)
        if diagnostics is not None and hasattr(diagnostics, 'bald_opt_stats'):
             # We can't set attributes on the Diagnostics object directly from here cleanly without passing it as 'diagnostics' arg
             # But we can return it or append to a list if passed. 
             # The caller (active_learning_pipeline) will handle passing this list to diagnostics.log_iteration
             # Wait, `diagnostics` arg is the Diagnostics object.
             # We can add a temporary attribute to it or just return stats?
             # Better: let's modifying log_iteration to accept this list.
             # Here we just need to return it? 
             # The signature of select_test doesn't support returning extra stuff easily without breaking interface.
             # So we will modify the Diagnostics object in place if possible, OR return it as a third element?
             # No, easier: add a method to Diagnostics or just hold it in a list passed in. 
             # Let's assume the caller passes a list `bald_diagnostics_list`?
             # Current signature: diagnostics=None. If it's the Diagnostics object, we can't easily stash data in it for the specific iteration unless we add a field.
             pass 
        
        # Actually, let's just use a side channel or return it if we can.
        # But wait, I modified diagnostics.py to accept `bald_opt_stats`.
        # So I should return these stats. But `select_test` signature is `(best_test, best_score)`.
        # Changing return signature breaks `ActiveLearner.step` unless I update it too.
        # Let's update `ActiveLearner.step` too.
        
        return best_test, best_score, diag_stats_list
