from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import itertools
import torch
from active_learning.src.legacy.user_distribution import UserDistribution
from active_learning.src.config import DEVICE

class PriorGenerator:
    """
    Handles the initialization of the prior distribution and problem structure
    (joints, limits, pairs) from the configuration.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = DEVICE
        self.prior_config = config.get('prior', {})

        self.joint_names = self._determine_joint_names()
        self.anatomical_limits = self._determine_anatomical_limits()
        self.pairs = self._determine_pairs()

    def _determine_joint_names(self) -> List[str]:
        return self.prior_config['joint_names']

    def _determine_anatomical_limits(self) -> Dict[str, Tuple[float, float]]:
        if 'anatomical_limits' in self.prior_config:
            limits = self.prior_config['anatomical_limits']
            # Convert to radians if specified in degrees
            if self.prior_config.get('units', 'radians') == 'degrees':
                converted_limits = {}
                for joint, (min_val, max_val) in limits.items():
                    converted_limits[joint] = (np.deg2rad(min_val), np.deg2rad(max_val))
                return converted_limits
            return limits
        return {}

    def _determine_pairs(self) -> List[Tuple[str, str]]:
        # Check if pairs are defined in config
        if 'pair_names' in self.prior_config:
             return [tuple(p) for p in self.prior_config['pair_names']]

        # Default to all combinations if not specified
        return list(itertools.combinations(self.joint_names, 2))

    def get_prior(self, true_limits: Optional[Dict] = None, true_bumps: Optional[Dict] = None) -> UserDistribution:
        """
        Returns an initialized UserDistribution object representing the prior.

        If true_limits and true_bumps are provided, generates a prior distribution initialized
        near the ground truth but with noise and high uncertainty (Simulates an LLM/RAG estimation).

        Args:
            true_limits: Ground truth joint limits (optional)
            true_bumps: Ground truth bumps (optional)
            noise_std: Standard deviation of the noise added to the means
            prior_std: Initial standard deviation (uncertainty) of the prior
        """
        prior = UserDistribution(
            joint_names=self.joint_names,
            pairs=self.pairs,
            anatomical_limits=self.anatomical_limits,
            config=self.config
        )

        if true_limits is None or true_bumps is None:
            return prior

        # 1. Noisy Joint Limits (using lower, upper parameterization)
        for joint, (true_lower, true_upper) in true_limits.items():
            # Use global anatomical range for scaling noise (not user's range)
            # This ensures consistent noise magnitude regardless of user's specific limits
            if joint in self.anatomical_limits:
                anat_min, anat_max = self.anatomical_limits[joint]
                global_range = np.abs(anat_max - anat_min)
            else:
                # Fallback to user range if no anatomical limits defined
                global_range = np.abs(true_upper - true_lower)

            # Add noise to lower and upper bounds directly
            noise_std = self.prior_config.get('mean_noise_std', 0.2)

            # Sample noise
            noise_lower = np.random.normal(0, noise_std * global_range)
            noise_upper = np.random.normal(0, noise_std * global_range)

            noisy_lower = true_lower + noise_lower
            noisy_upper = true_upper + noise_upper

            # Ensure the resulting bounds stay within anatomical limits
            if joint in self.anatomical_limits:
                noisy_lower = np.clip(noisy_lower, anat_min, anat_max)
                noisy_upper = np.clip(noisy_upper, anat_min, anat_max)

            # Update prior parameters with new parameterization
            # init_std is now the std of the location of the bounds
            init_std = self.prior_config.get('init_std', 2.0)

            # Assign to prior parameters
            # std is log_std in the storage
            prior.params['joint_limits'][joint]['lower_mean'] = torch.tensor(float(noisy_lower), device=self.device, dtype=torch.float32)
            prior.params['joint_limits'][joint]['lower_log_std'] = torch.tensor(np.log(init_std), device=self.device, dtype=torch.float32)

            prior.params['joint_limits'][joint]['upper_mean'] = torch.tensor(float(noisy_upper), device=self.device, dtype=torch.float32)
            prior.params['joint_limits'][joint]['upper_log_std'] = torch.tensor(np.log(init_std), device=self.device, dtype=torch.float32)

        # 2. Noisy Bumps
        # Clear existing bumps (if any default ones existed)
        prior.params['bumps'] = {pair: [] for pair in self.pairs}

        for pair, bump_info in true_bumps.items():
            if pair not in prior.params['bumps']:
                prior.params['bumps'][pair] = []

            for bump in bump_info['bumps']:
                # Extract true values
                true_mu = bump['mu']
                true_ls = bump['ls']
                true_alpha = bump['alpha']
                true_theta = bump['theta']

                # Add noise
                noisy_mu = [m + np.random.normal(0, noise_std) for m in true_mu]
                noisy_ls = [l + np.random.normal(0, noise_std * 0.5) for l in true_ls] # Less noise on lengthscale
                noisy_alpha = true_alpha + np.random.normal(0, noise_std * 0.5)
                noisy_theta = true_theta + np.random.normal(0, noise_std * 0.5)

                # Use init_std (scaled) for the prior width
                bump_std = init_std  # Use the same general uncertainty scale

                # Add to prior using add_bump_prior helper
                prior.add_bump_prior(
                    pair,
                    mu_mean=noisy_mu, mu_std=[bump_std, bump_std],
                    ls_mean=noisy_ls, ls_std=[bump_std * 0.5, bump_std * 0.5],
                    alpha_mean=noisy_alpha, alpha_std=bump_std * 0.5,
                    theta_mean=noisy_theta, theta_std=bump_std * 0.5
                )

        return prior
