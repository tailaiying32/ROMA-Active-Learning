from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from active_learning.src.config import DEVICE


class UserDistribution:
    '''
    Distribution of users, parameterized by independent Gaussian lower and upper bounds.

    Joint limit parameterization:
        lower ~ N(lower_mean, lower_std)
        upper ~ N(upper_mean, upper_std)
    '''
    def __init__(
            self,
            joint_names: List[str],
            pairs: List[Tuple[str, str]] = None,
            anatomical_limits: Dict[str, Tuple[float, float]] = None,
            config: Dict = None
            ):
        '''
        Initialize the user distribution.

        Args:
            joint_names: list of joint names
            pairs: list of joint pairs (tuples) that may have constraints
            anatomical_limits: dict of {joint_name: (min, max)} for uniform sampling
            config: configuration dictionary (expects 'prior' section)
        '''

        self.joint_names = joint_names
        self.pairs = pairs if pairs is not None else []
        self.anatomical_limits = anatomical_limits
        self.device = DEVICE
        self.config = config or {}

        # Load from config
        prior_config = self.config.get('prior', {})
        self.use_uniform_prior = prior_config.get('use_uniform_prior', False)
        # init_std now refers to the std of the bound location itself (in radians)
        self.init_std = prior_config.get('init_std', 2.0)

        # Load anatomical limits if not provided explicitly
        if self.anatomical_limits is None and 'anatomical_limits' in prior_config:
            self.anatomical_limits = prior_config['anatomical_limits']

            # Convert to radians if specified in degrees
            if prior_config.get('units', 'radians') == 'degrees':
                converted_limits = {}
                for joint, (min_val, max_val) in self.anatomical_limits.items():
                    converted_limits[joint] = (np.deg2rad(min_val), np.deg2rad(max_val))
                self.anatomical_limits = converted_limits

        self.n_joints = len(joint_names)
        self.init_parameters()

    def init_parameters(self):
        '''
        Initialize parameters for each joint and pair. (std is stored as log_std for stability)

        self.params = {
            'joint_limits': {
                joint_name: {
                    'lower_mean': float, 'lower_log_std': float,
                    'upper_mean': float, 'upper_log_std': float
                }, ...
            },
            'bumps': { ... }
        }
        '''

        self.params = {
            'joint_limits': {},
            'bumps': {}
        }

        # Initialize joint limits with (lower, upper) parameterization
        for joint in self.joint_names:
            # reasonable defaults if no anatomical info: [-pi/2, pi/2]
            default_lower = -np.pi / 2
            default_upper = np.pi / 2

            if self.anatomical_limits and joint in self.anatomical_limits:
                default_lower, default_upper = self.anatomical_limits[joint]

            self.params['joint_limits'][joint] = {
                'lower_mean': torch.tensor(float(default_lower), device=self.device),
                'lower_log_std': torch.tensor(np.log(self.init_std), device=self.device),
                'upper_mean': torch.tensor(float(default_upper), device=self.device),
                'upper_log_std': torch.tensor(np.log(self.init_std), device=self.device),
            }

        # Initialize bumps (empty by default, use add_bump_prior to populate)
        for pair in self.pairs:
            self.params['bumps'][pair] = []

    def add_bump_prior(self, pair, mu_mean, mu_std, ls_mean, ls_std, alpha_mean, alpha_std, theta_mean=0.0, theta_std=0.1):
        """Add a Gaussian bump prior to a specific pair."""
        if pair not in self.params['bumps']:
            self.params['bumps'][pair] = []

        self.params['bumps'][pair].append({
            'mu': {
                'mean': torch.tensor(mu_mean, device=self.device, dtype=torch.float32),
                'log_std': torch.tensor(np.log(mu_std), device=self.device, dtype=torch.float32)
            },
            'ls': {
                'mean': torch.tensor(ls_mean, device=self.device, dtype=torch.float32),
                'log_std': torch.tensor(np.log(ls_std), device=self.device, dtype=torch.float32)
            },
            'alpha': {
                'mean': torch.tensor(alpha_mean, device=self.device, dtype=torch.float32),
                'log_std': torch.tensor(np.log(alpha_std), device=self.device, dtype=torch.float32)
            },
            'theta': {
                'mean': torch.tensor(theta_mean, device=self.device, dtype=torch.float32),
                'log_std': torch.tensor(np.log(theta_std), device=self.device, dtype=torch.float32)
            }
        })

    def sample(self, n_samples, temperature=1.0, return_list=False):
        '''
        Sample n virtual users from the posterior using reparameterization trick.

        Args:
            n_samples: Number of samples to draw
            temperature: Float scalar to scale the standard deviation of valid samples.
                         Higher temperature (>1.0) encourages exploration by producing
                         more diverse samples (inflating variance). Used for BALD.
            return_list: If True, returns a list of dictionaries (legacy format).
                         If False (default), returns a dictionary of batched tensors.

        Returns:
            If return_list=False:
                Dict with:
                - 'joint_limits': (n_samples, n_joints, 2)
                - 'bumps': {pair: {'mu': (n_samples, n_bumps, 2), ...}}
            If return_list=True:
                List of theta dictionaries (legacy format)
        '''
        # 1. Sample Joint Limits (Batch)
        l_means = []
        l_log_stds = []
        u_means = []
        u_log_stds = []
        for joint in self.joint_names:
            p = self.params['joint_limits'][joint]
            l_means.append(p['lower_mean'])
            l_log_stds.append(p['lower_log_std'])
            u_means.append(p['upper_mean'])
            u_log_stds.append(p['upper_log_std'])

        l_mean = torch.stack(l_means) # (n_joints,)
        l_std = torch.exp(torch.stack(l_log_stds)) * temperature
        u_mean = torch.stack(u_means)
        u_std = torch.exp(torch.stack(u_log_stds)) * temperature

        # Sample all at once: (n_samples, n_joints)
        L = l_mean + l_std * torch.randn(n_samples, self.n_joints, device=self.device)
        U = u_mean + u_std * torch.randn(n_samples, self.n_joints, device=self.device)
        joint_limits = torch.stack([L, U], dim=-1) # (n_samples, n_joints, 2)

        # 2. Sample Bumps (Batch per pair)
        bumps_dict = {}
        for pair, bump_priors in self.params['bumps'].items():
            if not bump_priors:
                continue
            
            n_bumps = len(bump_priors)
            
            mu_means = torch.stack([bp['mu']['mean'] for bp in bump_priors]) # (n_bumps, 2)
            mu_stds = torch.exp(torch.stack([bp['mu']['log_std'] for bp in bump_priors])) * temperature
            
            ls_means = torch.stack([bp['ls']['mean'] for bp in bump_priors])
            ls_stds = torch.exp(torch.stack([bp['ls']['log_std'] for bp in bump_priors])) * temperature
            
            alpha_means = torch.stack([bp['alpha']['mean'] for bp in bump_priors]) # (n_bumps,)
            alpha_stds = torch.exp(torch.stack([bp['alpha']['log_std'] for bp in bump_priors])) * temperature
            
            theta_means = torch.stack([bp['theta']['mean'] for bp in bump_priors])
            theta_stds = torch.exp(torch.stack([bp['theta']['log_std'] for bp in bump_priors])) * temperature

            # Sample: (n_samples, n_bumps, ...)
            mu = mu_means + mu_stds * torch.randn(n_samples, n_bumps, 2, device=self.device)
            ls = torch.abs(ls_means + ls_stds * torch.randn(n_samples, n_bumps, 2, device=self.device))
            alpha = alpha_means + alpha_stds * torch.randn(n_samples, n_bumps, device=self.device)
            theta_rot = theta_means + theta_stds * torch.randn(n_samples, n_bumps, device=self.device)

            # Batched rotation matrices: (n_samples, n_bumps, 2, 2)
            c = torch.cos(theta_rot)
            s = torch.sin(theta_rot)
            R = torch.stack([
                torch.stack([c, -s], dim=-1),
                torch.stack([s, c], dim=-1)
            ], dim=-2)

            bumps_dict[pair] = {
                'mu': mu,
                'ls': ls,
                'alpha': alpha,
                'theta': theta_rot,
                'R': R
            }

        if not return_list:
            return {
                'joint_limits': joint_limits,
                'bumps': bumps_dict
            }

        # Legacy format conversion
        samples = []
        for i in range(n_samples):
            theta = {
                'joint_limits': {name: (joint_limits[i, j, 0], joint_limits[i, j, 1]) 
                                 for j, name in enumerate(self.joint_names)},
                'pairwise_constraints': {}
            }
            for pair, data in bumps_dict.items():
                pair_bumps = []
                for b in range(len(bump_priors)):
                    pair_bumps.append({
                        'mu': data['mu'][i, b],
                        'ls': data['ls'][i, b],
                        'alpha': data['alpha'][i, b],
                        'theta': data['theta'][i, b],
                        'R': data['R'][i, b]
                    })
                theta['pairwise_constraints'][pair] = {'bumps': pair_bumps}
            samples.append(theta)
        return samples

    def get_test_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get reasonable bounds for test points based on distribution means.

        Returns:
            (lower_bounds, upper_bounds) tensors of shape (n_joints,)
        """
        lowers = []
        uppers = []

        for joint in self.joint_names:
            p = self.params['joint_limits'][joint]
            # Use mean of L and U distributions
            lower_mean = p['lower_mean']
            upper_mean = p['upper_mean']

            lowers.append(lower_mean)
            uppers.append(upper_mean)

        return torch.stack(lowers), torch.stack(uppers)

    def get_limit_stats(self, joint: str) -> Dict[str, float]:
        """
        Get interpretable statistics for a joint's limit distribution.

        Returns:
            Dict with lower_mean, lower_std, upper_mean, upper_std
        """
        p = self.params['joint_limits'][joint]

        lower_mean = p['lower_mean'].item()
        lower_std = torch.exp(p['lower_log_std']).item()

        upper_mean = p['upper_mean'].item()
        upper_std = torch.exp(p['upper_log_std']).item()

        return {
            'lower_mean': lower_mean,
            'lower_std': lower_std,
            'upper_mean': upper_mean,
            'upper_std': upper_std
        }
