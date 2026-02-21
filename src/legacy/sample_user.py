
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from active_learning.src.legacy.feasibility_checker import FeasibilityChecker
from active_learning.src.config import DEVICE

class UserGenerator:
    """
    Generates synthetic "Ground Truth" users based on configuration.
    Generates random limits within global anatomical bounds and random bumps.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 joint_names: List[str],
                 anatomical_limits: Dict[str, Tuple[float, float]],
                 pairs: List[Tuple[str, str]]):
        self.config = config
        
        # Look for user_generation in deprecated first (new config structure), then root
        deprecated_cfg = config.get('deprecated', {}).get('user_generation', {})
        root_cfg = config.get('user_generation', {})
        
        # Merge or prefer deprecated (since legacy.yaml puts it there)
        # If deprecated has content, use it. Else use root.
        self.gen_config = deprecated_cfg if deprecated_cfg else root_cfg
        
        self.device = DEVICE

        self.joint_names = joint_names
        self.anatomical_limits = anatomical_limits
        self.pairs = pairs

        # Set random seed if provided
        seed = self.gen_config.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def generate_user(self) -> Tuple[Dict, Dict, FeasibilityChecker]:
        """
        Generates a user with random limits within anatomical bounds and random bumps.
        Returns:
            true_limits: Dict of joint limits
            true_bumps: Dict of bump constraints
            checker: FeasibilityChecker instance
        """
        # 1. Generate Joint Limits
        true_limits = {}
        width_frac_range = self.gen_config.get('limit_width_fraction', [0.5, 0.8])

        for name in self.joint_names:
            anat_min, anat_max = self.anatomical_limits[name]
            anat_width = anat_max - anat_min

            # Sample width
            target_width = anat_width * np.random.uniform(*width_frac_range)

            # Sample center such that the box fits within anatomical limits
            slack = anat_width - target_width
            offset = np.random.uniform(0, slack)

            l_val = anat_min + offset
            u_val = l_val + target_width

            true_limits[name] = (l_val, u_val)

        # 2. Generate Bumps
        true_bumps = {}
        bump_prob = self.gen_config.get('bump_prob', 0.5)
        max_bumps = self.gen_config.get('max_bumps_per_pair', 1)
        bump_params = self.gen_config.get('bump_params', {})

        for pair in self.pairs:
            if np.random.random() > bump_prob:
                continue

            n_bumps = np.random.randint(1, max_bumps + 1)
            pair_bumps = []

            # Get ranges for this pair
            j1, j2 = pair
            l1, u1 = true_limits[j1]
            l2, u2 = true_limits[j2]

            for _ in range(n_bumps):
                # Sample bump parameters

                # Center (mu) - sample within the user's box
                mu1 = np.random.uniform(l1, u1)
                mu2 = np.random.uniform(l2, u2)

                # Lengthscale (ls)
                ls_range = bump_params.get('ls_range', [0.3, 0.8])
                ls1 = np.random.uniform(*ls_range)
                ls2 = np.random.uniform(*ls_range)

                # Alpha (height)
                alpha_range = bump_params.get('alpha_range', [0.8, 1.5])
                alpha = np.random.uniform(*alpha_range)

                # Theta (rotation)
                theta_range = bump_params.get('theta_range', [-0.5, 0.5])
                theta = np.random.uniform(*theta_range)

                pair_bumps.append({
                    'mu': [mu1, mu2],
                    'ls': [ls1, ls2],
                    'alpha': alpha,
                    'theta': theta,
                    'R': None # Will be computed by checker if needed
                })

            if pair_bumps:
                true_bumps[pair] = {
                    'bumps': pair_bumps,
                    'box': {
                        'q1_range': true_limits[j1],
                        'q2_range': true_limits[j2]
                    }
                }

        # 3. Create Checker
        checker = FeasibilityChecker(
            joint_limits=true_limits,
            pairwise_constraints=true_bumps,
            config=self.config
        )

        return true_limits, true_bumps, checker

if __name__ == "__main__":
    # Test the generator
    from active_learning.src.config import load_config
    from active_learning.src.legacy.prior_generation import PriorGenerator
    config = load_config()

    prior_gen = PriorGenerator(config)

    gen = UserGenerator(
        config,
        prior_gen.joint_names,
        prior_gen.anatomical_limits,
        prior_gen.pairs
    )
    limits, bumps, checker = gen.generate_user()

    print("Generated User:")
    print("Limits:", limits)
    print("Bumps:", bumps.keys())
