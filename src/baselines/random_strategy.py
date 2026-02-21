"""
Random acquisition strategy for active learning baseline.
"""

import torch
import numpy as np
from active_learning.src.config import DEVICE, create_generator

class RandomStrategy:
    """
    Random acquisition strategy.
    Selects test points uniformly at random within the provided bounds.
    """

    def __init__(self, config: dict = None, isolated_seed: int = None):
        """
        Args:
            config: Configuration dictionary
            isolated_seed: If provided, creates a dedicated generator with this seed that is independent of config['seed']
        """
        self.config = config or {}
        self.device = DEVICE

        # Create generator for reproducibility
        # If isolated_seed is provided, we use it directly to create a fresh generator
        # This ensures that even if other components consume from the global stream or config-based stream,
        # this strategy remains deterministic and isolated.
        if isolated_seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(isolated_seed)
        else:
            # Fallback to config-based shared generator (legacy behavior)
            self.generator = create_generator(self.config, self.device)

    def select_test(self, bounds: torch.Tensor, verbose: bool = False, test_history: list = None, **kwargs) -> tuple:
        """
        Select a random test point within bounds.

        Args:
            bounds: Tensor of shape (n_joints, 2) with [lower, upper] bounds
            verbose: Print progress (not used for random)

        Returns:
            (test_point, score) tuple where score is dummy 0.0
        """
        n_joints = bounds.shape[0]

        # Ensure bounds are on the correct device
        bounds = bounds.to(self.device)
        lower = bounds[:, 0]
        upper = bounds[:, 1]

        # Sample uniformly: lower + rand * (upper - lower)
        test_point = lower + torch.rand(n_joints, device=self.device, generator=self.generator) * (upper - lower)

        if verbose:
            print(f"Random selection: {test_point.cpu().numpy()}")

        return test_point, 0.0
