import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.config import DEVICE, create_generator


class LatentUserGenerator:
    """
    Generates synthetic "Ground Truth" users in latent space.
    Simply samples a latent code z from a distribution.
    """

    def __init__(
            self,
            config: Dict[str, Any],
            decoder,
            latent_dim: int
    ):
        """
        Args:
            config: Configuration dictionary
            decoder: Decoder model to map latent codes to outputs
            latent_dim: Dimension of the latent space
        """
        self.config = config
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.device = DEVICE

        self.gen_config = config.get('user_generation', {})

        # Set random seed if provided (legacy behavior)
        seed = self.gen_config.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Create dedicated generator for reproducibility
        self.generator = create_generator(config, self.device)

    def generate_user(
            self,
            z: Optional[torch.Tensor] = None,
            sampling_std: float = 1.0
    ) -> Tuple[torch.Tensor, LatentFeasibilityChecker]:
        """
        Generates a user with a random latent code.

        Args:
            z: Optional pre-specified latent code. If None, samples from N(0, sampling_std^2 * I)
            sampling_std: Standard deviation for sampling z (if z not provided)

        Returns:
            ground_truth_z: The latent code representing the user, shape (latent_dim,)
            checker: LatentFeasibilityChecker instance for querying feasibility
        """
        if z is not None:
            ground_truth_z = z.to(self.device)
        else:
            # Sample from standard normal (or scaled)
            ground_truth_z = torch.randn(self.latent_dim, device=self.device, generator=self.generator) * sampling_std

        # Create checker
        checker = LatentFeasibilityChecker(
            decoder=self.decoder,
            z=ground_truth_z,
            device=self.device
        )

        return ground_truth_z, checker

    def generate_user_from_dataset(
            self,
            dataset_zs: torch.Tensor,
            index: Optional[int] = None
    ) -> Tuple[torch.Tensor, LatentFeasibilityChecker]:
        """
        Generates a user by selecting a latent code from a dataset.

        Args:
            dataset_zs: Tensor of latent codes from training data, shape (N, latent_dim)
            index: Optional index to select. If None, samples randomly.

        Returns:
            ground_truth_z: The latent code representing the user, shape (latent_dim,)
            checker: LatentFeasibilityChecker instance for querying feasibility
        """
        if index is None:
            index = np.random.randint(0, len(dataset_zs))

        ground_truth_z = dataset_zs[index].to(self.device)

        checker = LatentFeasibilityChecker(
            decoder=self.decoder,
            z=ground_truth_z,
            device=self.device
        )

        return ground_truth_z, checker


if __name__ == "__main__":
    # Test the generator (requires a decoder)
    print("LatentUserGenerator requires a decoder model to test.")
    print("Usage:")
    print("  gen = LatentUserGenerator(config, decoder, latent_dim=8)")
    print("  z, checker = gen.generate_user()")
