import torch
from active_learning.src.latent_bald import LatentBALD

class ParticleBALD(LatentBALD):
    """
    Particle-Based BALD acquisition function.
    Uses the fixed set of optimized particles to estimate uncertainty/disagreement.
    """

    def compute_score(self, test: torch.Tensor, zs: list = None, decoded_params: tuple = None, iteration: int = None) -> torch.Tensor:
        """
        Compute BALD score using the particle bank.
        
        Args:
            test: Test point tensor
            zs: Optional override. If None, uses ALL particles in the bank.
            decoded_params: Optional pre-decoded RBF params.
            iteration: Current iteration index.
        """
        # If no samples provided, use the full particle bank
        if zs is None and decoded_params is None:
            # Deterministic evaluation using current particles
            zs = self.posterior.get_particles()
            
        # Call parent implementation which handles the entropy math
        # LatentBALD logic: 
        # 1. Decode zs -> logits
        # 2. probs = sigmoid(logits)
        # 3. H(mean(p)) - mean(H(p))
        return super().compute_score(test, zs=zs, decoded_params=decoded_params, iteration=iteration)
