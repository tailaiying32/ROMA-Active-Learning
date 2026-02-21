import torch
from active_learning.src.config import DEVICE


class LatentUserDistribution:
    '''
    latent user distribution parametrized by a mean and std for each dimension in latent space
    '''

    def __init__(
            self,
            latent_dim,
            decoder,
            mean=None,
            log_std=None,
            precision_matrix=None,
            cov_cholesky=None,
            device=DEVICE
    ):
        '''
        Initialize the latent user distribution.

        Args:
            latent_dim: dimension of the latent space
            decoder: decoder model to map latent codes to outputs
            mean: initial mean tensor of shape (latent_dim,)
            log_std: initial log std tensor of shape (latent_dim,)
            precision_matrix: optional (latent_dim, latent_dim) precision matrix (inverse covariance)
            cov_cholesky: optional (latent_dim, latent_dim) lower-triangular Cholesky factor of covariance
            device: torch device
        '''

        self.latent_dim = latent_dim
        self.decoder = decoder
        self.device = device

        # Initialize mean and log_std
        self.mean = mean if mean is not None else torch.zeros(latent_dim, device=device)
        self.log_std = log_std if log_std is not None else torch.zeros(latent_dim, device=device)

        # Optional full covariance parameterization
        self.precision_matrix = precision_matrix  # (D, D) or None
        self.cov_cholesky = cov_cholesky          # (D, D) lower-triangular or None

    def sample(self, n_samples, temperature=1.0, generator=None):
        '''
        Sample n_samples latent codes from the distribution using reparameterization trick.

        Args:
            n_samples: number of samples to draw
            temperature: scaling factor for std (higher = more diverse samples)
            generator: optional torch.Generator for reproducibility (None = use global state)
        Returns:
            samples: tensor of shape (n_samples, latent_dim)
        '''
        eps = torch.randn(n_samples, self.latent_dim, device=self.device, generator=generator)  # (n_samples, latent_dim)
        if self.cov_cholesky is not None:
            # Full covariance: N(μ, t²Σ) = μ + t·L·ε where L·Lᵀ = Σ
            samples = self.mean.unsqueeze(0) + temperature * (eps @ self.cov_cholesky.T)
        else:
            # Diagonal covariance
            std = torch.exp(self.log_std) * temperature  # (latent_dim,)
            samples = self.mean.unsqueeze(0) + std.unsqueeze(0) * eps  # (n_samples, latent_dim)
        return samples