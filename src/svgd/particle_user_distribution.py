import torch
from active_learning.src.config import DEVICE
from active_learning.src.latent_user_distribution import LatentUserDistribution


class ParticleUserDistribution(LatentUserDistribution):
    '''
    Non-parametric user distribution represented by a bank of particles.
    Used for SVGD (Stein Variational Gradient Descent).
    '''

    def __init__(
            self,
            latent_dim,
            decoder,
            n_particles=20,
            init_particles=None,
            device=DEVICE
    ):
        '''
        Initialize the particle distribution.

        Args:
            latent_dim: dimension of the latent space
            decoder: decoder model to map latent codes to outputs
            n_particles: number of particles to use
            init_particles: optional tensor of shape (n_particles, latent_dim) to initialize with.
                            If None, initializes from standard normal.
            device: torch device
        '''
        # Initialize parent to set basic attributes
        # We pass dummy mean/std as they won't be used, but helps satisfy basic contract if accessed
        super().__init__(
            latent_dim=latent_dim,
            decoder=decoder,
            mean=torch.zeros(latent_dim, device=device),
            log_std=torch.zeros(latent_dim, device=device),
            device=device
        )

        self.n_particles = n_particles
        
        if init_particles is not None:
            if init_particles.shape != (n_particles, latent_dim):
                raise ValueError(f"init_particles shape {init_particles.shape} mismatch with (n_particles={n_particles}, latent_dim={latent_dim})")
            self.particles = init_particles.to(device).clone()
        else:
            # Initialize from standard normal prior
            self.particles = torch.randn(n_particles, latent_dim, device=device)

    @property
    def mean(self):
        """Dynamic mean of the particles."""
        return self.particles.mean(dim=0)

    @mean.setter
    def mean(self, value):
        """Dummy setter to satisfy LatentUserDistribution.__init__"""
        pass

    @property
    def log_std(self):
        """Dynamic log_std of the particles."""
        # Add eps for stability
        return torch.log(self.particles.std(dim=0) + 1e-6)

    @log_std.setter
    def log_std(self, value):
        """Dummy setter to satisfy LatentUserDistribution.__init__"""
        pass

    def sample(self, n_samples, temperature=1.0, generator=None):
        '''
        Sample n_samples latent codes from the particle bank.
        Since this is non-parametric, we bootstrap (sample with replacement) from the particles.

        Args:
            n_samples: number of samples to draw
            temperature: (Ignored for particles)
            generator: optional torch.Generator
        Returns:
            samples: tensor of shape (n_samples, latent_dim)
        '''
        # Sample indices with replacement
        indices = torch.randint(
            0, 
            self.n_particles, 
            (n_samples,), 
            device=self.device, 
            generator=generator
        )
        return self.particles[indices]

    def get_particles(self):
        """Return the current particle bank."""
        return self.particles
