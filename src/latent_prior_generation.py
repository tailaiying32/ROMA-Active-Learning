import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.config import DEVICE, get_bounds_from_config


def project_to_anatomical_limits(
    z: torch.Tensor,
    decoder,
    limits: torch.Tensor, 
    device: str = DEVICE,
    verbose: bool = True
) -> torch.Tensor:
    """
    Project latent code z so that its decoded joint limits reflect the anatomical limits.
    Uses gradient descent to minimize violation.
    
    Args:
        z: Latent code (latent_dim,)
        decoder: LevelSetDecoder model
        limits: Tensor of shape (n_joints, 2) [min, max]
        device: torch device
        
    Returns:
        z_projected: Projected latent code
    """
    # Clone and detach to start fresh optimization
    z_opt = z.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z_opt], lr=0.1)
    
    # limits are (n_joints, 2)
    min_vals = limits[:, 0]
    max_vals = limits[:, 1]
    
    # Optimization loop
    for i in range(100):
        optimizer.zero_grad()
        
        # Decode current z
        # decode_from_embedding expects (batch, latent_dim), returns (batch, n_joints) for lower/upper
        lowers, uppers, _, _, _ = decoder.decode_from_embedding(z_opt.unsqueeze(0))
        
        curr_lower = lowers.squeeze(0)
        curr_upper = uppers.squeeze(0)
        
        # Calculate violation loss
        # We want: curr_lower >= min_vals  =>  loss += ReLU(min_vals - curr_lower)
        # We want: curr_upper <= max_vals  =>  loss += ReLU(curr_upper - max_vals)
        
        loss_lower = torch.nn.functional.relu(min_vals - curr_lower).sum()
        loss_upper = torch.nn.functional.relu(curr_upper - max_vals).sum()
        
        total_loss = loss_lower + loss_upper
        
        if total_loss.item() < 1e-4:
            break
            
        total_loss.backward()
        optimizer.step()
        
    if i == 99 and total_loss.item() > 1e-2 and verbose:
        print(f"  Warning: Prior projection did not fully converge. Final loss: {total_loss.item():.4f}")
    elif i > 0 and verbose:
        print(f"  Projected prior to anatomical limits in {i+1} iterations.")

    return z_opt.detach()


def optimize_latent_to_match_limits(
    decoder,
    target_lower: torch.Tensor,
    target_upper: torch.Tensor,
    device: str = DEVICE,
    verbose: bool = True
) -> torch.Tensor:
    """
    Find a latent code z that decodes to the specified target limits.
    
    Args:
        decoder: LevelSetDecoder model
        target_lower: Target lower bounds (n_joints,)
        target_upper: Target upper bounds (n_joints,)
        
    Returns:
        z_opt: Optimized latent code
    """
    latent_dim = decoder.latent_dim if hasattr(decoder, 'latent_dim') else 32
    # Initialize from zero mean
    z_opt = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z_opt], lr=0.1)
    
    for i in range(200):
        optimizer.zero_grad()
        
        lowers, uppers, _, _, _ = decoder.decode_from_embedding(z_opt.unsqueeze(0))
        curr_lower = lowers.squeeze(0)
        curr_upper = uppers.squeeze(0)
        
        # MSE loss matching targets
        loss = torch.nn.functional.mse_loss(curr_lower, target_lower) + \
               torch.nn.functional.mse_loss(curr_upper, target_upper)
               
        if loss.item() < 1e-5:
            break
            
        loss.backward()
        optimizer.step()
        
    if verbose:
        print(f"  Optimized prior mean to targets in {i+1} iterations (Loss: {loss.item():.6f})")
    return z_opt.detach()



class LatentPriorGenerator:
    """
    Handles the initialization of the prior distribution in latent space.
    Much simpler than joint space - just needs latent_dim, mean, and log_std.
    """

    def __init__(self, config: Dict[str, Any], decoder, verbose: bool = True):
        """
        Args:
            config: Configuration dictionary
            decoder: Decoder model to map latent codes to outputs
            verbose: Whether to print initialization logs
        """
        self.config = config
        self.decoder = decoder
        self.verbose = verbose
        self.device = DEVICE
        self.prior_config = config.get('prior', {})
        self.latent_config = config.get('latent', {})
        
        # Create a dedicated generator for reproducibility
        seed = config.get('seed', None)
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            self.generator = None

        # Get latent dimension from config or decoder
        self.latent_dim = self.latent_config.get('latent_dim', None)
        if self.latent_dim is None and hasattr(decoder, 'latent_dim'):
            self.latent_dim = decoder.latent_dim
        if self.latent_dim is None:
            raise ValueError("latent_dim must be specified in config['latent']['latent_dim'] or decoder.latent_dim")

    @property
    def joint_names(self) -> List[str]:
        """Get joint names from config."""
        return self.prior_config.get('joint_names', [])

    @property
    def anatomical_limits(self) -> Dict[str, Tuple[float, float]]:
        """Get anatomical limits from config (converted to radians if needed)."""
        limits_raw = self.prior_config.get('anatomical_limits', {})
        units = self.prior_config.get('units', 'degrees')

        if units == 'degrees':
            return {
                joint: (np.deg2rad(lims[0]), np.deg2rad(lims[1]))
                for joint, lims in limits_raw.items()
            }
        return dict(limits_raw)

    def get_prior(self, ground_truth_z: Optional[torch.Tensor] = None, embeddings: Optional[torch.Tensor] = None) -> LatentUserDistribution:
        """
        Returns an initialized LatentUserDistribution object representing the prior.

        If ground_truth_z is provided, generates a prior distribution initialized
        near the ground truth but with noise and high uncertainty.

        Args:
            ground_truth_z: Ground truth latent code (optional), shape (latent_dim,)
            embeddings: All training embeddings (optional), shape (N, latent_dim).
                        Used when use_population_mean_prior is enabled.

        Returns:
            LatentUserDistribution representing the prior
        """
        # Get initialization parameters from config
        init_std = self.prior_config.get('init_std', 1.0)
        mean_noise_std = self.prior_config.get('mean_noise_std', 0.2)
        use_population_mean = self.prior_config.get('use_population_mean_prior', False)

        # Full covariance prior (computed in population mean branch if enabled)
        precision_matrix = None
        cov_cholesky = None

        # --- Population mean prior (highest priority) ---
        if use_population_mean and embeddings is not None:
            base_z = embeddings.mean(dim=0).to(self.device)
            if self.verbose:
                print(f"  Using population mean prior (mean of {len(embeddings)} embeddings)")

            # Perturb with noise (using seeded generator for reproducibility)
            if self.generator is not None:
                noise = torch.randn(self.latent_dim, device=self.device, generator=self.generator) * mean_noise_std
            else:
                noise = torch.randn(self.latent_dim, device=self.device) * mean_noise_std
            mean = base_z + noise
            log_std = torch.full((self.latent_dim,), np.log(init_std), device=self.device)

            if self.verbose:
                print(f"\nGenerating prior:")
                print(f"  Mean noise std: {mean_noise_std}")
                print(f"  Init std: {init_std}")
                print(f"  Perturbation norm: {noise.norm().item():.4f}")

            # Compute full empirical covariance if enabled
            if self.prior_config.get('use_empirical_covariance', False):
                centered = embeddings.to(self.device) - base_z.unsqueeze(0)
                cov = (centered.T @ centered) / (len(embeddings) - 1)
                cov += 1e-6 * torch.eye(self.latent_dim, device=self.device)
                precision_matrix = torch.linalg.inv(cov)
                cov_cholesky = torch.linalg.cholesky(cov)
                if self.verbose:
                    cond = cov.diag().max() / cov.diag().min()
                    print(f"  Using empirical covariance prior (condition number: {cond:.2f})")

        elif ground_truth_z is None:
            if use_population_mean:
                if self.verbose:
                    print("  Warning: use_population_mean_prior=True but no embeddings provided. Falling back.")

            # Check for healthy prior mode (anatomical limits)
            if self.prior_config.get('use_anatomical_limit_prior', False):
                if self.verbose:
                    print(f"  Generating prior centered at default anatomical limits (healthy person)...")
                bounds = get_bounds_from_config(self.config, self.device)
                if bounds is not None:
                    anat_lower = bounds[:, 0]
                    anat_upper = bounds[:, 1]
                    # Optimize z to match these targets
                    mean = optimize_latent_to_match_limits(self.decoder, anat_lower, anat_upper, self.device, verbose=self.verbose)
                else:
                    if self.verbose:
                        print("  Warning: use_anatomical_limit_prior requested but no bounds found. Using zero mean.")
                    mean = torch.zeros(self.latent_dim, device=self.device)
            else:
                # Standard prior: N(0, init_std^2 * I)
                mean = torch.zeros(self.latent_dim, device=self.device)

            log_std = torch.full((self.latent_dim,), np.log(init_std), device=self.device)
        else:
            # Prior near ground truth with noise OR centered at specific percentile

            # Check for fixed percentile range option
            # e.g. [0.25, 0.75]
            fixed_range = self.prior_config.get('fixed_percentile_range', None)

            if fixed_range is not None:
                if self.verbose:
                    print(f"  Generating prior centered at {fixed_range} of anatomical limits...")
                bounds = get_bounds_from_config(self.config, self.device) # (n_joints, 2)
                if bounds is not None:
                     anat_lower = bounds[:, 0]
                     anat_upper = bounds[:, 1]
                     anat_range = anat_upper - anat_lower

                     target_lower = anat_lower + fixed_range[0] * anat_range
                     target_upper = anat_lower + fixed_range[1] * anat_range

                     # Optimize z to match these targets
                     base_z = optimize_latent_to_match_limits(self.decoder, target_lower, target_upper, self.device, verbose=self.verbose)
                else:
                     if self.verbose:
                        print("  Warning: fixed_percentile_range requested but no bounds found. Using ground truth.")
                     base_z = ground_truth_z.to(self.device)
            else:
                base_z = ground_truth_z.to(self.device)

            # Perturb with noise (using seeded generator for reproducibility)
            if self.generator is not None:
                noise = torch.randn(self.latent_dim, device=self.device, generator=self.generator) * mean_noise_std
            else:
                noise = torch.randn(self.latent_dim, device=self.device) * mean_noise_std
            mean = base_z + noise

            # --- Enforce Anatomical Limits on Prior ---
            # Disabled for now as it messes up prior generation
            if False:
                bounds = get_bounds_from_config(self.config, self.device)
                if bounds is not None:
                    mean = project_to_anatomical_limits(mean, self.decoder, bounds, self.device, verbose=self.verbose)

            log_std = torch.full((self.latent_dim,), np.log(init_std), device=self.device)

            if self.verbose:
                print(f"\nGenerating prior:")
                print(f"  Mean noise std: {mean_noise_std}")
                print(f"  Init std: {init_std}")
                print(f"  Perturbation norm: {noise.norm().item():.4f}")


        prior = LatentUserDistribution(
            latent_dim=self.latent_dim,
            decoder=self.decoder,
            mean=mean,
            log_std=log_std,
            precision_matrix=precision_matrix,
            cov_cholesky=cov_cholesky,
            device=self.device
        )

        return prior

    def get_posterior_init(self, prior: LatentUserDistribution) -> LatentUserDistribution:
        """
        Returns a copy of the prior to use as the initial posterior.

        Args:
            prior: The prior distribution

        Returns:
            LatentUserDistribution initialized as a copy of the prior
        """
        return LatentUserDistribution(
            latent_dim=self.latent_dim,
            decoder=self.decoder,
            mean=prior.mean.clone(),
            log_std=prior.log_std.clone(),
            device=self.device
        )
