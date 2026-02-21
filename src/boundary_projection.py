"""
Marginal Envelope Boundary Projection.

Computes the range of possible boundary positions when marginalizing over
non-displayed dimensions, providing a more accurate representation of
4D feasibility boundaries in 2D projections.

Instead of showing a single boundary slice (which depends heavily on fixed
values for non-displayed dimensions), this shows:
- Inner boundary (h_max = 0): Points where NO configuration is feasible
- Outer boundary (h_min = 0): Points where ALL configurations are feasible
- The band between represents uncertainty due to marginalized dimensions
"""

import torch
from typing import Dict, Optional


def compute_marginal_envelope(
    checker,
    grid_j1: torch.Tensor,
    grid_j2: torch.Tensor,
    idx1: int,
    idx2: int,
    bounds: torch.Tensor,
    n_samples: int = 50,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Compute marginal envelope of feasibility boundary.

    For each (j1, j2) point, samples many (j3, j4, ...) combinations and
    computes envelope statistics showing the range of possible h values.

    Args:
        checker: Feasibility checker with logit_value(points) method
        grid_j1: (R,) values for joint 1
        grid_j2: (R,) values for joint 2
        idx1: Index of joint 1 in full joint vector
        idx2: Index of joint 2 in full joint vector
        bounds: (n_joints, 2) full bounds for all joints
        n_samples: Number of samples for marginalization (higher = smoother but slower)
        device: Device for computation

    Returns:
        {
            'h_min': (R, R) - min h over marginal samples (outer envelope)
            'h_max': (R, R) - max h over marginal samples (inner envelope)
            'h_mean': (R, R) - mean h over marginal samples
            'p_feasible': (R, R) - probability of feasibility
        }
    """
    R = len(grid_j1)
    n_joints = bounds.shape[0]

    # Create meshgrid for (j1, j2)
    J1, J2 = torch.meshgrid(grid_j1, grid_j2, indexing='ij')  # (R, R)

    # Identify marginal dimensions (all except idx1 and idx2)
    marginal_indices = [i for i in range(n_joints) if i not in (idx1, idx2)]

    if len(marginal_indices) == 0:
        # Only 2 joints, no marginalization needed
        points = torch.zeros(R * R, n_joints, device=device)
        points[:, idx1] = J1.flatten()
        points[:, idx2] = J2.flatten()
        h = checker.logit_value(points).view(R, R)
        return {
            'h_min': h,
            'h_max': h,
            'h_mean': h,
            'p_feasible': (h > 0).float(),
        }

    # Sample all marginal values at once: (n_samples, len(marginal_indices))
    marginal_samples = torch.zeros(n_samples, len(marginal_indices), device=device)
    for i, idx in enumerate(marginal_indices):
        low, high = bounds[idx, 0], bounds[idx, 1]
        marginal_samples[:, i] = low + torch.rand(n_samples, device=device) * (high - low)

    # Build all points: (n_samples, R, R, n_joints)
    points = torch.zeros(n_samples, R, R, n_joints, device=device)
    points[:, :, :, idx1] = J1.unsqueeze(0)  # Broadcast across samples
    points[:, :, :, idx2] = J2.unsqueeze(0)
    for i, idx in enumerate(marginal_indices):
        points[:, :, :, idx] = marginal_samples[:, i].view(n_samples, 1, 1)

    # Reshape for batched evaluation: (n_samples * R * R, n_joints)
    points_flat = points.view(-1, n_joints)

    # Evaluate h in batches to avoid OOM
    batch_size = 10000
    h_flat = torch.zeros(points_flat.shape[0], device=device)

    for start in range(0, points_flat.shape[0], batch_size):
        end = min(start + batch_size, points_flat.shape[0])
        h_flat[start:end] = checker.logit_value(points_flat[start:end])

    h = h_flat.view(n_samples, R, R)  # (n_samples, R, R)

    return {
        'h_min': h.min(dim=0).values,
        'h_max': h.max(dim=0).values,
        'h_mean': h.mean(dim=0),
        'p_feasible': (h > 0).float().mean(dim=0),
    }


def compute_marginal_envelope_for_decoder(
    decoder,
    z: torch.Tensor,
    grid_j1: torch.Tensor,
    grid_j2: torch.Tensor,
    idx1: int,
    idx2: int,
    bounds: torch.Tensor,
    n_samples: int = 50,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Compute marginal envelope using a decoder and latent code.

    Wrapper around compute_marginal_envelope that creates a checker
    from the decoder and latent code z.

    Args:
        decoder: LevelSetDecoder model
        z: (latent_dim,) latent code (e.g., posterior mean)
        grid_j1: (R,) values for joint 1
        grid_j2: (R,) values for joint 2
        idx1: Index of joint 1 in full joint vector
        idx2: Index of joint 2 in full joint vector
        bounds: (n_joints, 2) full bounds for all joints
        n_samples: Number of samples for marginalization
        device: Device for computation

    Returns:
        Same as compute_marginal_envelope
    """
    from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker

    # Create a checker from the decoder and latent code
    checker = LatentFeasibilityChecker(decoder, z, device=device)

    return compute_marginal_envelope(
        checker=checker,
        grid_j1=grid_j1,
        grid_j2=grid_j2,
        idx1=idx1,
        idx2=idx2,
        bounds=bounds,
        n_samples=n_samples,
        device=device
    )
