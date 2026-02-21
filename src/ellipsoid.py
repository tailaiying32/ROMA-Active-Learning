"""
Ellipsoid geometry utilities for G-BALD.

Provides:
- Minimum-volume enclosing ellipsoid (MVEE) fitting via online covariance
- Mahalanobis distance computation
- Boundary penalty calculation

Reference: "Bayesian Active Learning by Disagreements: A Geometric Perspective"
           Cao & Tsang, 2021 (arXiv:2105.02543)
"""

import torch
from typing import Optional


class AdaptiveEllipsoid:
    """
    Incrementally-updated ellipsoid for core-set diversity.

    Uses Welford's online algorithm for covariance estimation.
    Falls back to spherical (Euclidean) distance when insufficient points.

    The ellipsoid is defined as:
        E = {x : (x - μ)ᵀ A (x - μ) ≤ 1}

    Where A = Σ⁻¹ is the precision matrix (inverse covariance).
    """

    def __init__(self, dim: int, device: str = 'cuda',
                 regularization: float = 1e-4,
                 min_points_for_ellipsoid: int = None):
        """
        Args:
            dim: Dimensionality of the space (e.g., 4 for joint space)
            device: Torch device
            regularization: Ridge regularization for covariance inversion
            min_points_for_ellipsoid: Minimum points before using ellipsoid.
                                      Default: dim + 1
        """
        self.dim = dim
        self.device = device
        self.reg = regularization
        self.min_points = min_points_for_ellipsoid if min_points_for_ellipsoid is not None else (dim + 1)

        # Online statistics (Welford's algorithm)
        self.n = 0
        self.mean = torch.zeros(dim, device=device)
        self.M2 = torch.zeros(dim, dim, device=device)

        # Cached ellipsoid parameters
        self._A = None  # Precision matrix (inverse covariance)
        self._A_valid = False

    def update(self, point: torch.Tensor):
        """
        Update ellipsoid with a new point using Welford's online algorithm.

        This maintains running mean and covariance without storing all points.

        Args:
            point: New point tensor of shape (dim,)
        """
        point = point.to(self.device)
        if point.dim() > 1:
            point = point.squeeze()

        self.n += 1
        delta = point - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = point - self.mean
        self.M2 = self.M2 + torch.outer(delta, delta2)
        self._A_valid = False  # Invalidate cache

    def get_precision_matrix(self) -> Optional[torch.Tensor]:
        """
        Get the precision matrix A = Σ⁻¹.

        Returns:
            A: Precision matrix of shape (dim, dim), or None if insufficient points
        """
        if self.n < self.min_points:
            return None

        if not self._A_valid:
            # Covariance from Welford's M2
            cov = self.M2 / (self.n - 1)

            # Regularized inverse for numerical stability
            cov_reg = cov + self.reg * torch.eye(self.dim, device=self.device)
            self._A = torch.linalg.inv(cov_reg)
            self._A_valid = True

        return self._A

    def get_covariance_matrix(self) -> Optional[torch.Tensor]:
        """
        Get the covariance matrix Σ.

        Returns:
            Σ: Covariance matrix of shape (dim, dim), or None if insufficient points
        """
        if self.n < self.min_points:
            return None
        return self.M2 / (self.n - 1)

    def mahalanobis_distance_sq(self, x: torch.Tensor,
                                 y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute squared Mahalanobis distance: (x-y)ᵀ A (x-y)

        Args:
            x: Points of shape (N, dim) or (dim,)
            y: Reference points of shape (M, dim) or (dim,), or None for center

        Returns:
            If y is None: distances to center, shape (N,) or scalar
            If y is given: pairwise distances, shape (N, M)
        """
        A = self.get_precision_matrix()
        if A is None:
            raise ValueError("Not enough points for Mahalanobis distance")

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if y is None:
            # Distance to ellipsoid center
            diff = x - self.mean  # (N, dim)
            # (x - μ)ᵀ A (x - μ)
            result = torch.einsum('ni,ij,nj->n', diff, A, diff)
            return result.squeeze() if result.shape[0] == 1 else result
        else:
            if y.dim() == 1:
                y = y.unsqueeze(0)
            # Pairwise distances: (N, M)
            diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, dim)
            return torch.einsum('nmi,ij,nmj->nm', diff, A, diff)

    def euclidean_distance_sq(self, x: torch.Tensor,
                               y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fallback Euclidean distance when ellipsoid not available.

        Same signature as mahalanobis_distance_sq.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if y is None:
            diff = x - self.mean
            result = (diff ** 2).sum(dim=-1)
            return result.squeeze() if result.shape[0] == 1 else result
        else:
            if y.dim() == 1:
                y = y.unsqueeze(0)
            diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, dim)
            return (diff ** 2).sum(dim=-1)  # (N, M)

    def distance_sq(self, x: torch.Tensor,
                    y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute squared distance, using ellipsoid if available, else Euclidean.

        This is the main method to use - it automatically selects the
        appropriate distance metric based on available data.
        """
        if self.get_precision_matrix() is not None:
            return self.mahalanobis_distance_sq(x, y)
        else:
            return self.euclidean_distance_sq(x, y)

    def distance(self, x: torch.Tensor,
                 y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute distance (sqrt of squared distance)."""
        return torch.sqrt(self.distance_sq(x, y) + 1e-10)

    @property
    def center(self) -> torch.Tensor:
        """Return ellipsoid center (mean of points)."""
        return self.mean

    @property
    def is_ellipsoid_valid(self) -> bool:
        """Check if we have enough points for ellipsoid geometry."""
        return self.n >= self.min_points

    def reset(self):
        """Reset ellipsoid state."""
        self.n = 0
        self.mean = torch.zeros(self.dim, device=self.device)
        self.M2 = torch.zeros(self.dim, self.dim, device=self.device)
        self._A = None
        self._A_valid = False
