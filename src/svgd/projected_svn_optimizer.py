"""
Projected Stein Variational Newton (pSVN) Optimizer.

Uses eigenvectors of the averaged Hessian (second-order information) to:
1. Identify directions of highest posterior curvature
2. Project particles to low-dimensional subspace
3. Apply Newton-preconditioned SVGD updates in subspace
4. Lift back to full space

Reference: Chen & Ghattas (2019) "Projected Stein Variational Newton"
https://arxiv.org/abs/1901.08659
"""

import torch
from typing import Tuple, Optional, Callable, Dict


class ProjectedSVN:
    """
    Projected Stein Variational Newton Optimizer.

    Key differences from ProjectedSVGD:
    - Uses Hessian eigenvectors (curvature) instead of score covariance (gradient spread)
    - Applies Newton preconditioning (scales by inverse eigenvalues)
    - Typically converges in fewer iterations
    """

    def __init__(
        self,
        n_eigenvectors: int = 10,
        n_random_slices: int = 10,
        kernel_width: float = None,
        kernel_type: str = 'imq',
        use_gauss_newton: bool = True,
        hessian_update_freq: int = 5,
        regularization: float = 1e-4,
        include_prior_hessian: bool = True,
    ):
        """
        Args:
            n_eigenvectors: Number of Hessian eigenvectors to use for projection.
            n_random_slices: Additional random directions for coverage of low-curvature dims.
            kernel_width: Fixed 1D kernel width. None = median heuristic (recommended).
            kernel_type: 'rbf' or 'imq' (inverse multiquadric).
            use_gauss_newton: If True, approximate Hessian with Fisher (g @ g^T).
                              If False, requires hessian_fn to be passed to step().
            hessian_update_freq: Recompute Hessian eigenbasis every N inner iterations.
            regularization: Tikhonov regularization for (λ + ε)^{-1} stability.
            include_prior_hessian: Whether to add prior precision to Hessian estimate.
        """
        self.n_eigenvectors = n_eigenvectors
        self.n_random_slices = n_random_slices
        self.kernel_width = kernel_width
        self.kernel_type = kernel_type
        self.use_gauss_newton = use_gauss_newton
        self.hessian_update_freq = hessian_update_freq
        self.regularization = regularization
        self.include_prior_hessian = include_prior_hessian

        # Cached eigenbasis (updated periodically)
        self._cached_eigenvectors = None  # (D, r)
        self._cached_eigenvalues = None   # (r,)
        self._last_hessian_update = -1

    def reset_cache(self):
        """Reset cached Hessian eigenbasis (call at start of new posterior update)."""
        self._cached_eigenvectors = None
        self._cached_eigenvalues = None
        self._last_hessian_update = -1

    def _compute_gauss_newton_hessian(
        self,
        log_prob_grad: torch.Tensor,
        prior_precision: torch.Tensor = None,
        kl_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Gauss-Newton approximation of the averaged Hessian.

        H_GN ≈ (1/K) Σ_k [g_k @ g_k^T] + kl_weight * Σ_prior^{-1}

        This approximates the Hessian of the negative log-posterior.

        Args:
            log_prob_grad: (K, D) gradients of log p(z|D) for each particle
            prior_precision: (D, D) prior precision matrix, or None for identity
            kl_weight: Weight for prior term

        Returns:
            H_avg: (D, D) averaged Hessian approximation
        """
        K, D = log_prob_grad.shape
        device = log_prob_grad.device

        # Empirical Fisher from likelihood gradients
        # Note: We use negative gradients since H = -∇²log p = ∇²(-log p)
        # But for Fisher, we use |∇log p|² which is always positive
        H_fisher = log_prob_grad.T @ log_prob_grad / K  # (D, D)

        # Add prior Hessian (precision matrix)
        if self.include_prior_hessian and prior_precision is not None:
            H_avg = H_fisher + kl_weight * prior_precision
        else:
            H_avg = H_fisher

        return H_avg

    def _compute_hessian_eigenbasis(
        self,
        H_avg: torch.Tensor,
        n_eigenvectors: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top eigenvectors of averaged Hessian.

        Args:
            H_avg: (D, D) averaged Hessian matrix
            n_eigenvectors: Number of top eigenvectors to extract

        Returns:
            eigenvalues: (r,) top eigenvalues (descending)
            eigenvectors: (D, r) corresponding eigenvectors
        """
        # Symmetric eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(H_avg)

        # eigh returns ascending order, flip to descending
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # Take top r
        r = min(n_eigenvectors, len(eigenvalues))
        return eigenvalues[:r], eigenvectors[:, :r]

    def _compute_1d_svgd(
        self,
        s_proj: torch.Tensor,
        g_proj: torch.Tensor,
        K: int,
        h: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute 1D SVGD terms for a single direction.

        Args:
            s_proj: (K,) projected particle positions
            g_proj: (K,) projected (Newton-preconditioned) gradients
            K: number of particles
            h: bandwidth squared, or None for median heuristic
            device: torch device

        Returns:
            term1: (K,) gradient term
            term2: (K,) repulsion term
            h_val: bandwidth used
        """
        # Pairwise differences
        diff_1d = s_proj.unsqueeze(1) - s_proj.unsqueeze(0)  # (K, K)
        dist_sq_1d = diff_1d ** 2

        # Median heuristic bandwidth
        if h is None:
            median_dist = torch.median(dist_sq_1d)
            h_val = median_dist / torch.log(torch.tensor(K + 1.0, device=device))
            h_val = torch.clamp(h_val, min=1e-6)
        else:
            h_val = torch.tensor(h, device=device)

        # Kernel computation
        scaled_dist = dist_sq_1d / h_val

        if self.kernel_type == 'imq':
            base = 1.0 + scaled_dist
            k_mat = base.pow(-0.5)
            grad_factor = -(1.0 / h_val) * base.pow(-1.5)
        else:  # rbf
            k_mat = torch.exp(-scaled_dist)
            grad_factor = -(2.0 / h_val) * k_mat

        # SVGD terms
        term1 = k_mat.T @ g_proj / K  # (K,)
        term2 = (grad_factor * diff_1d).sum(dim=0) / K  # (K,)

        return term1, term2, h_val

    def _compute_1d_svgd_batched(
        self,
        s_proj: torch.Tensor,
        g_proj: torch.Tensor,
        K: int,
        h: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched 1D SVGD for multiple directions simultaneously.

        Args:
            s_proj: (K, S) projected particles for all slices
            g_proj: (K, S) projected gradients for all slices
            K: number of particles
            h: fixed bandwidth squared, or None for median heuristic
            device: torch device

        Returns:
            term1: (K, S) gradient terms
            term2: (K, S) repulsion terms
            h_vals: (S,) bandwidth values
        """
        S = s_proj.shape[1]

        # Pairwise differences: (K, K, S)
        diff_1d = s_proj.unsqueeze(1) - s_proj.unsqueeze(0)
        dist_sq_1d = diff_1d ** 2

        # Bandwidth computation
        if h is None:
            dist_sq_flat = dist_sq_1d.permute(2, 0, 1).reshape(S, -1)
            median_dist = torch.median(dist_sq_flat, dim=1).values
            log_K = torch.log(torch.tensor(K + 1.0, device=device))
            h_vals = torch.clamp(median_dist / log_K, min=1e-6)
        else:
            h_vals = torch.full((S,), h, device=device)

        # Broadcast h_vals to (1, 1, S)
        h_broadcast = h_vals.view(1, 1, S)
        scaled_dist = dist_sq_1d / h_broadcast

        # Kernel computation
        if self.kernel_type == 'imq':
            base = 1.0 + scaled_dist
            k_mat = base.pow(-0.5)
            grad_factor = -(1.0 / h_broadcast) * base.pow(-1.5)
        else:  # rbf
            k_mat = torch.exp(-scaled_dist)
            grad_factor = -(2.0 / h_broadcast) * k_mat

        # SVGD terms via einsum
        term1 = torch.einsum('jik,jk->ik', k_mat, g_proj) / K
        term2 = (grad_factor * diff_1d).sum(dim=0) / K

        return term1, term2, h_vals

    def step(
        self,
        particles: torch.Tensor,
        log_prob_grad: torch.Tensor,
        prior_precision: torch.Tensor = None,
        kl_weight: float = 1.0,
        iteration: int = 0,
        return_diagnostics: bool = False
    ) -> torch.Tensor:
        """
        Compute the Projected SVN update direction.

        Args:
            particles: (K, D) current particle positions
            log_prob_grad: (K, D) gradients of log p(z|D)
            prior_precision: (D, D) prior precision matrix (optional)
            kl_weight: Weight for prior in Hessian computation
            iteration: Current inner iteration (for Hessian update scheduling)
            return_diagnostics: If True, return (phi, diagnostics_dict)

        Returns:
            phi: (K, D) update direction
            diagnostics (optional): Dict with debug info
        """
        K, D = particles.shape
        device = particles.device

        h_fixed = self.kernel_width ** 2 if self.kernel_width is not None else None

        # --- Phase 1: Update Hessian eigenbasis (periodic) ---
        should_update = (
            self._cached_eigenvectors is None or
            iteration - self._last_hessian_update >= self.hessian_update_freq
        )

        if should_update:
            if self.use_gauss_newton:
                H_avg = self._compute_gauss_newton_hessian(
                    log_prob_grad, prior_precision, kl_weight
                )
            else:
                # Would need external Hessian computation
                # Fall back to Gauss-Newton
                H_avg = self._compute_gauss_newton_hessian(
                    log_prob_grad, prior_precision, kl_weight
                )

            eigenvalues, eigenvectors = self._compute_hessian_eigenbasis(
                H_avg, self.n_eigenvectors
            )

            self._cached_eigenvalues = eigenvalues
            self._cached_eigenvectors = eigenvectors
            self._last_hessian_update = iteration

        V = self._cached_eigenvectors  # (D, r)
        eigs = self._cached_eigenvalues  # (r,)
        r = V.shape[1]

        # --- Phase 2: Newton preconditioning factors ---
        # Inverse eigenvalues with regularization: (λ + ε)^{-1}
        inv_eigs = 1.0 / (eigs + self.regularization)  # (r,)

        # --- Phase 3: Project particles and gradients ---
        particles_proj = particles @ V  # (K, r)
        grad_proj = log_prob_grad @ V   # (K, r)

        # --- Phase 4: Compute SVGD in projected space with Newton preconditioning ---
        # Precondition gradients: g_newton = Λ^{-1} g
        grad_newton = grad_proj * inv_eigs.unsqueeze(0)  # (K, r)

        # Batched 1D SVGD
        term1_proj, term2_proj, h_vals_proj = self._compute_1d_svgd_batched(
            particles_proj, grad_newton, K, h_fixed, device
        )

        # Newton precondition both terms (as per SVN paper)
        # phi_1d = (term1 + term2) * inv_eig
        phi_proj = (term1_proj + term2_proj) * inv_eigs.unsqueeze(0)  # (K, r)

        # Lift back to full space
        phi_hessian = phi_proj @ V.T  # (K, D)

        # --- Phase 5: Random slices for coverage (no Newton preconditioning) ---
        phi_random = torch.zeros_like(particles)
        h_vals_random = []

        if self.n_random_slices > 0:
            random_dirs = []
            for _ in range(self.n_random_slices):
                r_dir = torch.randn(D, device=device)
                r_dir = r_dir / r_dir.norm()
                random_dirs.append(r_dir)
            random_dirs = torch.stack(random_dirs, dim=1)  # (D, n_random)

            s_random = particles @ random_dirs  # (K, n_random)
            g_random = log_prob_grad @ random_dirs  # (K, n_random)

            term1_rand, term2_rand, h_vals_rand = self._compute_1d_svgd_batched(
                s_random, g_random, K, h_fixed, device
            )

            phi_1d_rand = term1_rand + term2_rand  # (K, n_random)
            phi_random = phi_1d_rand @ random_dirs.T  # (K, D)
            h_vals_random = h_vals_rand.tolist()

        # --- Phase 6: Combine and normalize ---
        total_slices = r + self.n_random_slices
        phi = (phi_hessian + phi_random) / total_slices

        if return_diagnostics:
            # Compute diagnostics
            diff_full = particles.unsqueeze(1) - particles.unsqueeze(0)
            dist_sq_full = torch.sum(diff_full ** 2, dim=-1)
            mask = ~torch.eye(K, dtype=bool, device=device)

            h_vals_all = h_vals_proj.tolist() + h_vals_random

            diag = {
                # Standard keys (for compatibility with svgd_vi.py logger)
                'term1_norm': phi.norm().item(),
                'term2_norm': 0.0,
                'term1_per_particle': phi.norm(dim=-1).mean().item(),
                'term2_per_particle': 0.0,
                'h': sum(h_vals_all) / len(h_vals_all) if h_vals_all else 0.0,
                'repulsive_scale': 1.0,
                'mean_pairwise_dist': dist_sq_full.mean().item(),
                'min_pairwise_dist_offdiag': dist_sq_full[mask].min().item() if K > 1 else 0.0,
                'kernel_mean': 0.0,

                # Projected-specific
                'n_projected_dirs': r,
                'n_random_dirs': self.n_random_slices,
                'n_slices': total_slices,

                # pSVN-specific
                'eigenvalue_max': eigs[0].item() if len(eigs) > 0 else 0.0,
                'eigenvalue_min': eigs[-1].item() if len(eigs) > 0 else 0.0,
                'eigenvalue_condition': (eigs[0] / (eigs[-1] + 1e-10)).item() if len(eigs) > 0 else 0.0,
                'inv_eig_max': inv_eigs[0].item() if len(inv_eigs) > 0 else 0.0,
                'inv_eig_min': inv_eigs[-1].item() if len(inv_eigs) > 0 else 0.0,
                'hessian_updated': should_update,

                # Norms
                'phi_hessian_norm': phi_hessian.norm().item(),
                'phi_random_norm': phi_random.norm().item() if self.n_random_slices > 0 else 0.0,
                'term1_proj_norm': term1_proj.norm().item(),
                'term2_proj_norm': term2_proj.norm().item(),
            }
            return phi, diag

        return phi
