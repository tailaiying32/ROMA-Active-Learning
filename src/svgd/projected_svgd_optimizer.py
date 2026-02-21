import torch

# Toggle for vectorized implementation (set to False to use original loop version)
_USE_VECTORIZED_PSVGD = True


class ProjectedSVGD:
    """
    Projected Stein Variational Gradient Descent Optimizer.

    Combines ideas from Projected SVGD (informed direction selection),
    Sliced SVGD (1D kernels), and Matrix SVGD (eigenvalue preconditioning).

    Uses eigenvectors of the score covariance to identify directions where
    the posterior varies most, projects particles onto those 1D directions,
    computes 1D kernel SVGD updates per direction, weights by eigenvalue,
    and projects back. Remaining slices use random directions (as in
    standard Sliced SVGD) to maintain coverage of low-variance dimensions.
    """

    def __init__(self, n_slices=20, variance_threshold=0.95,
                 kernel_width=None, kernel_type='rbf', max_eigenweight=3.0,
                 eigen_smoothing=0.5):
        """
        Args:
            n_slices: Total number of 1D directions per step (projected + random).
            variance_threshold: Fraction of total score variance to capture with
                                eigenvector directions (0.0-1.0). Determines adaptive r.
            kernel_width: Fixed 1D kernel width. If None, uses median
                          heuristic per slice (recommended).
            kernel_type: Kernel function to use: 'rbf' or 'imq'.
            eigen_smoothing: Exponent alpha for smoothing eigenvalue weights.
                             0.0 = uniform weighting (stable)
                             0.5 = sqrt weighting (balanced)
                             1.0 = linear weighting (original, potentially unstable)
            max_eigenweight: Maximum eigenvalue weight for any single projected
                             direction. Prevents needle updates when the spectrum
                             is highly peaked. Set to 0 to disable clamping.
        """
        self.n_slices = n_slices
        self.variance_threshold = variance_threshold
        self.kernel_width = kernel_width
        self.kernel_type = kernel_type
        self.max_eigenweight = max_eigenweight
        self.eigen_smoothing = eigen_smoothing

    def _compute_1d_svgd(self, s_proj, g_proj, K, h, device):
        """Compute 1D kernel, term1, and term2 for a single direction."""
        diff_1d = s_proj.unsqueeze(1) - s_proj.unsqueeze(0)  # (K, K)
        dist_sq_1d = diff_1d ** 2  # (K, K)

        # Median heuristic bandwidth
        if h is None:
            median_dist = torch.median(dist_sq_1d)
            h_val = median_dist / torch.log(torch.tensor(K + 1.0, device=device))
            if h_val == 0:
                h_val = torch.tensor(1.0, device=device)
        else:
            h_val = torch.tensor(h, device=device)

        # Kernel + gradient factor
        scaled_dist = dist_sq_1d / h_val
        if self.kernel_type == 'imq':
            base = 1.0 + scaled_dist
            k_mat = base.pow(-0.5)
            grad_factor = -(1.0 / h_val) * base.pow(-1.5)
        else:
            k_mat = torch.exp(-scaled_dist)
            grad_factor = -(2.0 / h_val) * k_mat

        # 1D SVGD terms
        term1_1d = k_mat.t() @ g_proj / K  # (K,)
        term2_1d = (grad_factor * diff_1d).sum(dim=0) / K  # (K,)

        return term1_1d, term2_1d, h_val

    def _compute_1d_svgd_batched(self, s_proj, g_proj, K, h, device):
        """
        Compute 1D SVGD for multiple slices simultaneously.

        Args:
            s_proj: (K, S) projected particles for all slices
            g_proj: (K, S) projected gradients for all slices
            K: number of particles
            h: fixed bandwidth squared, or None for median heuristic
            device: torch device

        Returns:
            term1_1d: (K, S) gradient terms
            term2_1d: (K, S) repulsion terms
            h_vals: (S,) bandwidth values used
        """
        S = s_proj.shape[1]

        # Pairwise differences for all slices: (K, K, S)
        diff_1d = s_proj.unsqueeze(1) - s_proj.unsqueeze(0)
        dist_sq_1d = diff_1d ** 2

        # Bandwidth computation
        if h is None:
            # Reshape to (S, K*K) to compute median per slice
            dist_sq_flat = dist_sq_1d.permute(2, 0, 1).reshape(S, -1)
            median_dist = torch.median(dist_sq_flat, dim=1).values  # (S,)

            log_K_plus_1 = torch.log(torch.tensor(K + 1.0, device=device))
            h_vals = median_dist / log_K_plus_1  # (S,)

            # Handle zero median (degenerate case)
            h_vals = torch.where(h_vals == 0, torch.ones_like(h_vals), h_vals)
        else:
            h_vals = torch.full((S,), h, device=device)

        # Broadcast h_vals to (1, 1, S) for element-wise ops with (K, K, S)
        h_broadcast = h_vals.view(1, 1, S)
        scaled_dist = dist_sq_1d / h_broadcast

        # Kernel computation
        if self.kernel_type == 'imq':
            base = 1.0 + scaled_dist
            k_mat = base.pow(-0.5)  # (K, K, S)
            grad_factor = -(1.0 / h_broadcast) * base.pow(-1.5)
        else:  # rbf
            k_mat = torch.exp(-scaled_dist)  # (K, K, S)
            grad_factor = -(2.0 / h_broadcast) * k_mat

        # term1: k_mat.T @ g_proj for each slice
        # k_mat[i, j, s] is kernel between particle i and j for slice s
        # We need sum_j k_mat[j, i, s] * g_proj[j, s] (the transpose)
        # einsum 'jis,js->is' with k_mat indexed as (j, i, s)
        # Since k_mat is (i, j, s), we use 'jik,jk->ik' to read first dim as j
        term1_1d = torch.einsum('jik,jk->ik', k_mat, g_proj) / K  # (K, S)

        # term2: sum over first dim of (grad_factor * diff_1d)
        term2_1d = (grad_factor * diff_1d).sum(dim=0) / K  # (K, S)

        return term1_1d, term2_1d, h_vals

    def step(self, particles, log_prob_grad, return_diagnostics=False):
        """
        Compute the Projected SVGD update direction (phi).

        Same interface as SVGD.step() / SlicedSVGD.step():
            particles: Tensor of shape (K, D).
            log_prob_grad: Tensor of shape (K, D).
            return_diagnostics: If True, return (phi, diagnostics_dict).

        Returns:
            phi: Tensor of shape (K, D).
            diagnostics (optional): Dict with aggregate and projected-specific stats.
        """
        K, D = particles.shape
        device = particles.device

        phi = torch.zeros_like(particles)  # (K, D) accumulator
        h_fixed = self.kernel_width ** 2 if self.kernel_width is not None else None

        # --- Phase 1: Score covariance eigendecomposition ---
        C = log_prob_grad.t() @ log_prob_grad / K  # (D, D)
        eigenvalues, eigenvectors = torch.linalg.eigh(C)  # ascending order

        # Flip to descending
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # --- Phase 2: Adaptive r selection ---
        total_variance = eigenvalues.sum()
        if total_variance < 1e-12:
            # No score variance — fall back to pure random slicing
            r = 0
        else:
            cumulative = eigenvalues.cumsum(0) / total_variance
            above = (cumulative >= self.variance_threshold).nonzero(as_tuple=True)[0]
            r = (above[0].item() + 1) if len(above) > 0 else D

        r = min(r, self.n_slices)
        n_random = self.n_slices - r

        # Eigenvalue weights for preconditioning
        if r > 0:
            selected_eigenvalues = eigenvalues[:r]
            # Apply smoothing: w_k \propto \lambda_k^\alpha
            smoothed_eigs = selected_eigenvalues.pow(self.eigen_smoothing)
            
            # Avoid division by zero if all smoothed_eigs are close to 0
            sum_smoothed = smoothed_eigs.sum()
            if sum_smoothed < 1e-9:
                 # Fallback to uniform if degenerate
                eigenvalue_weights = torch.ones_like(smoothed_eigs)
            else:
                eigenvalue_weights = r * smoothed_eigs / sum_smoothed

            if self.max_eigenweight > 0:
                eigenvalue_weights = eigenvalue_weights.clamp(max=self.max_eigenweight)

        # --- Phases 3-4: Compute SVGD updates for all directions ---
        if _USE_VECTORIZED_PSVGD:
            # Vectorized: process all slices simultaneously
            directions_list = []
            weights_list = []

            # Eigenvector directions (first r)
            if r > 0:
                eigen_dirs = eigenvectors[:, :r]  # (D, r)
                directions_list.append(eigen_dirs)
                weights_list.append(eigenvalue_weights)  # (r,)

            # Random directions (remaining n_random)
            # Generate one-by-one to match loop version's random number ordering exactly
            # This ensures identical results with same seed on both CPU and CUDA
            if n_random > 0:
                random_dirs_list = []
                for _ in range(n_random):
                    r_dir = torch.randn(D, device=device)
                    r_dir = r_dir / r_dir.norm()
                    random_dirs_list.append(r_dir)
                random_dirs = torch.stack(random_dirs_list, dim=1)  # (D, n_random)
                directions_list.append(random_dirs)
                weights_list.append(torch.ones(n_random, device=device))

            # Combine into single matrices
            all_directions = torch.cat(directions_list, dim=1)  # (D, S)
            all_weights = torch.cat(weights_list)  # (S,)

            # Project particles and gradients onto all directions at once
            s_proj = particles @ all_directions  # (K, S)
            g_proj = log_prob_grad @ all_directions  # (K, S)

            # Batched 1D SVGD computation
            term1_1d, term2_1d, h_vals = self._compute_1d_svgd_batched(
                s_proj, g_proj, K, h_fixed, device
            )

            # Apply weights and reconstruct in D-space
            phi_1d = (term1_1d + term2_1d) * all_weights.unsqueeze(0)  # (K, S)
            phi = phi_1d @ all_directions.T  # (K, D)

            # Collect diagnostics from batched results
            if return_diagnostics:
                h_values = h_vals.tolist()
                term1_1d_norms = term1_1d.norm(dim=0).tolist()
                term2_1d_norms = term2_1d.norm(dim=0).tolist()
        else:
            # Original loop version
            if return_diagnostics:
                h_values = []
                term1_1d_norms = []
                term2_1d_norms = []

            # Phase 3: Projected directions (r eigenvector slices)
            for k in range(r):
                v = eigenvectors[:, k]  # (D,)

                s_proj = particles @ v  # (K,)
                g_proj = log_prob_grad @ v  # (K,)

                term1_1d, term2_1d, h_val = self._compute_1d_svgd(
                    s_proj, g_proj, K, h_fixed, device)

                # Eigenvalue preconditioning
                phi_1d = (term1_1d + term2_1d) * eigenvalue_weights[k]
                phi += torch.outer(phi_1d, v)

                if return_diagnostics:
                    h_values.append(h_val.item())
                    term1_1d_norms.append(term1_1d.norm().item())
                    term2_1d_norms.append(term2_1d.norm().item())

            # Phase 4: Random directions (n_random slices, no weighting)
            for _ in range(n_random):
                r_dir = torch.randn(D, device=device)
                r_dir = r_dir / r_dir.norm()

                s_proj = particles @ r_dir
                g_proj = log_prob_grad @ r_dir

                term1_1d, term2_1d, h_val = self._compute_1d_svgd(
                    s_proj, g_proj, K, h_fixed, device)

                phi_1d = term1_1d + term2_1d
                phi += torch.outer(phi_1d, r_dir)

                if return_diagnostics:
                    h_values.append(h_val.item())
                    term1_1d_norms.append(term1_1d.norm().item())
                    term2_1d_norms.append(term2_1d.norm().item())

        # --- Phase 5: Normalize ---
        phi /= self.n_slices

        if return_diagnostics:
            # Full pairwise distances for standard diagnostics
            diff_full = particles.unsqueeze(1) - particles.unsqueeze(0)
            dist_sq_full = torch.sum(diff_full ** 2, dim=-1)
            mask = ~torch.eye(K, dtype=bool, device=device)

            variance_explained = (
                eigenvalues[:r].sum().item() / total_variance.item()
                if r > 0 and total_variance > 1e-12 else 0.0
            )

            diag = {
                # Standard keys (required by svgd_vi.py inner-loop logger)
                'term1_norm': phi.norm().item(),
                'term2_norm': 0.0,
                'term1_per_particle': phi.norm(dim=-1).mean().item(),
                'term2_per_particle': 0.0,
                'h': sum(h_values) / len(h_values) if h_values else 0.0,
                'repulsive_scale': 1.0,
                'mean_pairwise_dist': dist_sq_full.mean().item(),
                'min_pairwise_dist_offdiag': dist_sq_full[mask].min().item() if K > 1 else 0.0,
                'kernel_mean': 0.0,
                # Sliced-compatible keys
                'n_slices': self.n_slices,
                'h_min': min(h_values) if h_values else 0.0,
                'h_max': max(h_values) if h_values else 0.0,
                'term1_1d_norm_mean': sum(term1_1d_norms) / len(term1_1d_norms) if term1_1d_norms else 0.0,
                'term2_1d_norm_mean': sum(term2_1d_norms) / len(term2_1d_norms) if term2_1d_norms else 0.0,
                # Projected-specific keys
                'n_projected_dirs': r,
                'n_random_dirs': n_random,
                'variance_explained': variance_explained,
                'eigenvalue_max': eigenvalues[0].item() if D > 0 else 0.0,
                'eigenvalue_min_used': eigenvalues[r - 1].item() if r > 0 else 0.0,
            }
            return phi, diag

        return phi
