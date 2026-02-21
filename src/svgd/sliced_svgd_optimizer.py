import torch


class SlicedSVGD:
    """
    Sliced Stein Variational Gradient Descent Optimizer.

    Projects particles and score gradients onto random 1D slices,
    computes 1D RBF kernel SVGD updates per slice, then projects
    back and averages. Avoids the curse of dimensionality in the
    kernel bandwidth estimation.

    Reference: Gong et al., "Sliced Kernelized Stein Discrepancy" (2020)
    """

    def __init__(self, n_slices=20, kernel_width=None, kernel_type='rbf'):
        """
        Args:
            n_slices: Number of random projection directions per step (S).
            kernel_width: Fixed 1D kernel width. If None, uses median
                          heuristic per slice (recommended).
            kernel_type: Kernel function to use: 'rbf' or 'imq'.
        """
        self.n_slices = n_slices
        self.kernel_width = kernel_width
        self.kernel_type = kernel_type

    def step(self, particles, log_prob_grad, return_diagnostics=False):
        """
        Compute the Sliced SVGD update direction (phi).

        Same interface as SVGD.step():
            particles: Tensor of shape (K, D) representing current particles.
            log_prob_grad: Tensor of shape (K, D) representing grad_log_prob(particles).
            return_diagnostics: If True, return (phi, diagnostics_dict).

        Returns:
            phi: Tensor of shape (K, D) representing the update direction.
            diagnostics (optional): Dict with per-slice and aggregate stats.
        """
        K, D = particles.shape
        device = particles.device

        phi = torch.zeros_like(particles)  # (K, D) accumulator

        # Diagnostic accumulators
        if return_diagnostics:
            h_values = []
            term1_1d_norms = []
            term2_1d_norms = []

        for _ in range(self.n_slices):
            # 1. Sample random direction on unit sphere S^{D-1}
            r = torch.randn(D, device=device)
            r = r / r.norm()  # (D,)

            # 2. Project particles to 1D
            s_proj = particles @ r  # (K,)

            # 3. Project score gradient to 1D
            g_proj = log_prob_grad @ r  # (K,)

            # 4. Compute 1D pairwise squared distances
            # diff_1d[j, i] = s_proj[j] - s_proj[i]
            diff_1d = s_proj.unsqueeze(1) - s_proj.unsqueeze(0)  # (K, K)
            dist_sq_1d = diff_1d ** 2  # (K, K)

            # 5. 1D median heuristic bandwidth
            if self.kernel_width is None:
                median_dist = torch.median(dist_sq_1d)
                h = median_dist / torch.log(torch.tensor(K + 1.0, device=device))
                if h == 0:
                    h = torch.tensor(1.0, device=device)
            else:
                h = torch.tensor(self.kernel_width ** 2, device=device)

            # 6. 1D kernel + 7. 1D SVGD terms
            scaled_dist = dist_sq_1d / h
            if self.kernel_type == 'imq':
                # IMQ: k(s_j, s_i) = (1 + dist_sq/h)^{-1/2}
                base = 1.0 + scaled_dist
                k_mat = base.pow(-0.5)  # (K, K)
                # ∇_{s_j} k = -(1/h) * (s_j - s_i) * (1 + dist_sq/h)^{-3/2}
                grad_factor = -(1.0 / h) * base.pow(-1.5)  # (K, K)
            else:
                # RBF: k(s_j, s_i) = exp(-dist_sq/h)
                k_mat = torch.exp(-scaled_dist)  # (K, K)
                # ∇_{s_j} k = -(2/h) * k * (s_j - s_i)
                grad_factor = -(2.0 / h) * k_mat  # (K, K)

            # term1[i] = sum_j k(s_j, s_i) * g_j / K
            term1_1d = k_mat.t() @ g_proj / K  # (K,)

            # term2[i] = sum_j ∇_{s_j} k(s_j, s_i) / K
            term2_1d = (grad_factor * diff_1d).sum(dim=0) / K  # (K,)

            # 8. Combine and project back to D dimensions
            phi_1d = term1_1d + term2_1d  # (K,)
            phi += torch.outer(phi_1d, r)  # (K, D)

            if return_diagnostics:
                h_values.append(h.item())
                term1_1d_norms.append(term1_1d.norm().item())
                term2_1d_norms.append(term2_1d.norm().item())

        # 9. Average over slices
        phi /= self.n_slices

        if return_diagnostics:
            # Full pairwise distances for diagnostics (matches vanilla SVGD format)
            diff_full = particles.unsqueeze(1) - particles.unsqueeze(0)
            dist_sq_full = torch.sum(diff_full ** 2, dim=-1)
            mask = ~torch.eye(K, dtype=bool, device=device)

            diag = {
                # Keys expected by svgd_vi.py inner-loop logger
                'term1_norm': phi.norm().item(),
                'term2_norm': 0.0,  # not separable after slice averaging
                'term1_per_particle': phi.norm(dim=-1).mean().item(),
                'term2_per_particle': 0.0,
                'h': sum(h_values) / len(h_values),  # mean bandwidth across slices
                'repulsive_scale': 1.0,
                'mean_pairwise_dist': dist_sq_full.mean().item(),
                'min_pairwise_dist_offdiag': dist_sq_full[mask].min().item() if K > 1 else 0.0,
                'kernel_mean': 0.0,  # not meaningful aggregated across slices
                # Sliced-specific diagnostics
                'n_slices': self.n_slices,
                'h_min': min(h_values),
                'h_max': max(h_values),
                'term1_1d_norm_mean': sum(term1_1d_norms) / len(term1_1d_norms),
                'term2_1d_norm_mean': sum(term2_1d_norms) / len(term2_1d_norms),
            }
            return phi, diag

        return phi
