import torch
import torch.nn.functional as F

class SVGD:
    """
    Stein Variational Gradient Descent Optimizer.

    Implements the update rule:
    phi(x) = E_{x'~q} [ k(x', x) grad_log_p(x') + grad_{x'} k(x', x) ]

    Uses RBF Kernel with median heuristic for bandwidth.
    """

    def __init__(self, kernel_width=None, repulsive_scaling=0.0):
        """
        Args:
            kernel_width: Fixed kernel width. If None, uses median heuristic per step.
            repulsive_scaling: Exponent alpha for D^alpha scaling of the repulsive force.
                               Counteracts the O(1/D) decay of RBF repulsion in high dimensions.
                               0.0 = disabled (standard SVGD), 0.5-1.0 = typical range.
        """
        self.kernel_width = kernel_width
        self.repulsive_scaling = repulsive_scaling

    def step(self, particles, log_prob_grad, return_diagnostics=False):
        """
        Compute the SVGD update direction (phi).

        Args:
            particles: Tensor of shape (K, D) representing current particles.
            log_prob_grad: Tensor of shape (K, D) representing grad_log_prob(particles).
                           (Driving force towards high probability regions)
            return_diagnostics: If True, return (phi, diagnostics_dict) instead of just phi.

        Returns:
            phi: Tensor of shape (K, D) representing the update direction.
                 particles <- particles + lr * phi
            diagnostics (optional): Dict with term1_norm, term2_norm, h, repulsive_scale
        """
        K, D = particles.shape

        # 1. Compute Pairwise Distance Matrix (Squared)
        # Use simple broadcasting: (K, 1, D) - (1, K, D)
        # dist_sq[i, j] = ||x_i - x_j||^2
        # Note: We sum over j in the formula, so let's be careful with indices.
        # Formula: sum_j [ k(x_j, x_i) ... ]
        # Let's organize inner loop variable as dim 0 (j) and outer variable as dim 1 (i).

        # x_j (source) -> dim 0
        # x_i (target) -> dim 1
        diff = particles.unsqueeze(1) - particles.unsqueeze(0)  # (K, K, D)
        # diff[j, i, :] = particles[j] - particles[i]

        dist_sq = torch.sum(diff ** 2, dim=-1)  # (K, K)

        # 2. Kernel Width (Median Heuristic)
        if self.kernel_width is None:
            # Get median of off-diagonal elements
            # We can just take median of all, valid approx.
            # actually better to ignore 0s on diagonal?
            # Classic median heuristic uses median(dist_sq) / log(K)
            median_dist = torch.median(dist_sq)
            h = median_dist / torch.log(torch.tensor(K + 1.0))
            # Fallback for numerical stability if particles collapse
            if h == 0:
                h = torch.tensor(1.0, device=particles.device)
        else:
            h = torch.tensor(self.kernel_width ** 2, device=particles.device) # Assume width is std dev approx

        # 3. Compute Kernel Matrix
        # k(x_j, x_i) = exp( - ||x_j - x_i||^2 / h )
        k_mat = torch.exp(-dist_sq / h)  # (K, K)

        # 4. Term 1: Driving Force
        # sum_j k(x_j, x_i) * grad_log_p(x_j)
        # Matrix multiplication: (K, K) * (K, D) -> (K, D)
        # We need sum over j.
        # k_mat is symmetric, so k_mat[j, i] == k_mat[i, j].
        # (K_mat^T @ grad) -> sum_j k_mat[j, i] * grad[j]
        term1 = torch.matmul(k_mat.t(), log_prob_grad)  # (K, D)

        # 5. Term 2: Repulsive Force
        # sum_j grad_{x_j} k(x_j, x_i)
        # grad_{x_j} exp(-||x_j - x_i||^2 / h)
        # = exp(...) * (-1/h) * grad(||x_j - x_i||^2)
        # = k(x_j, x_i) * (-1/h) * 2 * (x_j - x_i)
        # = - (2/h) * k(x_j, x_i) * (x_j - x_i)

        # We perform weighted sum over j
        # diff[j, i] = x_j - x_i
        # weights[j, i] = k_mat[j, i]

        # We can reshape k_mat to (K, K, 1) and multiply with diff (K, K, D)
        weighted_diff = k_mat.unsqueeze(-1) * diff  # (K, K, D)

        # Sum over j (dim 0)
        sum_weighted_diff = torch.sum(weighted_diff, dim=0)  # (K, D)

        term2 = - (2.0 / h) * sum_weighted_diff

        # 6. Final Phi
        # Scale repulsive force by D^alpha to counteract O(1/D) decay in high dimensions
        repulsive_scale = 1.0
        if self.repulsive_scaling > 0:
            repulsive_scale = D ** self.repulsive_scaling
            phi = (term1 + repulsive_scale * term2) / K
        else:
            phi = (term1 + term2) / K

        if return_diagnostics:
            diag = {
                'term1_norm': term1.norm().item() / K,
                'term2_norm': (repulsive_scale * term2).norm().item() / K,
                'term1_per_particle': (term1 / K).norm(dim=-1).mean().item(),
                'term2_per_particle': (repulsive_scale * term2 / K).norm(dim=-1).mean().item(),
                'h': h.item(),
                'repulsive_scale': repulsive_scale,
                'mean_pairwise_dist': dist_sq.mean().item(),
                'min_pairwise_dist_offdiag': dist_sq[~torch.eye(K, dtype=bool, device=particles.device)].min().item() if K > 1 else 0.0,
                'kernel_mean': k_mat.mean().item(),
            }
            return phi, diag

        return phi
