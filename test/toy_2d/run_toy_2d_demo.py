"""
2D Toy Demo: BALD + Projected SVGD on a non-convex feasibility region.

Demonstrates feasibility boundary convergence without the ambiguity of 2D slices.
Ground truth and predictions are both true 2D regions.

Usage:
    python run_toy_2d_demo.py --budget 20 --particles 50
    python run_toy_2d_demo.py --budget 20 --shape crescent
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.config import DEVICE

# Output directory
OUTPUT_DIR = "active_learning/images/toy_2d"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Ground Truth Feasibility Checker
# =============================================================================

class Toy2DChecker:
    """
    Ground truth 2D feasibility defined as sum of Gaussian blobs.

    h(x,y) = sum_i w_i * exp(-||[x,y] - c_i||^2 / (2*s_i^2)) - threshold

    Feasible where h > 0.
    """

    def __init__(self, centers, sizes, weights, threshold=0.5, device=DEVICE):
        """
        Args:
            centers: (K, 2) blob centers
            sizes: (K,) blob sizes (std dev)
            weights: (K,) blob weights (can be negative for cutouts)
            threshold: scalar threshold for feasibility
        """
        self.centers = torch.tensor(centers, dtype=torch.float32, device=device)
        self.sizes = torch.tensor(sizes, dtype=torch.float32, device=device)
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
        self.threshold = threshold
        self.device = device

    def logit_value(self, points):
        """
        Compute h(x,y) for given points.

        Args:
            points: (N, 2) or (2,)

        Returns:
            h: (N,) or scalar, positive = feasible
        """
        if points.dim() == 1:
            points = points.unsqueeze(0)
        points = points.to(self.device)

        # (N, K, 2) - (K, 2) -> (N, K, 2)
        diff = points.unsqueeze(1) - self.centers.unsqueeze(0)  # (N, K, 2)
        dist_sq = (diff ** 2).sum(dim=-1)  # (N, K)

        # Gaussian blobs
        gaussians = torch.exp(-dist_sq / (2 * self.sizes.unsqueeze(0) ** 2))  # (N, K)

        # Weighted sum
        h = (gaussians * self.weights.unsqueeze(0)).sum(dim=-1) - self.threshold  # (N,)

        return h.squeeze()


def create_crescent_gt(device=DEVICE):
    """Create a crescent/kidney-shaped GT region."""
    centers = [
        [0.0, 0.0],    # Main positive blob
        [0.35, 0.0],   # Negative cutout (offset right)
    ]
    sizes = [0.5, 0.35]
    weights = [1.0, -1.2]  # Negative weight creates cutout
    return Toy2DChecker(centers, sizes, weights, threshold=0.3, device=device)


def create_pac_man_gt(device=DEVICE):
    """Create a pac-man shaped GT region."""
    # Main circle + negative wedge approximated by blobs
    centers = [
        [0.0, 0.0],     # Main circle
        [0.3, 0.2],     # Mouth cutout top
        [0.3, -0.2],    # Mouth cutout bottom
    ]
    sizes = [0.5, 0.25, 0.25]
    weights = [1.0, -0.8, -0.8]
    return Toy2DChecker(centers, sizes, weights, threshold=0.3, device=device)


def create_irregular_blob_gt(device=DEVICE):
    """Create an irregular blob with multiple bumps."""
    centers = [
        [0.0, 0.0],      # Core
        [-0.3, 0.3],     # Bump top-left
        [0.35, 0.15],    # Bump right
        [0.0, -0.4],     # Bump bottom
        [0.4, -0.3],     # Small negative cutout
    ]
    sizes = [0.4, 0.25, 0.22, 0.25, 0.15]
    weights = [1.0, 0.6, 0.5, 0.55, -0.5]
    return Toy2DChecker(centers, sizes, weights, threshold=0.35, device=device)


def create_two_islands_gt(device=DEVICE):
    """Create two disconnected feasible islands - tests multi-modal discovery."""
    centers = [
        [-0.5, 0.3],     # Left island
        [0.5, -0.3],     # Right island
    ]
    sizes = [0.28, 0.32]
    weights = [1.0, 1.0]
    return Toy2DChecker(centers, sizes, weights, threshold=0.3, device=device)


def create_star_gt(device=DEVICE):
    """Create a star shape with multiple protrusions."""
    # Central core + 5 outward protrusions
    import math
    centers = [[0.0, 0.0]]  # Core
    sizes = [0.25]
    weights = [1.0]

    # 5 protrusions at 72° intervals
    for i in range(5):
        angle = i * 2 * math.pi / 5 - math.pi / 2  # Start from top
        r = 0.45
        centers.append([r * math.cos(angle), r * math.sin(angle)])
        sizes.append(0.18)
        weights.append(0.7)

    return Toy2DChecker(centers, sizes, weights, threshold=0.35, device=device)


def create_annulus_gt(device=DEVICE):
    """Create a ring/donut shape with hole in center."""
    centers = [
        [0.0, 0.0],      # Outer ring (large positive)
        [0.0, 0.0],      # Inner hole (negative cutout)
    ]
    sizes = [0.55, 0.25]
    weights = [1.0, -1.5]  # Strong negative to cut hole
    return Toy2DChecker(centers, sizes, weights, threshold=0.25, device=device)


def create_snake_gt(device=DEVICE):
    """Create a curved snake/S-shape corridor."""
    # Series of overlapping blobs forming a curved path
    centers = [
        [-0.5, 0.4],
        [-0.25, 0.3],
        [0.0, 0.1],
        [0.0, -0.15],
        [-0.25, -0.35],
        [-0.5, -0.45],
    ]
    sizes = [0.22, 0.2, 0.2, 0.2, 0.2, 0.22]
    weights = [0.8, 0.9, 1.0, 1.0, 0.9, 0.8]
    return Toy2DChecker(centers, sizes, weights, threshold=0.4, device=device)


def create_random_blob_gt(seed, n_blobs=6, device=DEVICE):
    """
    Create a random non-convex blob shape based on seed.

    Generates interesting shapes with:
    - A central core blob
    - Several peripheral blobs for bumps/protrusions
    - 1-2 negative blobs for non-convex cutouts
    """
    rng = np.random.RandomState(seed)

    centers = []
    sizes = []
    weights = []

    # 1. Central core blob (always present, near origin)
    centers.append([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)])
    sizes.append(rng.uniform(0.3, 0.45))
    weights.append(1.0)

    # 2. Peripheral positive blobs (protrusions)
    n_positive = n_blobs - 2  # Leave room for negative blobs
    for _ in range(n_positive):
        # Place around the periphery
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.25, 0.5)
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
        sizes.append(rng.uniform(0.15, 0.3))
        weights.append(rng.uniform(0.5, 0.9))

    # 3. Negative blobs (cutouts for non-convexity)
    n_negative = min(2, max(1, n_blobs // 3))
    for _ in range(n_negative):
        # Place cutouts not at center
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.15, 0.4)
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
        sizes.append(rng.uniform(0.12, 0.22))
        weights.append(rng.uniform(-0.7, -0.4))  # Negative weight

    # Threshold chosen to give reasonable feasible volume
    threshold = rng.uniform(0.3, 0.45)

    return Toy2DChecker(centers, sizes, weights, threshold=threshold, device=device)


# =============================================================================
# Parametric Decoder (maps latent z to blob parameters)
# =============================================================================

class Toy2DDecoder:
    """
    Decodes latent vector z into blob parameters for feasibility evaluation.

    z layout: [cx1, cy1, cx2, cy2, ..., s1, s2, ..., w1, w2, ...]

    Latent dim = K * 4 (2 for center + 1 size + 1 weight per blob)
    """

    def __init__(self, n_blobs=4, device=DEVICE):
        self.n_blobs = n_blobs
        self.latent_dim = n_blobs * 4
        self.device = device

        # Bounds for decoding (soft constraints via sigmoid/softplus)
        self.center_scale = 0.8  # Centers in [-0.8, 0.8]
        self.size_min = 0.1
        self.size_max = 0.5
        self.weight_scale = 1.5

    def decode(self, z):
        """
        Decode z into blob parameters.

        Args:
            z: (latent_dim,) or (B, latent_dim)

        Returns:
            centers: (K, 2) or (B, K, 2)
            sizes: (K,) or (B, K)
            weights: (K,) or (B, K)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B = z.shape[0]
        K = self.n_blobs

        # Parse z
        idx = 0
        centers_raw = z[:, idx:idx + K*2].view(B, K, 2)
        idx += K * 2
        sizes_raw = z[:, idx:idx + K]
        idx += K
        weights_raw = z[:, idx:idx + K]

        # Apply constraints
        centers = torch.tanh(centers_raw) * self.center_scale
        sizes = self.size_min + torch.sigmoid(sizes_raw) * (self.size_max - self.size_min)
        weights = torch.tanh(weights_raw) * self.weight_scale

        if squeeze:
            return centers.squeeze(0), sizes.squeeze(0), weights.squeeze(0)
        return centers, sizes, weights

    def evaluate(self, z, points, threshold=0.35):
        """
        Evaluate h(x,y) for given z and points.

        Args:
            z: (latent_dim,) or (B, latent_dim)
            points: (N, 2)
            threshold: feasibility threshold

        Returns:
            h: (N,) or (B, N)
        """
        centers, sizes, weights = self.decode(z)

        if centers.dim() == 2:
            # Single z: centers (K, 2), sizes (K,), weights (K,)
            diff = points.unsqueeze(1) - centers.unsqueeze(0)  # (N, K, 2)
            dist_sq = (diff ** 2).sum(dim=-1)  # (N, K)
            gaussians = torch.exp(-dist_sq / (2 * sizes.unsqueeze(0) ** 2))
            h = (gaussians * weights.unsqueeze(0)).sum(dim=-1) - threshold
        else:
            # Batch z: centers (B, K, 2), sizes (B, K), weights (B, K)
            B = centers.shape[0]
            N = points.shape[0]
            # (B, N, K, 2)
            diff = points.unsqueeze(0).unsqueeze(2) - centers.unsqueeze(1)
            dist_sq = (diff ** 2).sum(dim=-1)  # (B, N, K)
            gaussians = torch.exp(-dist_sq / (2 * sizes.unsqueeze(1) ** 2))
            h = (gaussians * weights.unsqueeze(1)).sum(dim=-1) - threshold  # (B, N)

        return h


# =============================================================================
# Oracle
# =============================================================================

class Toy2DOracle:
    """Binary oracle using ground truth checker."""

    def __init__(self, gt_checker):
        self.gt_checker = gt_checker

    def query(self, point):
        """Returns 1.0 if feasible, 0.0 if infeasible."""
        h = self.gt_checker.logit_value(point)
        return 1.0 if h.item() > 0 else 0.0


# =============================================================================
# Particle Posterior
# =============================================================================

class Toy2DParticlePosterior:
    """
    Particle-based posterior over latent z.
    """

    def __init__(self, latent_dim, n_particles, prior_mean, prior_std, device=DEVICE):
        self.latent_dim = latent_dim
        self.n_particles = n_particles
        self.device = device

        # Initialize particles from prior
        self.particles = prior_mean + prior_std * torch.randn(
            n_particles, latent_dim, device=device
        )
        self.particles.requires_grad_(True)

        # Prior parameters
        self.prior_mean = prior_mean.to(device)
        self.prior_std = prior_std.to(device)

    def get_particles(self):
        return self.particles

    @property
    def mean(self):
        return self.particles.mean(dim=0)

    def sample(self, n_samples):
        """Sample n particles (with replacement if needed)."""
        indices = torch.randint(0, self.n_particles, (n_samples,), device=self.device)
        return self.particles[indices].detach()


# =============================================================================
# SVGD Update
# =============================================================================

class Toy2DSVGD:
    """
    Simplified SVGD for 2D toy problem.
    """

    def __init__(self, decoder, posterior, prior_mean, prior_std,
                 step_size=0.1, n_iters=50, device=DEVICE):
        self.decoder = decoder
        self.posterior = posterior
        self.prior_mean = prior_mean.to(device)
        self.prior_std = prior_std.to(device)
        self.step_size = step_size
        self.n_iters = n_iters
        self.device = device

        # Data storage
        self.observations = []  # List of (point, outcome)

        # Likelihood temperature
        self.tau = 0.3

    def add_observation(self, point, outcome):
        """Add a new observation."""
        self.observations.append((point.to(self.device), outcome))

    def log_likelihood(self, particles, points, outcomes):
        """
        Compute log likelihood of observations given particles.

        Args:
            particles: (M, latent_dim)
            points: list of (2,) tensors
            outcomes: list of 0/1 floats

        Returns:
            ll: (M,) log likelihood per particle
        """
        if len(points) == 0:
            return torch.zeros(particles.shape[0], device=self.device)

        # Stack points: (N, 2)
        pts = torch.stack(points)
        outcomes_t = torch.tensor(outcomes, device=self.device)

        # Evaluate h for all particles and points: (M, N)
        h = self.decoder.evaluate(particles, pts)

        # Likelihood: p(y=1|h) = sigmoid(h/tau)
        p = torch.sigmoid(h / self.tau)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)

        # Log likelihood
        ll = outcomes_t * torch.log(p) + (1 - outcomes_t) * torch.log(1 - p)
        return ll.sum(dim=-1)  # (M,)

    def log_prior(self, particles):
        """Gaussian log prior."""
        diff = (particles - self.prior_mean) / self.prior_std
        return -0.5 * (diff ** 2).sum(dim=-1)

    def rbf_kernel(self, particles):
        """RBF kernel with median heuristic."""
        M = particles.shape[0]

        # Pairwise distances
        diff = particles.unsqueeze(0) - particles.unsqueeze(1)  # (M, M, D)
        dist_sq = (diff ** 2).sum(dim=-1)  # (M, M)

        # Median heuristic for bandwidth
        median_dist = torch.median(dist_sq.view(-1))
        h = median_dist / np.log(M + 1)
        h = torch.clamp(h, min=1e-4)

        # Kernel
        K = torch.exp(-dist_sq / h)

        # Gradient of kernel
        grad_K = -2 * diff / h * K.unsqueeze(-1)  # (M, M, D)

        return K, grad_K

    def update(self):
        """Run SVGD updates."""
        points = [obs[0] for obs in self.observations]
        outcomes = [obs[1] for obs in self.observations]

        particles = self.posterior.particles
        M = particles.shape[0]

        for _ in range(self.n_iters):
            particles = particles.detach().requires_grad_(True)

            # Compute log posterior
            ll = self.log_likelihood(particles, points, outcomes)
            lp = self.log_prior(particles)
            log_post = ll + lp

            # Gradient of log posterior
            grad_log_post = torch.autograd.grad(log_post.sum(), particles)[0]  # (M, D)

            # Kernel
            K, grad_K = self.rbf_kernel(particles.detach())

            # SVGD update direction
            # phi = (1/M) * sum_j [K(x_j, x_i) * grad_log_p(x_j) + grad_K(x_j, x_i)]
            phi = (K @ grad_log_post + grad_K.sum(dim=0)) / M

            # Update
            particles = particles.detach() + self.step_size * phi

        self.posterior.particles = particles.detach().requires_grad_(True)


# =============================================================================
# BALD Acquisition
# =============================================================================

class Toy2DBALD:
    """
    BALD acquisition for 2D toy problem.
    """

    def __init__(self, decoder, posterior, tau=0.3, device=DEVICE):
        self.decoder = decoder
        self.posterior = posterior
        self.tau = tau
        self.device = device

    def compute_score(self, points, particles=None):
        """
        Compute BALD score: I(y; z | x) = H[E[p]] - E[H[p]]

        Args:
            points: (N, 2) candidate points
            particles: (M, latent_dim) optional pre-sampled particles

        Returns:
            scores: (N,) BALD scores
        """
        if particles is None:
            particles = self.posterior.get_particles()

        # h values: (M, N)
        h = self.decoder.evaluate(particles, points)

        # Probabilities: (M, N)
        p = torch.sigmoid(h / self.tau)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)

        # Mean probability: (N,)
        p_mean = p.mean(dim=0)

        # H[E[p]] - entropy of mean
        H_mean = -p_mean * torch.log(p_mean) - (1 - p_mean) * torch.log(1 - p_mean)

        # E[H[p]] - mean of entropies
        H_each = -p * torch.log(p) - (1 - p) * torch.log(1 - p)  # (M, N)
        E_H = H_each.mean(dim=0)  # (N,)

        # BALD = mutual information
        bald = H_mean - E_H
        return torch.clamp(bald, min=0)

    def select_test(self, bounds, n_candidates=5000, n_restarts=10, n_iters=30):
        """
        Select test point maximizing BALD.

        Uses random candidates + gradient refinement.
        """
        lower = bounds[:, 0].to(self.device)
        upper = bounds[:, 1].to(self.device)

        particles = self.posterior.get_particles().detach()

        # Random candidates
        candidates = lower + (upper - lower) * torch.rand(
            n_candidates, 2, device=self.device
        )

        with torch.no_grad():
            scores = self.compute_score(candidates, particles)

        # Take top candidates for refinement
        top_k = min(n_restarts, n_candidates)
        _, top_indices = scores.topk(top_k)

        # Gradient refinement
        best_point = candidates[top_indices[0]].clone()
        best_score = scores[top_indices[0]].item()

        for idx in top_indices:
            point = candidates[idx].clone().requires_grad_(True)
            optimizer = torch.optim.Adam([point], lr=0.02)

            for _ in range(n_iters):
                optimizer.zero_grad()
                score = self.compute_score(point.unsqueeze(0), particles)
                (-score).backward()
                optimizer.step()

                # Clamp to bounds
                with torch.no_grad():
                    point.data = torch.clamp(point.data, lower, upper)

            with torch.no_grad():
                final_score = self.compute_score(point.unsqueeze(0), particles).item()

            if final_score > best_score:
                best_score = final_score
                best_point = point.detach().clone()

        return best_point, best_score


# =============================================================================
# Visualization
# =============================================================================

def compute_boundary_data(gt_checker, decoder, posterior, bounds, resolution=200, tau=0.3):
    """
    Compute GT and predicted feasibility on a grid.

    Uses ensemble averaging: averages p(feasible|z) across all particles
    rather than evaluating at the mean z (which can collapse to trivial shapes).
    """
    device = gt_checker.device

    x = torch.linspace(bounds[0, 0], bounds[0, 1], resolution, device=device)
    y = torch.linspace(bounds[1, 0], bounds[1, 1], resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # (R*R, 2)

    with torch.no_grad():
        # GT: logit value (positive = feasible)
        Z_gt = gt_checker.logit_value(points).view(resolution, resolution)

        # Predicted: ensemble average of p(feasible|z) across particles
        particles = posterior.get_particles()  # (M, latent_dim)
        h_all = decoder.evaluate(particles, points)  # (M, N)
        p_all = torch.sigmoid(h_all / tau)  # (M, N) probabilities
        p_mean = p_all.mean(dim=0)  # (N,) average probability

        # Convert back to "logit-like" for consistent visualization
        # p > 0.5 means feasible, so Z_pred > 0 means feasible
        Z_pred = (p_mean - 0.5).view(resolution, resolution)

    return {
        'X': X.cpu().numpy(),
        'Y': Y.cpu().numpy(),
        'Z_gt': Z_gt.cpu().numpy(),
        'Z_pred': Z_pred.cpu().numpy(),
    }


def plot_figure(data, query_history, iteration, save_path, dpi=300):
    """Plot a single figure."""
    fig, ax = plt.subplots(figsize=(5, 5))

    X, Y = data['X'], data['Y']
    Z_gt, Z_pred = data['Z_gt'], data['Z_pred']

    # Colors
    GT_FILL = '#87CEEB'
    GT_BORDER = '#00CED1'
    PRED_FILL = '#FFB366'
    PRED_BORDER = '#FF6600'

    # GT region (bottom)
    gt_feasible = Z_gt > 0
    ax.contourf(X, Y, gt_feasible.astype(float), levels=[0.5, 1.5],
               colors=[GT_FILL], alpha=0.5, zorder=1)
    try:
        ax.contour(X, Y, Z_gt, levels=[0], colors=[GT_BORDER],
                  linewidths=3, linestyles='solid', zorder=2)
    except:
        pass

    # Predicted region (on top)
    pred_feasible = Z_pred > 0
    ax.contourf(X, Y, pred_feasible.astype(float), levels=[0.5, 1.5],
               colors=[PRED_FILL], alpha=0.4, zorder=3)
    try:
        ax.contour(X, Y, Z_pred, levels=[0], colors=[PRED_BORDER],
                  linewidths=3, linestyles='dashed', zorder=4)
    except:
        pass

    # Query points
    n_queries = len(query_history)
    for i, (point, outcome) in enumerate(query_history):
        pt = point.cpu().numpy()
        is_current = (i == n_queries - 1)  # Last query is current

        if is_current:
            # Current query: gold star (matches legend)
            ax.plot(pt[0], pt[1], '*', color='#FFD700',
                   markersize=20, markeredgecolor='#B8860B', markeredgewidth=1.5, zorder=11)
        else:
            # Previous queries: circle or X
            if outcome > 0.5:
                ax.plot(pt[0], pt[1], 'o', color='#27ae60',
                       markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=10)
            else:
                ax.plot(pt[0], pt[1], 'X', color='#e74c3c',
                       markersize=10, markeredgecolor='white', markeredgewidth=2, zorder=10)

    # Clean formatting
    ax.set_title(f'Iteration {iteration}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('auto')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_combined_figure(all_data, save_path, dpi=300):
    """Plot combined 1x4 figure."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    iterations = sorted(all_data.keys())

    GT_FILL = '#87CEEB'
    GT_BORDER = '#00CED1'
    PRED_FILL = '#FFB366'
    PRED_BORDER = '#FF6600'

    for ax_idx, iteration in enumerate(iterations):
        ax = axes[ax_idx]
        data, query_history = all_data[iteration]

        X, Y = data['X'], data['Y']
        Z_gt, Z_pred = data['Z_gt'], data['Z_pred']

        # GT
        gt_feasible = Z_gt > 0
        ax.contourf(X, Y, gt_feasible.astype(float), levels=[0.5, 1.5],
                   colors=[GT_FILL], alpha=0.5, zorder=1)
        try:
            ax.contour(X, Y, Z_gt, levels=[0], colors=[GT_BORDER],
                      linewidths=2.5, linestyles='solid', zorder=2)
        except:
            pass

        # Predicted
        pred_feasible = Z_pred > 0
        ax.contourf(X, Y, pred_feasible.astype(float), levels=[0.5, 1.5],
                   colors=[PRED_FILL], alpha=0.4, zorder=3)
        try:
            ax.contour(X, Y, Z_pred, levels=[0], colors=[PRED_BORDER],
                      linewidths=2.5, linestyles='dashed', zorder=4)
        except:
            pass

        # Query points
        n_queries = len(query_history)
        for i, (point, outcome) in enumerate(query_history):
            pt = point.cpu().numpy()
            is_current = (i == n_queries - 1)  # Last query is current

            if is_current:
                # Current query: gold star (matches legend)
                ax.plot(pt[0], pt[1], '*', color='#FFD700',
                       markersize=16, markeredgecolor='#B8860B', markeredgewidth=1, zorder=11)
            else:
                # Previous queries: circle or X
                if outcome > 0.5:
                    ax.plot(pt[0], pt[1], 'o', color='#27ae60',
                           markersize=7, markeredgecolor='white', markeredgewidth=1.5, zorder=10)
                else:
                    ax.plot(pt[0], pt[1], 'X', color='#e74c3c',
                           markersize=7, markeredgecolor='white', markeredgewidth=1.5, zorder=10)

        ax.set_title(f'Iteration {iteration}', fontsize=12, fontweight='bold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect('auto')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=GT_FILL, edgecolor=GT_BORDER,
                      linewidth=2, alpha=0.5, label='Ground Truth'),
        mpatches.Patch(facecolor=PRED_FILL, edgecolor=PRED_BORDER,
                      linewidth=2, linestyle='--', alpha=0.4, label='Predicted'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
                  markersize=8, label='Feasible'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#e74c3c',
                  markersize=8, label='Infeasible'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700',
                  markersize=12, markeredgecolor='#B8860B', markeredgewidth=1,
                  label='Current Query'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="2D Toy Demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--particles", type=int, default=50)
    parser.add_argument("--n_blobs", type=int, default=10,
                       help="Number of blobs in decoder (more = more expressive)")
    parser.add_argument("--gt_blobs", type=int, default=7,
                       help="Number of blobs in random GT shape")
    parser.add_argument("--shape", type=str, default="random",
                       choices=["crescent", "pacman", "blob", "two_islands", "star", "annulus", "snake", "random"])
    parser.add_argument("--iterations", type=str, default="0,5,10,20",
                       help="Iterations to plot")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Set seeds - GT uses seed directly, learning uses seed+1000 for independence
    gt_seed = args.seed
    learning_seed = args.seed + 1000
    torch.manual_seed(learning_seed)
    np.random.seed(learning_seed)

    plot_iterations = [int(x) for x in args.iterations.split(',')]

    print(f"2D Toy Demo: shape={args.shape}, seed={args.seed}, budget={args.budget}, particles={args.particles}, n_blobs={args.n_blobs}")

    # 1. Create GT
    if args.shape == "random":
        gt_checker = create_random_blob_gt(gt_seed, n_blobs=args.gt_blobs, device=DEVICE)
        print(f"  Random GT generated with {args.gt_blobs} blobs (seed={gt_seed})")
    else:
        shape_creators = {
            "crescent": create_crescent_gt,
            "pacman": create_pac_man_gt,
            "blob": create_irregular_blob_gt,
            "two_islands": create_two_islands_gt,
            "star": create_star_gt,
            "annulus": create_annulus_gt,
            "snake": create_snake_gt,
        }
        gt_checker = shape_creators[args.shape](DEVICE)

    oracle = Toy2DOracle(gt_checker)

    # 2. Create decoder and posterior
    n_blobs = args.n_blobs
    decoder = Toy2DDecoder(n_blobs=n_blobs, device=DEVICE)

    # Prior: start slightly larger than GT with negative blobs for non-convexity
    # z layout: [cx1,cy1,cx2,cy2,..., s1,s2,..., w1,w2,...]
    prior_mean = torch.zeros(decoder.latent_dim, device=DEVICE)

    # Indices
    size_start = n_blobs * 2
    weight_start = n_blobs * 3
    n_positive = (n_blobs * 2) // 3  # ~2/3 positive blobs

    # Sizes: sigmoid(0.3) ≈ 0.57 → size ≈ 0.33
    prior_mean[size_start:weight_start] = 0.3

    # Positive weights: tanh(0.85) ≈ 0.69 → weight ≈ 1.04
    prior_mean[weight_start:weight_start + n_positive] = 0.95

    # Small negative weights: tanh(-0.15) ≈ -0.15 → weight ≈ -0.22 (minimal cutouts)
    prior_mean[weight_start + n_positive:] = -0.20

    prior_std = torch.ones(decoder.latent_dim, device=DEVICE) * 1.0

    posterior = Toy2DParticlePosterior(
        decoder.latent_dim, args.particles, prior_mean, prior_std, DEVICE
    )

    # 3. Create SVGD and BALD
    svgd = Toy2DSVGD(decoder, posterior, prior_mean, prior_std,
                     step_size=0.05, n_iters=50, device=DEVICE)
    bald = Toy2DBALD(decoder, posterior, tau=0.3, device=DEVICE)

    # Bounds for test selection and visualization
    bounds = torch.tensor([[-1.2, 1.2], [-1.2, 1.2]], device=DEVICE)

    # 4. Run active learning loop
    query_history = []  # Completed queries (already used to update posterior)
    all_data = {}

    print("\nRunning active learning...")
    for i in range(args.budget):
        # Select test point (this is the "current" query for iteration i)
        test_point, score = bald.select_test(bounds)

        # Query oracle to get outcome (needed for visualization)
        outcome = oracle.query(test_point)

        # Capture visualization BEFORE posterior update
        # Shows: current posterior + the just-selected test point
        if i in plot_iterations:
            print(f"  Capturing iteration {i}...")
            data = compute_boundary_data(gt_checker, decoder, posterior, bounds)
            # Include the current query in the visualization
            display_history = list(query_history) + [(test_point.clone(), outcome)]
            all_data[i] = (data, display_history)

            # Save individual figure
            save_path = os.path.join(OUTPUT_DIR, f"toy2d_iter_{i:02d}.png")
            plot_figure(data, display_history, i, save_path, args.dpi)
            print(f"    Saved: {save_path}")

        # Now update: add to history and update posterior
        query_history.append((test_point.clone(), outcome))
        svgd.add_observation(test_point, outcome)
        svgd.update()

        status = "feasible" if outcome > 0.5 else "infeasible"
        print(f"  [{i+1}] BALD={score:.4f}, {status}")

    # Final iteration: show final state (no new query, just final posterior)
    if args.budget in plot_iterations:
        print(f"  Capturing iteration {args.budget} (final)...")
        data = compute_boundary_data(gt_checker, decoder, posterior, bounds)
        all_data[args.budget] = (data, list(query_history))
        save_path = os.path.join(OUTPUT_DIR, f"toy2d_iter_{args.budget:02d}.png")
        plot_figure(data, query_history, args.budget, save_path, args.dpi)
        print(f"    Saved: {save_path}")

    # 5. Save combined figure
    combined_path = os.path.join(OUTPUT_DIR, "toy2d_convergence.png")
    plot_combined_figure(all_data, combined_path, args.dpi)
    print(f"\nSaved combined figure: {combined_path}")

    print(f"\nDone! Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
