"""
Sampling functions for generating random boxes and SIRS bump parameters.
"""

import numpy as np
from . import config
from .sirs import compute_feasible_fraction


def sample_box(rng):
    """
    Sample a random box within global joint limits.

    The box size is sampled as a fraction of the global range,
    then randomly positioned within the global limits.

    Args:
        rng: numpy random number generator

    Returns:
        Dictionary with keys 'q1_range' and 'q2_range'
    """
    # Global ranges
    q1_min_global, q1_max_global = config.Q1_RANGE
    q2_min_global, q2_max_global = config.Q2_RANGE

    q1_range_global = q1_max_global - q1_min_global
    q2_range_global = q2_max_global - q2_min_global

    # Sample box size fractions
    q1_fraction = rng.uniform(config.BOX_SIZE_FRACTION_MIN, config.BOX_SIZE_FRACTION_MAX)
    q2_fraction = rng.uniform(config.BOX_SIZE_FRACTION_MIN, config.BOX_SIZE_FRACTION_MAX)

    q1_box_width = q1_fraction * q1_range_global
    q2_box_width = q2_fraction * q2_range_global

    # Sample box position (lower bound)
    q1_min = rng.uniform(q1_min_global, q1_max_global - q1_box_width)
    q2_min = rng.uniform(q2_min_global, q2_max_global - q2_box_width)

    q1_max = q1_min + q1_box_width
    q2_max = q2_min + q2_box_width

    return {
        'q1_range': (q1_min, q1_max),
        'q2_range': (q2_min, q2_max)
    }


def sample_bumps(box, rng, edge_bias=False):
    """
    Sample random SIRS bumps within a box.

    Args:
        box: Box dictionary with 'q1_range' and 'q2_range'
        rng: numpy random number generator
        edge_bias: If True, bias bump centers toward edges/corners using Beta distribution

    Returns:
        List of bump dictionaries, each with keys:
        - 'mu': [mu1, mu2] center position
        - 'ls': [ls1, ls2] lengthscales
        - 'alpha': strength parameter
        - 'R': 2x2 rotation matrix (if config.ENABLE_ROTATION is True)
        - 'theta': rotation angle in radians (if config.ENABLE_ROTATION is True)
    """
    # Sample number of bumps
    K = rng.integers(config.NUM_BUMPS_MIN, config.NUM_BUMPS_MAX + 1)

    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    q1_width = q1_max - q1_min
    q2_width = q2_max - q2_min

    bumps = []

    for _ in range(K):
        # Sample bump center (uniform or edge-biased)
        if edge_bias:
            # Beta distribution pushes samples toward 0 or 1 (edges)
            # Beta(α, α) is symmetric; higher α = more edge bias
            u1 = rng.beta(config.EDGE_BIAS_BETA_ALPHA, config.EDGE_BIAS_BETA_ALPHA)
            u2 = rng.beta(config.EDGE_BIAS_BETA_ALPHA, config.EDGE_BIAS_BETA_ALPHA)
            mu1 = q1_min + u1 * q1_width
            mu2 = q2_min + u2 * q2_width
        else:
            # Uniform within box
            mu1 = rng.uniform(q1_min, q1_max)
            mu2 = rng.uniform(q2_min, q2_max)

        # Sample lengthscales (log-normal around fraction of box width)
        mean_ls1 = config.LENGTHSCALE_BOX_FRACTION * q1_width
        mean_ls2 = config.LENGTHSCALE_BOX_FRACTION * q2_width

        # Log-normal: if X ~ Normal(mu, sigma), then exp(X) ~ LogNormal
        ls1 = mean_ls1 * np.exp(rng.normal(0, config.LENGTHSCALE_LOGNORMAL_SIGMA))
        ls2 = mean_ls2 * np.exp(rng.normal(0, config.LENGTHSCALE_LOGNORMAL_SIGMA))

        # Clamp lengthscales to prevent degenerate bumps
        ls1 = np.clip(ls1, config.LENGTHSCALE_MIN_FRACTION * q1_width, config.LENGTHSCALE_MAX_FRACTION * q1_width)
        ls2 = np.clip(ls2, config.LENGTHSCALE_MIN_FRACTION * q2_width, config.LENGTHSCALE_MAX_FRACTION * q2_width)

        # Sample alpha (log-normal)
        log_alpha = rng.normal(config.ALPHA_LOGNORMAL_MEAN, config.ALPHA_LOGNORMAL_SIGMA)
        alpha = np.exp(log_alpha)

        # Sample rotation angle if enabled
        bump_dict = {
            'mu': np.array([mu1, mu2]),
            'ls': np.array([ls1, ls2]),
            'alpha': alpha
        }

        if config.ENABLE_ROTATION:
            # Sample rotation angle uniformly in [0, 2π)
            theta = rng.uniform(0, 2 * np.pi)
            # Compute rotation matrix
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            R = np.array([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ])
            bump_dict['theta'] = theta
            bump_dict['R'] = R

        bumps.append(bump_dict)

    return bumps


def calibrate_alpha(box, bumps, target_frac, rng=None, grid_n=300, max_iter=None, tolerance=None):
    """
    Scale all bump alphas by a constant factor to achieve target feasible fraction.

    Uses deterministic binary search to find scaling factor c such that:
    feasible_fraction(box, c * bumps) ≈ target_frac

    Args:
        box: Box dictionary
        bumps: List of bump dictionaries
        target_frac: Target feasible fraction (0 to 1)
        rng: numpy random number generator (unused; kept for API consistency)
        grid_n: Grid resolution for evaluation
        max_iter: Maximum iterations (default from config)
        tolerance: Convergence tolerance (default from config)

    Returns:
        New list of bumps with scaled alphas

    Note:
        Calibration is fully deterministic (uses fixed grid evaluation).
        Same inputs always produce same outputs.
    """
    if max_iter is None:
        max_iter = config.CALIBRATION_MAX_ITERATIONS
    if tolerance is None:
        tolerance = config.CALIBRATION_TOLERANCE

    if not bumps:
        # No bumps - cannot calibrate
        return bumps

    # Initial feasible fraction
    initial_frac = compute_feasible_fraction(box, bumps, grid_n)

    if abs(initial_frac - target_frac) < tolerance:
        # Already at target
        return bumps

    # Binary search bounds for scaling factor
    if initial_frac > target_frac:
        # Need to increase alpha (decrease feasibility)
        scale_min = 1.0
        scale_max = config.CALIBRATION_BINARY_SEARCH_FACTOR
        # Expand upper bound if needed
        for _ in range(20):  # Max 20 doublings = factor of ~1 million
            scaled_bumps = _scale_bumps(bumps, scale_max)
            frac = compute_feasible_fraction(box, scaled_bumps, grid_n)
            if frac <= target_frac:
                break
            scale_max *= config.CALIBRATION_BINARY_SEARCH_FACTOR
    else:
        # Need to decrease alpha (increase feasibility)
        scale_min = 1.0 / config.CALIBRATION_BINARY_SEARCH_FACTOR
        scale_max = 1.0
        # Expand lower bound if needed
        for _ in range(20):  # Max 20 halvings
            scaled_bumps = _scale_bumps(bumps, scale_min)
            frac = compute_feasible_fraction(box, scaled_bumps, grid_n)
            if frac >= target_frac:
                break
            scale_min /= config.CALIBRATION_BINARY_SEARCH_FACTOR

    # Binary search
    for iteration in range(max_iter):
        scale = (scale_min + scale_max) / 2.0
        scaled_bumps = _scale_bumps(bumps, scale)
        frac = compute_feasible_fraction(box, scaled_bumps, grid_n)

        if abs(frac - target_frac) < tolerance:
            # Converged
            return scaled_bumps

        if frac > target_frac:
            # Need more shrinkage (increase alpha)
            scale_min = scale
        else:
            # Need less shrinkage (decrease alpha)
            scale_max = scale

    # Return best result after max iterations
    return scaled_bumps


def _scale_bumps(bumps, scale_factor):
    """
    Scale all bump alphas by a constant factor.

    Args:
        bumps: List of bump dictionaries
        scale_factor: Multiplicative factor for alpha

    Returns:
        New list of bumps with scaled alphas (preserves rotation)
    """
    scaled_bumps = []
    for bump in bumps:
        scaled_bump = {
            'mu': bump['mu'].copy(),
            'ls': bump['ls'].copy(),
            'alpha': bump['alpha'] * scale_factor
        }

        # Preserve rotation if present
        if 'theta' in bump:
            scaled_bump['theta'] = bump['theta']
        if 'R' in bump:
            scaled_bump['R'] = bump['R'].copy()

        scaled_bumps.append(scaled_bump)

    return scaled_bumps


def generate_sirs_user(rng=None, target_frac=None, edge_bias=False, grid_n=300):
    """
    Convenience wrapper to generate a complete SIRS configuration.

    Generates box, bumps, and optionally calibrates to target feasibility.

    Args:
        rng: numpy random number generator (creates new if None)
        target_frac: Target feasible fraction (None = no calibration)
        edge_bias: If True, bias bump centers toward edges/corners
        grid_n: Grid resolution for calibration

    Returns:
        Dictionary with keys:
        - 'box': Box dictionary
        - 'bumps': List of bump dictionaries
        - 'feasible_fraction': Actual feasible fraction
        - 'X', 'Y', 'H', 'M': Grid evaluation results

    Example:
        >>> from sirs2d.sampler import generate_sirs_user
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> user = generate_sirs_user(rng, target_frac=0.6, edge_bias=True)
        >>> print(f"Feasible fraction: {user['feasible_fraction']:.2%}")
    """
    if rng is None:
        rng = np.random.default_rng(config.DEFAULT_RANDOM_SEED)

    # Generate box and bumps
    box = sample_box(rng)
    bumps = sample_bumps(box, rng, edge_bias=edge_bias)

    # Calibrate if target specified
    if target_frac is not None:
        bumps = calibrate_alpha(box, bumps, target_frac, grid_n=grid_n)

    # Evaluate on grid
    from .sirs import feasible_mask_grid
    X, Y, H, M = feasible_mask_grid(box, bumps, grid_n)

    return {
        'box': box,
        'bumps': bumps,
        'feasible_fraction': np.mean(M),
        'X': X,
        'Y': Y,
        'H': H,
        'M': M
    }
