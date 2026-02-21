"""
Core mathematical functions for SIRS (Sparse Interaction RBF Shrinkage).

All functions are vectorized using numpy for efficient grid evaluation.
"""

import numpy as np
from scipy import ndimage


def smooth_min_cubic(a, b, k):
    """
    Smooth minimum of two values using cubic polynomial blending.

    Provides C^2 continuous approximation of min(a, b) with local support.
    When |a - b| > k, returns exact min(a, b).
    When |a - b| < k, blends smoothly between the values.

    Args:
        a, b: Input values (arrays or scalars)
        k: Smoothing radius (same units as a, b)
           k=0 returns exact min(a, b)
           Larger k = more smoothing

    Returns:
        Smooth minimum, conservative (never less than true minimum)

    Reference:
        Inigo Quilez - https://iquilezles.org/articles/smin/
    """
    if k == 0.0:
        return np.minimum(a, b)

    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * h * k / 6.0


def inside_box(q, box):
    """
    Check if point(s) q are inside the box.

    Args:
        q: Array of shape (..., 2) with joint angles [q1, q2]
        box: Dictionary with keys 'q1_range' and 'q2_range'

    Returns:
        Boolean array of same shape as q[..., 0]
    """
    q1, q2 = q[..., 0], q[..., 1]
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    return (q1 >= q1_min) & (q1 <= q1_max) & (q2 >= q2_min) & (q2 <= q2_max)


def inside_box_nd(q, box):
    """
    N-dimensional version of inside_box for arbitrary dimensions.

    Args:
        q: Array of shape (..., N) with joint angles
        box: Dictionary with keys 'q1_range', 'q2_range', ..., 'qN_range'

    Returns:
        Boolean array of same shape as q[..., 0]
    """
    n_dims = q.shape[-1]
    inside = np.ones(q.shape[:-1], dtype=bool)

    for i in range(n_dims):
        dim_name = f'q{i+1}_range'
        if dim_name not in box:
            raise ValueError(f"Box missing key: {dim_name}")

        q_min, q_max = box[dim_name]
        q_i = q[..., i]
        inside &= (q_i >= q_min) & (q_i <= q_max)

    return inside


def box_margin(q, box):
    """
    Compute distance to nearest box edge for point(s) inside box.
    Returns large negative value for points outside box.

    Args:
        q: Array of shape (..., 2) with joint angles [q1, q2]
        box: Dictionary with keys 'q1_range' and 'q2_range'

    Returns:
        Array of distances to nearest edge (shape same as q[..., 0])
    """
    q1, q2 = q[..., 0], q[..., 1]
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    # Distance to each edge
    d_q1_min = q1 - q1_min
    d_q1_max = q1_max - q1
    d_q2_min = q2 - q2_min
    d_q2_max = q2_max - q2

    # Minimum distance to any edge
    margin = np.minimum.reduce([d_q1_min, d_q1_max, d_q2_min, d_q2_max])

    # Note: We return the raw margin (which is naturally negative outside)
    # instead of clamping to -1e6 or flipping signs. 
    # This provides a valid Signed Distance Function (SDF).
    return margin


def smooth_box_margin(q, box, smoothing_k=0.0):
    """
    Compute smooth distance to nearest box edge for point(s) inside box.
    Returns large negative value for points outside box.

    Uses cubic polynomial smoothing to create C^2 continuous corners.
    Works with arbitrary-dimensional boxes.

    Args:
        q: Array of shape (..., N) with joint angles
        box: Dictionary with keys 'q1_range', 'q2_range', ..., 'qN_range'
        smoothing_k: Corner smoothing radius (0 = sharp corners, >0 = smooth)
                     k has same units as q (e.g., degrees)
                     Recommended: 1-5 for subtle smoothing, 5-10 for visible rounding

    Returns:
        Array of distances to nearest edge (shape same as q[..., 0])

    Examples:
        >>> box = {'q1_range': (0, 100), 'q2_range': (0, 100)}
        >>> q = np.array([[50, 50], [10, 10]])  # center and near corner
        >>> margin_sharp = smooth_box_margin(q, box, smoothing_k=0.0)
        >>> margin_smooth = smooth_box_margin(q, box, smoothing_k=5.0)
    """
    # Extract all dimensions dynamically
    n_dims = q.shape[-1]

    # Collect all edge distances (2 per dimension: lower and upper)
    distances = []
    for i in range(n_dims):
        dim_name = f'q{i+1}_range'
        if dim_name not in box:
            raise ValueError(f"Box missing key: {dim_name}")

        q_min, q_max = box[dim_name]
        q_i = q[..., i]

        # Distance to lower and upper edge in this dimension
        d_min = q_i - q_min
        d_max = q_max - q_i

        distances.extend([d_min, d_max])

    # Apply smooth minimum reduction
    if smoothing_k == 0.0:
        # Sharp corners (original behavior)
        margin = np.minimum.reduce(distances)
    else:
        # Smooth corners using pairwise reduction
        margin = distances[0]
        for i in range(1, len(distances)):
            margin = smooth_min_cubic(margin, distances[i], smoothing_k)

    return margin


def compute_auto_smoothing_k(box, fraction=0.1):
    """
    Compute automatic smoothing parameter based on box size.

    Returns k = fraction * (smallest box dimension)

    Args:
        box: Box dictionary with 'q1_range', 'q2_range', ..., 'qN_range'
        fraction: Fraction of smallest dimension to use (default: 0.02 = 2%)

    Returns:
        Float, recommended smoothing_k value

    Examples:
        >>> box = {'q1_range': (0, 100), 'q2_range': (0, 200)}
        >>> k = compute_auto_smoothing_k(box)  # Returns 2.0 (2% of 100)
    """
    # Extract all dimension widths
    widths = []
    i = 1
    while f'q{i}_range' in box:
        q_min, q_max = box[f'q{i}_range']
        widths.append(q_max - q_min)
        i += 1

    if not widths:
        raise ValueError("Box has no valid dimension ranges")

    # Return fraction of smallest dimension
    return fraction * min(widths)


def rbf_value(q, mu, ls, R=None):
    """
    Evaluate anisotropic Gaussian RBF at point(s) q with optional rotation.

    Without rotation (axis-aligned):
        RBF(q) = exp(-0.5 * ((q1 - mu1)^2 / ls1^2 + (q2 - mu2)^2 / ls2^2))

    With rotation (non-axis-aligned):
        dq_rot = R.T @ (q - mu)
        RBF(q) = exp(-0.5 * ((dq_rot[0] / ls1)^2 + (dq_rot[1] / ls2)^2))

    Args:
        q: Array of shape (..., 2) with joint angles [q1, q2]
        mu: Array [mu1, mu2] - RBF center
        ls: Array [ls1, ls2] - Lengthscales for each dimension
        R: Optional 2x2 rotation matrix for tilted ellipse

    Returns:
        Array of RBF values (shape same as q[..., 0])
    """
    ls1, ls2 = ls[0], ls[1]

    if R is None:
        # Axis-aligned case (original implementation)
        q1, q2 = q[..., 0], q[..., 1]
        mu1, mu2 = mu[0], mu[1]
        sq_dist = ((q1 - mu1) / ls1)**2 + ((q2 - mu2) / ls2)**2
    else:
        # Rotated case: apply R.T to (q - mu)
        # q has shape (..., 2), need to handle batch dimensions
        dq = q - mu  # Shape: (..., 2)

        # Apply rotation: dq_rot = R.T @ dq
        # For batched application: (2, 2) @ (..., 2) -> (..., 2)
        dq_rot = np.einsum('ij,...j->...i', R.T, dq)

        # Squared Mahalanobis distance in rotated frame
        sq_dist = (dq_rot[..., 0] / ls1)**2 + (dq_rot[..., 1] / ls2)**2

    return np.exp(-0.5 * sq_dist)


def delta_penalty(q, bumps):
    """
    Compute total penalty from all bumps at point(s) q.

    delta(q) = sum_k alpha_k * RBF_k(q)

    Args:
        q: Array of shape (..., 2) with joint angles [q1, q2]
        bumps: List of bump dictionaries, each with keys:
               - 'mu': [mu1, mu2] center
               - 'ls': [ls1, ls2] lengthscales
               - 'alpha': strength
               - 'R': (optional) 2x2 rotation matrix

    Returns:
        Array of penalty values (shape same as q[..., 0])
    """
    if not bumps:
        # No bumps - zero penalty everywhere
        return np.zeros(q.shape[:-1])

    total_penalty = np.zeros(q.shape[:-1])

    for bump in bumps:
        # Get rotation matrix if present
        R = bump.get('R', None)
        rbf = rbf_value(q, bump['mu'], bump['ls'], R=R)
        total_penalty += bump['alpha'] * rbf

    return total_penalty


def h_value(q, box, bumps, use_smooth=False, smoothing_k=None):
    """
    Compute feasibility function h(q) = box_margin(q) - delta(q).

    A point is feasible if h(q) >= 0.

    Args:
        q: Array of shape (..., 2) with joint angles [q1, q2]
        box: Box dictionary
        bumps: List of bump dictionaries
        use_smooth: If True, use smooth_box_margin instead of box_margin (default: False)
        smoothing_k: Smoothing radius for corners. If None, auto-computed from box size.
                     Only used when use_smooth=True. (default: None = auto)

    Returns:
        Array of h values (shape same as q[..., 0])
    """
    if use_smooth:
        # Use smooth corner version
        if smoothing_k is None or smoothing_k == 'auto':
            # Auto-compute smoothing parameter
            smoothing_k = compute_auto_smoothing_k(box)
        margin = smooth_box_margin(q, box, smoothing_k=smoothing_k)
    else:
        # Use original sharp corner version
        margin = box_margin(q, box)

    penalty = delta_penalty(q, bumps)

    return margin - penalty


def feasible_mask_grid(box, bumps, grid_n=300, use_smooth=False, smoothing_k=None):
    """
    Generate grid and compute feasible mask over box region.

    Args:
        box: Box dictionary with 'q1_range' and 'q2_range'
        bumps: List of bump dictionaries
        grid_n: Grid resolution (number of points per dimension)
        use_smooth: If True, use smooth corners (default: False)
        smoothing_k: Smoothing radius for corners. If None, auto-computed. (default: None)

    Returns:
        Tuple (X, Y, H, M) where:
        - X, Y: meshgrid coordinates (shape: grid_n x grid_n)
        - H: h values at each grid point (shape: grid_n x grid_n)
        - M: Boolean mask, True where h >= 0 (shape: grid_n x grid_n)
    """
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    # Create grid
    q1_grid = np.linspace(q1_min, q1_max, grid_n)
    q2_grid = np.linspace(q2_min, q2_max, grid_n)
    X, Y = np.meshgrid(q1_grid, q2_grid)

    # Stack into (..., 2) array for vectorized evaluation
    q = np.stack([X, Y], axis=-1)

    # Evaluate h function
    H = h_value(q, box, bumps, use_smooth=use_smooth, smoothing_k=smoothing_k)

    # Compute feasible mask
    M = H >= 0.0

    return X, Y, H, M


def compute_feasible_fraction(box, bumps, grid_n=300):
    """
    Compute fraction of grid points that are feasible.

    Args:
        box: Box dictionary
        bumps: List of bump dictionaries
        grid_n: Grid resolution

    Returns:
        Float in [0, 1] representing feasible fraction
    """
    _, _, _, M = feasible_mask_grid(box, bumps, grid_n)
    return np.mean(M)


def check_2d_connectivity(box, bumps, dim1, dim2, grid_n=100, use_smooth=False, smoothing_k=None):
    """
    Check if feasible region is connected in a 2D projection.

    Projects the N-dimensional box onto the (dim1, dim2) plane and checks
    if the feasible region forms a single connected component.

    Args:
        box: Box dictionary with 'q1_range', 'q2_range', ..., 'qN_range'
        bumps: List of bump dictionaries
        dim1: First dimension index (1-indexed, e.g., 1 for q1)
        dim2: Second dimension index (1-indexed, e.g., 2 for q2)
        grid_n: Grid resolution for connectivity check
        use_smooth: If True, use smooth corners
        smoothing_k: Smoothing radius (None = auto)

    Returns:
        Dictionary with keys:
        - 'is_connected': Boolean, True if single connected component
        - 'num_components': Number of connected components
        - 'largest_fraction': Fraction of feasible points in largest component
        - 'feasible_fraction': Overall fraction of feasible points
    """
    # Extract the 2D box for this projection
    dim1_key = f'q{dim1}_range'
    dim2_key = f'q{dim2}_range'

    if dim1_key not in box or dim2_key not in box:
        raise ValueError(f"Box missing keys: {dim1_key} or {dim2_key}")

    q1_min, q1_max = box[dim1_key]
    q2_min, q2_max = box[dim2_key]

    # Create 2D grid
    q1_grid = np.linspace(q1_min, q1_max, grid_n)
    q2_grid = np.linspace(q2_min, q2_max, grid_n)
    Q1, Q2 = np.meshgrid(q1_grid, q2_grid)

    # For N-dimensional case, need to set other dimensions to center of their ranges
    # Get number of dimensions
    n_dims = 1
    while f'q{n_dims}_range' in box:
        n_dims += 1
    n_dims -= 1  # Actual number of dimensions

    # Create N-dimensional query points with other dims at box center
    q_list = []
    for d in range(1, n_dims + 1):
        if d == dim1:
            q_list.append(Q1)
        elif d == dim2:
            q_list.append(Q2)
        else:
            # Set other dimensions to center of their range
            dim_key = f'q{d}_range'
            center = (box[dim_key][0] + box[dim_key][1]) / 2.0
            q_list.append(np.full_like(Q1, center))

    # Stack into (..., N) array
    q = np.stack(q_list, axis=-1)

    # Evaluate feasibility
    H = h_value(q, box, bumps, use_smooth=use_smooth, smoothing_k=smoothing_k)
    M = H >= 0.0

    # Check connectivity using scipy
    labeled_array, num_components = ndimage.label(M)

    # Compute statistics
    feasible_fraction = np.mean(M)

    if num_components == 0:
        # No feasible points
        return {
            'is_connected': False,
            'num_components': 0,
            'largest_fraction': 0.0,
            'feasible_fraction': 0.0
        }

    # Find size of largest component
    component_sizes = ndimage.sum(M, labeled_array, range(1, num_components + 1))
    largest_size = np.max(component_sizes)
    largest_fraction = largest_size / M.size

    return {
        'is_connected': (num_components == 1),
        'num_components': int(num_components),
        'largest_fraction': float(largest_fraction),
        'feasible_fraction': float(feasible_fraction)
    }


def check_pairwise_connectivity(box, bumps, grid_n=100, use_smooth=False, smoothing_k=None):
    """
    Check if feasible region is connected in all pairwise 2D projections.

    For an N-dimensional box, checks all (N choose 2) pairs of dimensions.
    Useful for rejection sampling: if any projection is disconnected,
    the configuration may be problematic for sampling.

    Args:
        box: Box dictionary with 'q1_range', 'q2_range', ..., 'qN_range'
        bumps: List of bump dictionaries
        grid_n: Grid resolution for connectivity checks
        use_smooth: If True, use smooth corners
        smoothing_k: Smoothing radius (None = auto)

    Returns:
        Dictionary with keys:
        - 'all_connected': Boolean, True if all projections are connected
        - 'num_dimensions': Number of dimensions
        - 'num_pairs': Number of dimension pairs checked
        - 'pair_results': Dict mapping (dim1, dim2) -> connectivity result
        - 'disconnected_pairs': List of (dim1, dim2) tuples that are disconnected

    Example:
        >>> result = check_pairwise_connectivity(box, bumps)
        >>> if result['all_connected']:
        ...     print("Feasible region is connected in all projections")
        >>> else:
        ...     print(f"Disconnected in {len(result['disconnected_pairs'])} projections")
    """
    # Determine number of dimensions
    n_dims = 1
    while f'q{n_dims}_range' in box:
        n_dims += 1
    n_dims -= 1  # Actual number of dimensions

    if n_dims < 2:
        raise ValueError(f"Need at least 2 dimensions, got {n_dims}")

    # Check all pairs
    pair_results = {}
    disconnected_pairs = []

    for i in range(1, n_dims + 1):
        for j in range(i + 1, n_dims + 1):
            result = check_2d_connectivity(
                box, bumps, i, j, grid_n,
                use_smooth=use_smooth,
                smoothing_k=smoothing_k
            )
            pair_results[(i, j)] = result

            if not result['is_connected']:
                disconnected_pairs.append((i, j))

    all_connected = (len(disconnected_pairs) == 0)
    num_pairs = len(pair_results)

    return {
        'all_connected': all_connected,
        'num_dimensions': n_dims,
        'num_pairs': num_pairs,
        'pair_results': pair_results,
        'disconnected_pairs': disconnected_pairs
    }
