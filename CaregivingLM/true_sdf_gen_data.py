"""
Generates a 'True SDF' dataset from SIRS samples using Surface Projection and Nearest Neighbor search.

Methodology:
1.  Generate Raw Points: Sample Halton within the box limits.
2.  Surface Extraction: Identify points near the zero-level set (h ~= 0).
3.  Refinement (Root Finding): Use gradient-based root finding (Newton-Raphson) to project these 'near' points EXACTLY onto the surface (h(q)=0).
4.  True SDF Calculation: For all volume points, compute the Euclidean distance to the nearest point in the refined surface set (using Faiss).
5.  Sign Assignment: Apply the sign from the original h(q) function but inverted (positive=outside, negative=inside).

This converts the "distorted" SIRS pseudo-SDF into a high-quality Euclidean SDF, which is much easier for neural networks to learn.
"""

import numpy as np
import h5py
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count
import time

import faiss

# Add scone directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scone'))
from batch_joint_limit_sampler import generate_halton_samples, halton_sequence
from feasibility_checker import SIRSFeasibilityChecker
from sirs2d.sirs import h_value, rbf_value

# --- Helper Functions for Multi-Constraint Surface Projection ---

def compute_box_constraint(q, box_limits):
    """
    Compute box margin h(q) and its gradient for N-dimensional box.

    Box margin = min over all dims of (q_i - min_i, max_i - q_i)
    Gradient points away from nearest wall (toward interior).

    Args:
        q: (N_points, n_dims) array of query points
        box_limits: (n_dims, 2) array with [min, max] for each dimension

    Returns:
        h_vals: (N_points,) - signed distance to nearest wall (negative outside)
        grads: (N_points, n_dims) - gradient vector
    """
    n_points, n_dims = q.shape

    # Compute distances to all walls for each dimension
    all_margins = []
    for i in range(n_dims):
        lower, upper = box_limits[i]
        dist_to_lower = q[:, i] - lower
        dist_to_upper = upper - q[:, i]
        all_margins.extend([dist_to_lower, dist_to_upper])

    # Stack and find minimum margin (closest wall)
    all_margins = np.stack(all_margins, axis=1)  # (N_points, 2*n_dims)
    h_vals = np.min(all_margins, axis=1)         # (N_points,)
    closest_wall_idx = np.argmin(all_margins, axis=1)  # (N_points,)

    # Determine gradient based on which wall is closest
    grads = np.zeros((n_points, n_dims))
    for pt_idx in range(n_points):
        wall_idx = closest_wall_idx[pt_idx]
        dim_idx = wall_idx // 2  # Which dimension
        is_lower = (wall_idx % 2 == 0)  # Lower or upper wall

        # If closer to lower wall, gradient is +1 (move away = increase q)
        # If closer to upper wall, gradient is -1 (move away = decrease q)
        grads[pt_idx, dim_idx] = 1.0 if is_lower else -1.0

    return h_vals, grads


def compute_pairwise_gradient(q_pair, box, bumps):
    """
    Complete gradient for h_pair(q) = box_margin(q) - delta(q)
    where q is 2D: [q1, q2]

    Args:
        q_pair: (N_points, 2) array
        box: dict with 'q1_range' and 'q2_range'
        bumps: list of bump dicts with 'mu', 'ls', 'alpha'

    Returns:
        grads: (N_points, 2) array - full gradient including box and bump components
    """
    n_points = q_pair.shape[0]
    grads = np.zeros((n_points, 2))

    # Part 1: Gradient from 2D box margin
    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    # Compute distances to all 4 walls
    dist_to_q1_lower = q_pair[:, 0] - q1_min
    dist_to_q1_upper = q1_max - q_pair[:, 0]
    dist_to_q2_lower = q_pair[:, 1] - q2_min
    dist_to_q2_upper = q2_max - q_pair[:, 1]

    # Find closest wall for each point
    all_dists = np.stack([dist_to_q1_lower, dist_to_q1_upper,
                          dist_to_q2_lower, dist_to_q2_upper], axis=1)
    closest_wall = np.argmin(all_dists, axis=1)

    # Set gradient based on closest wall
    # Wall 0 (q1 lower): grad = [+1, 0]
    # Wall 1 (q1 upper): grad = [-1, 0]
    # Wall 2 (q2 lower): grad = [0, +1]
    # Wall 3 (q2 upper): grad = [0, -1]
    grads[closest_wall == 0, 0] = 1.0
    grads[closest_wall == 1, 0] = -1.0
    grads[closest_wall == 2, 1] = 1.0
    grads[closest_wall == 3, 1] = -1.0

    # Part 2: Gradient from bumps (penalty term)
    # h = margin - penalty, so d(h)/dq = d(margin)/dq - d(penalty)/dq
    # RBF = exp(-0.5 * ((q - mu) / ls)^2)
    # d(RBF)/dq = RBF * (-(q - mu) / ls^2)

    for bump in bumps:
        mu = np.array(bump['mu'])  # (2,)
        ls = np.array(bump['ls'])  # (2,)
        alpha = bump['alpha']  # scalar

        # Compute RBF value
        dq = q_pair - mu  # (N, 2)
        dq_scaled = dq / ls
        sq_dist = np.sum(dq_scaled**2, axis=1, keepdims=True)  # (N, 1)
        rbf_val = np.exp(-0.5 * sq_dist)  # (N, 1)

        # Gradient of RBF: d(RBF)/dq = RBF * [-(q - mu) / ls^2]
        rbf_grad = rbf_val * (-dq / (ls**2))  # (N, 2)

        # Since h = margin - penalty, d(h)/dq = d(margin)/dq - alpha * d(RBF)/dq
        grads -= alpha * rbf_grad

    return grads


def compute_h_and_active_gradients(q, checker, box_limits):
    """
    Compute h(q) = min(all constraints) and gradient of active constraint.

    For each point, determines which constraint is active (has minimum h value)
    and returns that constraint's gradient.

    Args:
        q: (N_points, n_dims) array of query points
        checker: SIRSFeasibilityChecker instance
        box_limits: (n_dims, 2) array with [min, max] for each dimension

    Returns:
        h_total: (N_points,) array - minimum h value for each point
        active_grads: (N_points, n_dims) array - gradient of active constraint
    """
    n_points, n_dims = q.shape

    # 1. Compute N-dimensional box constraint
    box_h, box_grad = compute_box_constraint(q, box_limits)

    # 2. Compute all pairwise constraints
    pairwise_hs = [box_h]  # Start w box constraint
    pairwise_grads = [box_grad]

    for (i1, i2), constraint in checker.pairwise_constraints.items():
        q_pair = q[:, [i1, i2]]  # Extract 2D slice

        # Compute h value for this pair
        h_pair = h_value(q_pair, constraint['box'], constraint['bumps'],
                        use_smooth=checker.use_smooth,
                        smoothing_k=checker.smoothing_k)

        # Compute gradient in 2D, then embed in full N space
        grad_2d = compute_pairwise_gradient(q_pair, constraint['box'],
                                           constraint['bumps'])

        # Embed 2D gradient into full N-dimensional space
        grad_full = np.zeros((n_points, n_dims))
        grad_full[:, i1] = grad_2d[:, 0]
        grad_full[:, i2] = grad_2d[:, 1]

        pairwise_hs.append(h_pair)
        pairwise_grads.append(grad_full)

    # 3. Stack all constraint values and find minimum
    all_h_vals = np.column_stack(pairwise_hs)   # (N_points, num_constraints)
    active_idx = np.argmin(all_h_vals, axis=1)  # (N_points,) - which constraint is active
    h_total = np.min(all_h_vals, axis=1)        # (N_points,) - the actual h value

    # 4. Select gradient corresponding to active constraint for each point
    all_grads = np.stack(pairwise_grads)  # (num_constraints, N_points, n_dims)
    active_grads = np.zeros((n_points, n_dims))
    for pt_idx in range(n_points):
        active_grads[pt_idx] = all_grads[active_idx[pt_idx], pt_idx]

    return h_total, active_grads


def project_to_surface_global(q_init, checker, box_limits, max_iters=20, tolerance=1e-4, damping=0.8):
    """
    Project points to h(q) = 0 surface using gradient descent on active constraints.

    This properly handles the multi-constraint surface defined by:
    h(q) = min(box_margin(q), h_pair1(q), h_pair2(q), ...) = 0

    Args:
        q_init: (N_points, n_dims) initial points
        checker: SIRSFeasibilityChecker instance
        box_limits: (n_dims, 2) array with [min, max] for each dimension
        max_iters: maximum gradient descent iterations
        tolerance: convergence threshold for |h(q)|
        damping: step size damping factor (0.5-0.9 recommended)

    Returns:
        q_proj: (N_points, n_dims) projected points on surface
        converged_mask: (N_points,) boolean array indicating which points converged
    """
    q = q_init.copy()
    n_points = q.shape[0]
    converged_mask = np.zeros(n_points, dtype=bool)

    for iteration in range(max_iters):
        # 1. Evaluate h and active gradients for all non-converged points
        h_total, active_grads = compute_h_and_active_gradients(q, checker, box_limits)

        # 2. Check convergence
        newly_converged = np.abs(h_total) < tolerance
        converged_mask |= newly_converged

        if np.all(converged_mask):
            print(f"  All points converged at iteration {iteration}")
            break

        # 3. GD step for non-converged points
        # Move toward h = 0: q_new = q_old - h * grad / |grad|^2
        grad_norm_sq = np.sum(active_grads**2, axis=1, keepdims=True)
        grad_norm_sq = np.maximum(grad_norm_sq, 1e-8)  # Avoid division by zero

        step = damping * h_total[:, None] * active_grads / grad_norm_sq
        q = q - step

        # 4. Clamp to box limits (prevent escaping valid region)
        # ONLY clamp if we're moving away from the box
        for dim in range(q.shape[1]):
            lower, upper = box_limits[dim]
            q[:, dim] = np.clip(q[:, dim], lower, upper)

    # print convergence statistics
    conv_rate = np.sum(converged_mask) / n_points * 100
    print(f"  Projection converged: {np.sum(converged_mask)}/{n_points} ({conv_rate:.1f}%)")

    return q, converged_mask

def generate_halton_batch(start_index, n_samples, n_dims):
    # (Same as before - simplified copy)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    samples = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        for j in range(n_dims):
            samples[i, j] = halton_sequence(start_index + i + 1, primes[j])
    return samples

def process_sample_true_sdf(args):
    """
    Worker function to process a single SIRS sample into TRUE SDF.
    """
    (sample_id, hdf5_path, n_raw_points, n_joints, n_surface_points) = args
    
    print(f"Worker {sample_id}: Started.")
    start_time = time.time()
    
    try:
        checker = SIRSFeasibilityChecker(hdf5_path, sample_id=sample_id)
        box_limits = checker.get_box_limits()
        box_arr = np.array(list(box_limits.values()))
        lower = box_arr[:, 0]
        upper = box_arr[:, 1]
        
        # 1. Generate MASSIVE batch of raw points to find candidate surface points
        # We need good coverage to find the zero-crossing.
        # Since Halton is deterministic, we just generate a big block.
        print(f"Worker {sample_id}: Generating {n_raw_points} raw points...")
        halton = generate_halton_batch(0, n_raw_points, n_joints)
        q_raw = lower + halton * (upper - lower)
        
        # Compute Pseudo-H (Original SIRS value)
        # We need this to (a) find surface candidates, (b) assign signs later.
        
        # Reconstruct full constraint set for h_value call
        # We need to aggregate h = min(box, min(pairs)).
        
        # Let's compute h_raw fully.
        all_margins = []
        for i in range(n_joints):
            q_i = q_raw[:, i]
            q_min, q_max = box_arr[i]
            all_margins.append(q_i - q_min)
            all_margins.append(q_max - q_i)
        nd_box_margin = np.min(np.stack(all_margins, axis=1), axis=1)
        
        num_pairs = len(checker.pairwise_constraints)
        if num_pairs > 0:
            sirs_h_vals = np.zeros((n_raw_points, num_pairs))

            for i, ((i1, i2), constraint) in enumerate(checker.pairwise_constraints.items()):
                q_pair = q_raw[:, [i1, i2]]
                h_p = h_value(q_pair, constraint['box'], constraint['bumps'],
                              use_smooth=checker.use_smooth, smoothing_k=checker.smoothing_k)
                sirs_h_vals[:, i] = h_p

            all_vals = np.hstack([nd_box_margin[:, np.newaxis], sirs_h_vals])
            h_raw = np.min(all_vals, axis=1)
        else:
            h_raw = nd_box_margin

        # Identify Surface Candidates
        # Points where |h| is small - these are close to the surface
        abs_h = np.abs(h_raw)
        candidate_indices = np.argsort(abs_h)[:n_surface_points]
        q_candidates = q_raw[candidate_indices]

        print(f"Worker {sample_id}: Projecting {len(q_candidates)} candidates to surface...")

        # Project to Surface using Multi-Constraint Gradient Descent
        # This properly handles h(q) = min(all constraints) = 0
        q_surface, converged_mask = project_to_surface_global(
            q_candidates, checker, box_arr,
            max_iters=20, tolerance=1e-4, damping=0.8
        )

        # Filter out points that didn't converge (optional, but recommended)
        if not np.all(converged_mask):
            print(f"  Warning: {np.sum(~converged_mask)} points did not converge. Using best effort.")
            # We keep them anyway as they're still close to the surface
            
        # 5. Compute True SDF using Faiss
        print(f"Worker {sample_id}: Computing True SDF with Faiss...")


        # Build Index on Surface Points
        # Faiss expects float32
        db_vecs = q_surface.astype(np.float32)
        query_vecs = q_raw.astype(np.float32)
        
        index = faiss.IndexFlatL2(n_joints)
        index.add(db_vecs)
        
        # Search for nearest neighbor (k=1)
        # D_sq is squared distance
        D_sq, I = index.search(query_vecs, 1)
        
        true_dist = np.sqrt(D_sq).flatten()
        
        # Assign Sign
        final_sdf = true_dist.copy()
        # Flip sign for inside points
        inside_mask = (h_raw > 0)
        final_sdf[inside_mask] = -final_sdf[inside_mask]
        
        # Norm (Metric Preserving)
        # We need to normalize the COORDINATES and the DISTANCES by the same factor R.
        min_vec = lower
        max_vec = upper
        mu = (max_vec + min_vec) / 2.0
        R = np.max(max_vec - min_vec) / 2.0
        
        q_norm = (q_raw - mu) / R
        sdf_norm = final_sdf / R
        
        # Clamp
        # TODO: Is this necessary for true sdf???
        sdf_clamped = np.clip(sdf_norm, -1.0, 1.0)
        
        # Package Result
        norm_min_bound = (min_vec - mu) / R
        norm_max_bound = (max_vec - mu) / R
        norm_bounds = np.stack([norm_min_bound, norm_max_bound], axis=1)
        
        result = {
            'coords': q_norm,
            'values': sdf_clamped[:, np.newaxis], # (N, 1)
            'center': mu,
            'scale': R,
            'norm_bounds': norm_bounds
        }
        
        end_time = time.time()
        print(f"Worker {sample_id}: Finished {n_raw_points} pts in {end_time - start_time:.2f}s")
        return (sample_id, result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (sample_id, None)

def main():
    # --- Configuration ---
    N_SAMPLES = 16
    N_RAW_POINTS = 500000     # Total volume points to process
    N_SURFACE_POINTS = 100000 # Number of points to refine for surface
    
    if len(sys.argv) >= 3:
        N_SAMPLES = int(sys.argv[1])
        N_RAW_POINTS = int(sys.argv[2])
        
    HDF5_PATH = '/media/liu/Lexar/data/ROMA/data/raw/sirs_samples_195992_4joints.h5'
    OUTPUT_PATH = f'/home/liu/dev/emprise/SDF/data/mini/true_sdf_N{N_SAMPLES}.npy'
    N_WORKERS = max(1, cpu_count() - 2)
    
    print("=" * 70)
    print("True SDF Generator (Surface Projection + Faiss)")
    print("=" * 70)
    print(f"Input: {HDF5_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Volume Points: {N_RAW_POINTS}")
    print(f"Surface Anchors: {N_SURFACE_POINTS}")
    
    if not Path(HDF5_PATH).exists():
        print(f"Error: HDF5 file not found at {HDF5_PATH}")
        return

    # Determine dimensions
    temp_checker = SIRSFeasibilityChecker(HDF5_PATH, sample_id=0)
    n_joints = len(temp_checker.joint_names)
    print(f"Detected {n_joints} joints.")

    worker_args = [(i, HDF5_PATH, N_RAW_POINTS, n_joints, N_SURFACE_POINTS) 
                   for i in range(N_SAMPLES)]
    results_list = [None] * N_SAMPLES

    with Pool(processes=N_WORKERS) as pool:
        results_iterator = pool.imap_unordered(process_sample_true_sdf, worker_args)
        for sample_id, result_data in results_iterator:
            if result_data is not None:
                results_list[sample_id] = result_data

    # Aggregate
    successful = [r for r in results_list if r is not None]
    if not successful:
        print("No results.")
        return

    final_dataset = {
        'coords': np.array([r['coords'] for r in successful], dtype=object),
        'values': np.array([r['values'] for r in successful], dtype=object),
        'center': np.stack([r['center'] for r in successful], axis=0),
        'scale': np.stack([r['scale'] for r in successful], axis=0),
        'norm_bounds': np.stack([r['norm_bounds'] for r in successful], axis=0)
    }
    
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, final_dataset)
    print("✓ True SDF Generation complete!")

if __name__ == '__main__':
    main()
