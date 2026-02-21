"""
Generates raw SIRS dataset for Implicit Neural Representations.

Key features:
1. Outputs RAW coordinates and SDF values (not normalized).
2. Uses temporary normalization for importance sampling (scale-adaptive).
3. Samples within box_limits only (no void region).
4. Stores base_limits and box_limits as metadata.

Output: A single .npz file with raw point clouds and metadata.
"""

import numpy as np
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count
import time
import argparse

# Add scone directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scone'))
from batch_joint_limit_sampler import generate_halton_samples, halton_sequence

from feasibility_checker import SIRSFeasibilityChecker
from sirs2d.sirs import h_value


def get_base_limits_array():
    """
    Get base joint limits for 4-DOF configuration (shoulder + elbow).
    Order matches HDF5 file: shoulder_flexion, shoulder_abduction, shoulder_rotation, elbow_flexion.

    Returns:
        np.ndarray: Shape (4, 2) with [min, max] for each joint in radians.
    """
    deg_to_rad = np.pi / 180.0
    base_limits = np.array([
        [-45*deg_to_rad, 150*deg_to_rad],   # shoulder_flexion_r
        [-60*deg_to_rad, 120*deg_to_rad],   # shoulder_abduction_r
        [-60*deg_to_rad, 60*deg_to_rad],    # shoulder_rotation_r
        [0*deg_to_rad, 130*deg_to_rad]      # elbow_flexion_r
    ])
    return base_limits


def generate_halton_batch(start_index, n_samples, n_dims):
    """
    Generate a batch of Halton samples starting from a specific index.

    Args:
        start_index (int): The starting index in the sequence (0-based).
        n_samples (int): Number of samples to generate.
        n_dims (int): Number of dimensions.

    Returns:
        np.ndarray: Shape (n_samples, n_dims) with values in [0, 1].
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    if n_dims > len(primes):
        raise ValueError(f"Too many dimensions ({n_dims}). Max supported: {len(primes)}")

    samples = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        for j in range(n_dims):
            samples[i, j] = halton_sequence(start_index + i + 1, primes[j])
    return samples


def process_sample(args):
    """
    Worker function to process a single SIRS sample.

    Workflow:
    1. Load Checker & Get box_limits.
    2. Adaptive Loop (sample within box_limits):
       a. Generate batch of raw points (Halton).
       b. Calculate raw h(q).
       c. Temporarily normalize for importance sampling.
       d. Prune (Surface + Anchors).
       e. Keep RAW coordinates and values (discard normalization).
       f. Accumulate results.
    3. Return raw data + metadata.

    Args:
        args (tuple): (sample_id, hdf5_path, initial_raw_k, n_joints, surface_threshold,
                       anchor_ratio, min_output_points, base_limits)

    Returns:
        tuple: (sample_id, result_dict)
    """
    (sample_id, hdf5_path, initial_raw_k, n_joints,
     surface_threshold, anchor_ratio, min_output_points, base_limits) = args

    print(f"Worker started for sample_id: {sample_id} (Target: {min_output_points}+ pts)")
    start_time = time.time()

    try:
        # 1. Load Checker & Get Limits
        checker = SIRSFeasibilityChecker(hdf5_path, sample_id=sample_id)
        box_limits_dict = checker.get_box_limits()
        box_limits_array = np.array(list(box_limits_dict.values()))
        lower_bounds = box_limits_array[:, 0]
        upper_bounds = box_limits_array[:, 1]

        # Calculate Normalization Constants (for importance sampling only)
        mu = (upper_bounds + lower_bounds) / 2.0
        R = np.max(upper_bounds - lower_bounds) / 2.0

        # Adaptive Sampling Loop (within box_limits)
        accumulated_outside_coords = []
        accumulated_outside_values = []
        accumulated_inside_coords = []
        accumulated_inside_values = []

        current_raw_count = 0
        total_kept_points = 0

        batch_size = initial_raw_k
        max_raw_points = 5_000_000

        iteration = 0

        while total_kept_points < min_output_points:
            iteration += 1
            if current_raw_count >= max_raw_points:
                print(f"Sample {sample_id}: Reached max raw points ({max_raw_points}). Stopping with {total_kept_points} pts.")
                break

            # 2a. Generate RAW Points (within box_limits)
            halton_01 = generate_halton_batch(current_raw_count, batch_size, n_joints)
            q_raw = lower_bounds + halton_01 * (upper_bounds - lower_bounds)

            # 2b. Calculate Raw h(q)
            # Box Margin
            all_margins = []
            for i in range(n_joints):
                q_i = q_raw[:, i]
                q_min, q_max = box_limits_array[i]
                all_margins.append(q_i - q_min)
                all_margins.append(q_max - q_i)

            stacked_margins = np.stack(all_margins, axis=1)
            nd_box_margin = np.min(stacked_margins, axis=1)

            # SIRS Constraints
            num_pairs = len(checker.pairwise_constraints)
            if num_pairs > 0:
                sirs_h_values = np.zeros((batch_size, num_pairs))
                for i, ((i1, i2), constraint) in enumerate(checker.pairwise_constraints.items()):
                    q_pair = q_raw[:, [i1, i2]]
                    h_vals_for_pair = h_value(q_pair, constraint['box'], constraint['bumps'],
                                              use_smooth=checker.use_smooth,
                                              smoothing_k=checker.smoothing_k)
                    sirs_h_values[:, i] = h_vals_for_pair

                all_constraints = np.hstack([nd_box_margin[:, np.newaxis], sirs_h_values])
                h_raw = np.min(all_constraints, axis=1)
            else:
                h_raw = nd_box_margin

            h_raw = h_raw[:, np.newaxis]

            # 2c. TEMPORARY Normalization (for importance sampling only)
            h_norm = h_raw / R

            # 2d. Importance Sampling (Pruning)
            surface_mask = np.abs(h_norm.flatten()) < surface_threshold
            surface_indices = np.where(surface_mask)[0]

            non_surface_indices = np.where(~surface_mask)[0]
            if len(non_surface_indices) > 0:
                num_anchors = int(len(non_surface_indices) * anchor_ratio)
                rng = np.random.default_rng(sample_id + current_raw_count)
                anchor_indices = rng.choice(non_surface_indices, size=num_anchors, replace=False)
            else:
                anchor_indices = np.array([], dtype=int)

            keep_indices = np.concatenate([surface_indices, anchor_indices])
            keep_indices.sort()

            # 2e. Keep RAW coordinates and values (discard normalization)
            batch_coords_raw = q_raw[keep_indices]
            batch_values_raw = h_raw[keep_indices]

            # Invert sign (SDF convention: < 0 inside)
            batch_values_raw = -batch_values_raw

            # NO CLAMPING for raw values (downstream decides)

            # Separate inside/outside based on sign
            outside_mask = batch_values_raw[:, 0] > 0
            accumulated_outside_coords.append(batch_coords_raw[outside_mask])
            accumulated_outside_values.append(batch_values_raw[outside_mask])
            accumulated_inside_coords.append(batch_coords_raw[~outside_mask])
            accumulated_inside_values.append(batch_values_raw[~outside_mask])

            count_in_batch = len(keep_indices)
            total_kept_points += count_in_batch
            current_raw_count += batch_size

            if total_kept_points < min_output_points:
                print(f"  Sample {sample_id} Iter {iteration}: Got {count_in_batch}. Total {total_kept_points}/{min_output_points}. Sampling more...")

        # Concatenate final points
        final_outside_coords = np.concatenate(accumulated_outside_coords, axis=0) if accumulated_outside_coords else np.empty((0, n_joints))
        final_outside_values = np.concatenate(accumulated_outside_values, axis=0) if accumulated_outside_values else np.empty((0, 1))
        final_inside_coords = np.concatenate(accumulated_inside_coords, axis=0) if accumulated_inside_coords else np.empty((0, n_joints))
        final_inside_values = np.concatenate(accumulated_inside_values, axis=0) if accumulated_inside_values else np.empty((0, 1))

        result = {
            'outside_coords': final_outside_coords,
            'inside_coords': final_inside_coords,
            'outside_values': final_outside_values,
            'inside_values': final_inside_values,
            'box_limits': box_limits_array
        }

        end_time = time.time()
        total_points = len(final_outside_coords) + len(final_inside_coords)
        print(f"Worker finished sample {sample_id}: {total_points} pts "
              f"({len(final_outside_coords)} out, {len(final_inside_coords)} in) "
              f"in {end_time - start_time:.2f}s")
        return (sample_id, result)

    except Exception as e:
        import traceback
        print(f"Worker for sample_id {sample_id} FAILED with error: {e}")
        traceback.print_exc()
        return (sample_id, None)


def main():
    # --- Configuration ---
    N_SAMPLES = 3
    RAW_K_POINTS = 500000    # Initial batch size
    TARGET_POINTS = 500000   # Target output size
    SURFACE_THRESHOLD = 0.1  # Normalized threshold (temporary)
    ANCHOR_RATIO = 0.1       # Keep 10% of non-surface points

    if len(sys.argv) >= 3:
        N_SAMPLES = int(sys.argv[1])
        RAW_K_POINTS = int(sys.argv[2])

    MIN_OUTPUT_POINTS = TARGET_POINTS // 2

    HDF5_PATH = '/media/liu/Lexar/data/ROMA/data/raw/sirs_samples_195992_4joints.h5'
    OUTPUT_PATH = f'/media/liu/Lexar/data/ROMA/data/raw_signed_dist_pc/raw_signed_dist_pc_N{N_SAMPLES}.npz'
    N_WORKERS = None
    # --- End Configuration ---

    print("=" * 70)
    print("SIRS Raw Data Generator")
    print("=" * 70)
    print(f"Input HDF5: {HDF5_PATH}")
    print(f"Output NPZ: {OUTPUT_PATH}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Adaptive Sampling: Batch {RAW_K_POINTS}, Target Output > {MIN_OUTPUT_POINTS} pts")
    print(f"Pruning: Surf < {SURFACE_THRESHOLD} (temp norm), Anchors {ANCHOR_RATIO*100}%")
    print(f"Output: RAW coordinates and SDF values (not normalized)")

    if not Path(HDF5_PATH).exists():
        print(f"Error: HDF5 file not found at {HDF5_PATH}")
        return

    # Get base_limits
    base_limits = get_base_limits_array()

    # Determine dimensions
    temp_checker = SIRSFeasibilityChecker(HDF5_PATH, sample_id=0)
    n_joints = len(temp_checker.joint_names)
    print(f"Detected {n_joints} joints: {temp_checker.joint_names}")

    if n_joints != 4:
        print(f"Warning: Expected 4 joints, got {n_joints}. base_limits may not match!")

    # Parallel Processing
    if N_WORKERS is None:
        workers = max(1, cpu_count() - 1)
    else:
        workers = N_WORKERS

    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_args = [(i, HDF5_PATH, RAW_K_POINTS, n_joints, SURFACE_THRESHOLD, ANCHOR_RATIO,
                    MIN_OUTPUT_POINTS, base_limits)
                   for i in range(N_SAMPLES)]
    results_list = [None] * N_SAMPLES

    with Pool(processes=workers) as pool:
        print(f"\nStarting parallel processing with {workers} workers...")
        results_iterator = pool.imap_unordered(process_sample, worker_args)

        for sample_id, result_data in results_iterator:
            if result_data is not None:
                results_list[sample_id] = result_data

    # Aggregate
    successful_results = [res for res in results_list if res is not None]
    if not successful_results:
        print("Error: No samples processed successfully.")
        return

    print(f"\nAggregating {len(successful_results)} samples...")

    final_dataset = {
        'outside_coords': np.array([r['outside_coords'] for r in successful_results], dtype=object),
        'inside_coords': np.array([r['inside_coords'] for r in successful_results], dtype=object),
        'outside_values': np.array([r['outside_values'] for r in successful_results], dtype=object),
        'inside_values': np.array([r['inside_values'] for r in successful_results], dtype=object),
        'base_limits': base_limits,
        'box_limits': np.stack([r['box_limits'] for r in successful_results], axis=0)
    }

    print("Final Shapes:")
    print(f"  Outside Coords: {final_dataset['outside_coords'].shape} (Object array, raw)")
    print(f"  Inside Coords:  {final_dataset['inside_coords'].shape} (Object array, raw)")
    print(f"  Outside Values: {final_dataset['outside_values'].shape} (Object array, raw)")
    print(f"  Inside Values:  {final_dataset['inside_values'].shape} (Object array, raw)")
    print(f"  Base Limits:    {final_dataset['base_limits'].shape}")
    print(f"  Box Limits:     {final_dataset['box_limits'].shape}")

    # Stats
    point_counts = [len(r['outside_coords']) + len(r['inside_coords']) for r in successful_results]
    outside_counts = [len(r['outside_coords']) for r in successful_results]
    inside_counts = [len(r['inside_coords']) for r in successful_results]
    print(f"\nPoint Statistics:")
    print(f"  Total Points - Min: {min(point_counts)}, Max: {max(point_counts)}, Mean: {np.mean(point_counts):.1f}")
    print(f"  Outside Points - Min: {min(outside_counts)}, Max: {max(outside_counts)}, Mean: {np.mean(outside_counts):.1f}")
    print(f"  Inside Points  - Min: {min(inside_counts)}, Max: {max(inside_counts)}, Mean: {np.mean(inside_counts):.1f}")

    print(f"\nSaving to {OUTPUT_PATH}...")
    np.savez_compressed(OUTPUT_PATH, **final_dataset)
    print("✓ Generation complete!")
    print("\nSee DATASET_FORMAT.md for data structure and access patterns.")

if __name__ == '__main__':
    main()
