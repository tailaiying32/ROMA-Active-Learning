"""
SIRS-enhanced batch joint limit sampler.

Extends the original batch_joint_limit_sampler.py by adding SIRS bumps
to model pairwise joint coupling constraints.
"""

import numpy as np
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Add scone directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scone'))

from batch_joint_limit_sampler import generate_procedural_joint_limits
from sirs2d.sampler import calibrate_alpha
import sirs_sampling_config as config


def sample_sirs_parameters(rng):
    """
    Sample SIRS bump generation parameters from configured ranges.

    Returns dictionary with sampled parameters for diversity.
    """
    params = {}

    # Edge bias beta parameter
    if config.EDGE_BIAS:
        params['edge_bias_beta_alpha'] = rng.uniform(
            config.EDGE_BIAS_BETA_MIN,
            config.EDGE_BIAS_BETA_MAX
        )
    else:
        params['edge_bias_beta_alpha'] = None

    # Alpha (bump strength) log-normal parameters
    params['alpha_lognormal_mean'] = rng.uniform(
        config.ALPHA_LOGNORMAL_MEAN_MIN,
        config.ALPHA_LOGNORMAL_MEAN_MAX
    )
    params['alpha_lognormal_sigma'] = rng.uniform(
        config.ALPHA_LOGNORMAL_SIGMA_MIN,
        config.ALPHA_LOGNORMAL_SIGMA_MAX
    )

    # Lengthscale parameters
    params['lengthscale_fraction'] = rng.uniform(
        config.LENGTHSCALE_FRACTION_MIN,
        config.LENGTHSCALE_FRACTION_MAX
    )
    params['lengthscale_sigma'] = rng.uniform(
        config.LENGTHSCALE_SIGMA_MIN,
        config.LENGTHSCALE_SIGMA_MAX
    )

    # Grid resolution (fixed for performance)
    params['grid_resolution'] = config.GRID_RESOLUTION

    return params


def generate_bumps_with_params(box, rng, params):
    """
    Generate SIRS bumps using sampled parameters.

    Similar to sirs2d.sampler.sample_bumps() but uses our sampled parameters.

    Args:
        box: Dictionary with 'q1_range' and 'q2_range'
        rng: numpy random number generator
        params: Dictionary of sampled SIRS parameters

    Returns:
        List of bump dictionaries
    """
    # Sample number of bumps
    K = rng.integers(config.NUM_BUMPS_MIN, config.NUM_BUMPS_MAX + 1)

    q1_min, q1_max = box['q1_range']
    q2_min, q2_max = box['q2_range']

    q1_width = q1_max - q1_min
    q2_width = q2_max - q2_min

    bumps = []

    for _ in range(K):
        # Sample bump center
        if params['edge_bias_beta_alpha'] is not None:
            # Beta distribution for edge bias
            beta_alpha = params['edge_bias_beta_alpha']
            u1 = rng.beta(beta_alpha, beta_alpha)
            u2 = rng.beta(beta_alpha, beta_alpha)
            mu1 = q1_min + u1 * q1_width
            mu2 = q2_min + u2 * q2_width
        else:
            # Uniform placement
            mu1 = rng.uniform(q1_min, q1_max)
            mu2 = rng.uniform(q2_min, q2_max)

        # Sample lengthscales using sampled parameters
        mean_ls1 = params['lengthscale_fraction'] * q1_width
        mean_ls2 = params['lengthscale_fraction'] * q2_width

        ls_sigma = params['lengthscale_sigma']
        ls1 = mean_ls1 * np.exp(rng.normal(0, ls_sigma))
        ls2 = mean_ls2 * np.exp(rng.normal(0, ls_sigma))

        # Clamp lengthscales
        ls1 = np.clip(ls1,
                     config.LENGTHSCALE_MIN_ABS_FRACTION * q1_width,
                     config.LENGTHSCALE_MAX_ABS_FRACTION * q1_width)
        ls2 = np.clip(ls2,
                     config.LENGTHSCALE_MIN_ABS_FRACTION * q2_width,
                     config.LENGTHSCALE_MAX_ABS_FRACTION * q2_width)

        # Sample alpha using sampled parameters
        log_alpha = rng.normal(
            params['alpha_lognormal_mean'],
            params['alpha_lognormal_sigma']
        )
        alpha = np.exp(log_alpha)

        # Build bump dictionary
        bump_dict = {
            'mu': np.array([mu1, mu2]),
            'ls': np.array([ls1, ls2]),
            'alpha': alpha
        }

        # Add rotation if enabled
        if config.ENABLE_ROTATION:
            theta = rng.uniform(0, 2 * np.pi)
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


def add_sirs_bumps_to_sample(joint_limit_set, rng, joint_pairs=None,
                             target_feas_range=(0.5, 0.9)):
    """
    Add SIRS bumps to a single joint limit set for specified joint pairs.

    Now samples SIRS parameters from configured ranges for diversity.

    Args:
        joint_limit_set: Dictionary with 'id', 'joint_limits', 'metadata'
        rng: numpy random generator
        joint_pairs: List of (joint1, joint2) tuples (None = use config)
        target_feas_range: (min, max) range for target feasibility

    Returns:
        Enhanced joint_limit_set with 'sirs_bumps', 'sirs_metadata', and 'sirs_params'
    """
    if joint_pairs is None:
        joint_pairs = config.MAIN_JOINT_PAIRS

    # Sample SIRS parameters once per sample (shared across all pairs)
    sirs_params = sample_sirs_parameters(rng)

    sirs_bumps = {}
    sirs_metadata = {}

    for (joint1, joint2) in joint_pairs:
        # Extract box limits for this pair
        if joint1 not in joint_limit_set['joint_limits'] or \
           joint2 not in joint_limit_set['joint_limits']:
            print(f"Warning: Joint pair ({joint1}, {joint2}) not in joint_limits, skipping")
            continue

        lower1, upper1 = joint_limit_set['joint_limits'][joint1]
        lower2, upper2 = joint_limit_set['joint_limits'][joint2]

        box = {
            'q1_range': (lower1, upper1),
            'q2_range': (lower2, upper2)
        }

        # Sample target feasibility for this pair
        target_feas = rng.uniform(target_feas_range[0], target_feas_range[1])

        # Generate bumps using sampled parameters
        bumps = generate_bumps_with_params(box, rng, sirs_params)

        # Use sampled grid resolution for calibration
        grid_n = sirs_params['grid_resolution']

        # Calibrate to target feasibility
        bumps = calibrate_alpha(
            box, bumps, target_feas,
            grid_n=grid_n,
            max_iter=config.CALIBRATION_MAX_ITER,
            tolerance=config.CALIBRATION_TOLERANCE
        )

        # Compute actual achieved feasibility
        from sirs2d.sirs import compute_feasible_fraction
        actual_feas = compute_feasible_fraction(
            box, bumps, grid_n=grid_n
        )

        # Store bumps
        pair_key = (joint1, joint2)
        sirs_bumps[pair_key] = bumps

        # Store metadata for this pair
        sirs_metadata[pair_key] = {
            'n_bumps': len(bumps),
            'target_feasibility': float(target_feas),
            'actual_feasibility': float(actual_feas),
            'box_width_q1': float(upper1 - lower1),
            'box_width_q2': float(upper2 - lower2),
        }

    # Add SIRS data to joint_limit_set
    enhanced_set = joint_limit_set.copy()
    enhanced_set['sirs_bumps'] = sirs_bumps
    enhanced_set['sirs_metadata'] = sirs_metadata
    enhanced_set['sirs_params'] = sirs_params  # Store sampled parameters

    return enhanced_set


def _process_single_sample(args):
    """
    Worker function for parallel processing of a single sample.

    Args:
        args: Tuple of (sample_index, joint_limit_set, random_seed, joint_pairs,
                       target_feas_range, reject_disconnected, max_rejection_attempts)

    Returns:
        Tuple of (sample_index, enhanced_sample or None, rejection_stats_dict)
    """
    (i, jls, random_seed, joint_pairs, target_feas_range,
     reject_disconnected, max_rejection_attempts) = args

    # Create independent RNG for this worker
    rng = np.random.default_rng(random_seed)

    # Try generating SIRS bumps with rejection sampling
    accepted = False
    rejection_stats = {'rejected': 0, 'accepted': 0, 'failed': 0}
    enhanced = None

    for attempt in range(max_rejection_attempts):
        enhanced = add_sirs_bumps_to_sample(
            jls, rng,
            joint_pairs=joint_pairs,
            target_feas_range=target_feas_range
        )

        # Check connectivity if rejection sampling is enabled
        if not reject_disconnected:
            accepted = True
            rejection_stats['accepted'] += 1
            break

        # Check if any pair is disconnected
        from validate_sirs_samples import check_sample_connectivity
        conn_results = check_sample_connectivity(
            enhanced,
            grid_n=config.CONNECTIVITY_GRID_N,
            use_smooth=config.USE_SMOOTH_CORNERS,
            early_exit=True  # Use early exit optimization
        )

        is_disconnected = any(not result['is_connected']
                            for result in conn_results.values())

        if not is_disconnected:
            accepted = True
            rejection_stats['accepted'] += 1
            break
        else:
            rejection_stats['rejected'] += 1

    if accepted:
        return (i, enhanced, rejection_stats)
    else:
        # Failed to generate connected sample
        rejection_stats['failed'] += 1
        if config.FAIL_ON_DISCONNECTED:
            return (i, None, rejection_stats)  # Sample lost
        else:
            return (i, enhanced, rejection_stats)  # Use last attempt with warning


def generate_sirs_enhanced_joint_limits(
    n_samples_per_batch=None,
    coverage_factor=None,
    min_range_factor=None,
    main_dofs_only=None,
    seed=None,
    enable_sirs=None,
    target_feasibility_range=None,
    joint_pairs=None,
    sirs_seed=None,
    reject_disconnected=True,
    max_rejection_attempts=10,
    n_workers=None,
    verbose=True
):
    """
    Generate joint limit sets with SIRS bumps for pairwise coupling.

    Args:
        n_samples_per_batch: Samples per batch (None = use config)
        coverage_factor: Fraction of base range (None = use config)
        n_workers: Number of parallel workers (None = auto-detect, 1 = serial)
        min_range_factor: Min range fraction (None = use config)
        main_dofs_only: Only sample main DOFs (None = use config)
        seed: Random seed for batch sampler (None = use config)
        enable_sirs: Add SIRS bumps (None = use config)
        target_feasibility_range: (min, max) for target feasibility (None = use config)
        joint_pairs: List of (j1, j2) tuples (None = use config)
        sirs_seed: Random seed for SIRS (None = use batch seed)
        verbose: Print progress

    Returns:
        List of enhanced joint limit dictionaries with SIRS bumps
    """
    # Use config defaults if not specified
    n_samples_per_batch = n_samples_per_batch or config.N_SAMPLES_PER_BATCH
    coverage_factor = coverage_factor or config.COVERAGE_FACTOR
    min_range_factor = min_range_factor or config.MIN_RANGE_FACTOR
    main_dofs_only = main_dofs_only if main_dofs_only is not None else config.MAIN_DOFS_ONLY
    seed = seed or config.SEED
    enable_sirs = enable_sirs if enable_sirs is not None else config.ENABLE_SIRS
    joint_pairs = joint_pairs or config.MAIN_JOINT_PAIRS
    sirs_seed = sirs_seed or config.SIRS_SEED or seed

    if target_feasibility_range is None:
        target_feasibility_range = (config.TARGET_FEASIBILITY_MIN,
                                   config.TARGET_FEASIBILITY_MAX)

    if verbose:
        print("=" * 70)
        print("SIRS-Enhanced Joint Limit Sampling")
        print("=" * 70)
        print(f"\n[Batch Sampler]")
        print(f"  Samples per batch: {n_samples_per_batch}")
        print(f"  Coverage factor: {coverage_factor}")
        print(f"  Min range factor: {min_range_factor}")
        print(f"  Seed: {seed}")
        print(f"\n[SIRS]")
        print(f"  Enabled: {enable_sirs}")
        if enable_sirs:
            print(f"  Target feasibility range: {target_feasibility_range}")
            print(f"  Joint pairs: {len(joint_pairs)}")
            print(f"  SIRS seed: {sirs_seed}")
        print()

    # Step 1: Generate base box limits
    joint_limit_sets = generate_procedural_joint_limits(
        n_samples_per_batch=n_samples_per_batch,
        coverage_factor=coverage_factor,
        min_range_factor=min_range_factor,
        main_dofs_only=main_dofs_only,
        seed=seed
    )

    if not enable_sirs:
        if verbose:
            print("\nSIRS disabled, returning box limits only")
        return joint_limit_sets

    # Step 2: Add SIRS bumps to each sample with rejection sampling
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one core free

    use_parallel = n_workers > 1 and len(joint_limit_sets) > 1

    if verbose:
        print(f"\nAdding SIRS bumps to {len(joint_limit_sets)} samples...")
        if reject_disconnected:
            print(f"  Rejection sampling enabled (max {max_rejection_attempts} attempts per sample)")
        if use_parallel:
            print(f"  Parallel processing: {n_workers} workers")
        else:
            print(f"  Serial processing (1 worker)")

    # Generate random seeds for each sample (for reproducibility with parallel processing)
    seed_rng = np.random.default_rng(sirs_seed)
    sample_seeds = seed_rng.integers(0, 2**31, size=len(joint_limit_sets))

    enhanced_sets = []
    rejection_stats = {'rejected': 0, 'accepted': 0, 'failed': 0}

    if use_parallel:
        # Parallel processing using multiprocessing.Pool
        worker_args = [
            (i, jls, sample_seeds[i], joint_pairs, target_feasibility_range,
             reject_disconnected, max_rejection_attempts)
            for i, jls in enumerate(joint_limit_sets)
        ]

        with Pool(processes=n_workers) as pool:
            # Use imap_unordered for better performance (doesn't preserve order)
            results = pool.imap_unordered(_process_single_sample, worker_args, chunksize=10)

            # Collect results with progress reporting
            completed = 0
            results_dict = {}
            for sample_idx, enhanced, stats in results:
                results_dict[sample_idx] = enhanced
                rejection_stats['rejected'] += stats['rejected']
                rejection_stats['accepted'] += stats['accepted']
                rejection_stats['failed'] += stats['failed']

                completed += 1
                if verbose and (completed % 100 == 0 or completed == len(joint_limit_sets)):
                    print(f"  Processed {completed}/{len(joint_limit_sets)} samples...", end='\r')

        # Sort results by sample index and filter out None (failed samples)
        for i in range(len(joint_limit_sets)):
            if i in results_dict and results_dict[i] is not None:
                enhanced_sets.append(results_dict[i])

    else:
        # Serial processing (original logic for n_workers=1)
        rng = np.random.default_rng(sirs_seed)
        for i, jls in enumerate(joint_limit_sets):
            if verbose and (i % 10 == 0 or i == len(joint_limit_sets) - 1):
                print(f"  Processing sample {i+1}/{len(joint_limit_sets)}...", end='\r')

            # Process single sample
            sample_idx, enhanced, stats = _process_single_sample(
                (i, jls, sample_seeds[i], joint_pairs, target_feasibility_range,
                 reject_disconnected, max_rejection_attempts)
            )

            rejection_stats['rejected'] += stats['rejected']
            rejection_stats['accepted'] += stats['accepted']
            rejection_stats['failed'] += stats['failed']

            if enhanced is not None:
                enhanced_sets.append(enhanced)
            elif verbose and config.FAIL_ON_DISCONNECTED:
                print(f"\n  ✗ Sample {i} failed after {max_rejection_attempts} attempts - SKIPPING (disconnected)")

    if verbose:
        print(f"\n✓ Generated {len(enhanced_sets)} SIRS-enhanced joint limit sets")
        if reject_disconnected:
            total_attempts = rejection_stats['accepted'] + rejection_stats['rejected']
            acceptance_rate = rejection_stats['accepted'] / total_attempts * 100 if total_attempts > 0 else 0
            print(f"  Rejection stats: {rejection_stats['rejected']} rejected, "
                  f"{rejection_stats['accepted']} accepted ({acceptance_rate:.1f}% acceptance rate)")
            if rejection_stats['failed'] > 0:
                if config.FAIL_ON_DISCONNECTED:
                    print(f"  ✗ CRITICAL: {rejection_stats['failed']} samples SKIPPED due to disconnection")
                    print(f"  Final count: {len(enhanced_sets)} samples (lost {rejection_stats['failed']} samples)")
                else:
                    print(f"  Warning: {rejection_stats['failed']} samples used last attempt (may be disconnected)")

    return enhanced_sets


def print_sample_structure(sample, sample_id=0):
    """Print the structure of a SIRS-enhanced sample for validation."""
    print("\n" + "=" * 70)
    print(f"SIRS-Enhanced Sample Structure (ID: {sample_id})")
    print("=" * 70)

    print(f"\n[Basic Info]")
    print(f"  Sample ID: {sample['id']}")
    print(f"  Number of joints: {len(sample['joint_limits'])}")

    print(f"\n[Metadata]")
    for key, value in sample['metadata'].items():
        print(f"  {key}: {value}")

    print(f"\n[Joint Limits] (first 3 joints)")
    for i, (joint_name, (lower, upper)) in enumerate(list(sample['joint_limits'].items())[:3]):
        range_deg = np.degrees(upper - lower)
        print(f"  {joint_name}:")
        print(f"    [{np.degrees(lower):.1f}°, {np.degrees(upper):.1f}°] (range: {range_deg:.1f}°)")
    print(f"  ... ({len(sample['joint_limits']) - 3} more joints)")

    if 'sirs_bumps' in sample:
        print(f"\n[SIRS Bumps]")
        print(f"  Number of joint pairs: {len(sample['sirs_bumps'])}")

        for pair, bumps in sample['sirs_bumps'].items():
            j1, j2 = pair
            metadata = sample['sirs_metadata'][pair]

            print(f"\n  Pair: {j1} × {j2}")
            print(f"    Number of bumps: {len(bumps)}")
            print(f"    Target feasibility: {metadata['target_feasibility']:.2%}")
            print(f"    Actual feasibility: {metadata['actual_feasibility']:.2%}")
            print(f"    Box size: [{metadata['box_width_q1']:.2f}° × {metadata['box_width_q2']:.2f}°]")

            if len(bumps) > 0:
                print(f"    First bump:")
                b = bumps[0]
                print(f"      mu: [{np.degrees(b['mu'][0]):.1f}°, {np.degrees(b['mu'][1]):.1f}°]")
                print(f"      ls: [{np.degrees(b['ls'][0]):.1f}°, {np.degrees(b['ls'][1]):.1f}°]")
                print(f"      alpha: {b['alpha']:.2f}")
                if 'theta' in b:
                    print(f"      theta: {np.degrees(b['theta']):.1f}°")

    print("=" * 70)


if __name__ == '__main__':
    # Test: Generate 10 samples with SIRS
    print("Testing SIRS batch sampler...")

    samples = generate_sirs_enhanced_joint_limits(
        n_samples_per_batch=1,  # 1 sample per batch = 10 total samples
        enable_sirs=True,
        verbose=True
    )

    print(f"\n✓ Generated {len(samples)} samples")

    # Print structure of first sample
    if len(samples) > 0:
        print_sample_structure(samples[0], sample_id=0)
