"""
Configuration file for SIRS-enhanced joint limit sampling.

Centralizes all parameters for batch sampling with SIRS constraints.
"""

import numpy as np

# ============================================================================
# Original Batch Sampler Parameters
# ============================================================================

# Sampling parameters
N_SAMPLES_PER_BATCH = 5000 # 50000  # Number of samples per batch
COVERAGE_FACTOR = 0.8  # Use central 80% of joint range
MIN_RANGE_FACTOR = 0.2  # Minimum range is 10% of base range
MAIN_DOFS_ONLY = True  # If True, only sample shoulder + elbow (4 DOFs)
SEED = 42  # Random seed for reproducibility

# ============================================================================
# SIRS Parameters
# ============================================================================

# Enable SIRS constraints
ENABLE_SIRS = True

# Target feasibility sampling
TARGET_FEASIBILITY_MIN = 0.5  # Minimum target feasibility (50%)
TARGET_FEASIBILITY_MAX = 0.9  # Maximum target feasibility (90%)

# Bump generation parameters
NUM_BUMPS_MIN = 1  # Minimum number of bumps per pair
NUM_BUMPS_MAX = 3  # Maximum number of bumps per pair

# Edge bias - Beta distribution parameter sampling
# β(α, α): α < 1 = edges, α = 1 = uniform, α > 1 = center
EDGE_BIAS = True  # If False, use uniform placement (ignore beta params)
EDGE_BIAS_BETA_MIN = 0.1   # More edge bias (sample per-sample)
EDGE_BIAS_BETA_MAX = 0.4   # Less edge bias

ENABLE_ROTATION = True  # Enable rotated (non-axis-aligned) Gaussians

# Bump strength (alpha) - log-normal distribution parameters
# Sampled per-sample for diversity in constraint severity
ALPHA_LOGNORMAL_MEAN_MIN = 3.5   # ln(alpha) mean minimum
ALPHA_LOGNORMAL_MEAN_MAX = 4.5   # ln(alpha) mean maximum
ALPHA_LOGNORMAL_SIGMA_MIN = 0.4  # ln(alpha) std minimum
ALPHA_LOGNORMAL_SIGMA_MAX = 0.8  # ln(alpha) std maximum

# Lengthscale parameters - controls bump width
# Sampled per-sample for diversity in constraint spatial extent
LENGTHSCALE_FRACTION_MIN = 0.10      # Minimum: 10% of box width
LENGTHSCALE_FRACTION_MAX = 0.25      # Maximum: 25% of box width
LENGTHSCALE_SIGMA_MIN = 0.3          # Minimum log-normal std
LENGTHSCALE_SIGMA_MAX = 0.5          # Maximum log-normal std
LENGTHSCALE_MIN_ABS_FRACTION = 0.05  # Absolute minimum lengthscale
LENGTHSCALE_MAX_ABS_FRACTION = 0.5   # Absolute maximum lengthscale

# Grid resolution for feasibility evaluation
# Fixed resolution for performance (was 300-400, now fixed at 200 for 2.25x speedup)
GRID_RESOLUTION = 200

# SIRS smoothing
USE_SMOOTH_CORNERS = True  # Use smooth box margins
SMOOTHING_K_AUTO = True  # Auto-compute smoothing parameter

# Calibration parameters
CALIBRATION_GRID_N = 200  # Grid resolution for calibration
CALIBRATION_MAX_ITER = 50  # Maximum calibration iterations
CALIBRATION_TOLERANCE = 0.05  # ±5% tolerance

# Random seed for SIRS (None = use same as batch sampler)
SIRS_SEED = None

# ============================================================================
# Joint Pair Selection
# ============================================================================

# Main 4 pairs for shoulder + elbow coupling
MAIN_JOINT_PAIRS = [
    ('shoulder_flexion_r', 'shoulder_abduction_r'),
    ('shoulder_flexion_r', 'shoulder_rotation_r'),
    ('shoulder_abduction_r', 'shoulder_rotation_r'),
    ('shoulder_abduction_r', 'elbow_flexion_r'),
]

# Joint names (must match batch_joint_limit_sampler.py)
ALL_JOINT_NAMES = [
    'clavicle_protraction_r',
    'clavicle_elevation_r',
    'clavicle_rotation_r',
    'scapula_abduction_r',
    'scapula_elevation_r',
    'scapula_winging_r',
    'shoulder_flexion_r',
    'shoulder_abduction_r',
    'shoulder_rotation_r',
    'elbow_flexion_r',
]

# ============================================================================
# Output Settings
# ============================================================================

# Directory for outputs
OUTPUT_DIR = 'output/sirs_sampling'

# HDF5 settings
HDF5_FILENAME = 'sirs_samples_100.h5'
HDF5_COMPRESSION = 'gzip'  # Compression algorithm
HDF5_COMPRESSION_LEVEL = 4  # Compression level (0-9, higher = more compression)

# Visualization settings
VIZ_GRID_RESOLUTION = 300  # Grid resolution for visualizations
VIZ_DPI = 150  # DPI for saved figures
VIZ_FIGSIZE_SINGLE = (10, 8)  # Figure size for single plots
VIZ_FIGSIZE_GRID = (16, 12)  # Figure size for grid plots

# ============================================================================
# Validation Settings
# ============================================================================

# Connectivity validation
CHECK_CONNECTIVITY = True  # Check for disconnected regions
CONNECTIVITY_GRID_N = 50  # Grid resolution for connectivity check (reduced from 100 for 4x speedup)
MAX_DISCONNECTED_FRACTION = 0.1  # Reject if >10% pairs disconnected

# Rejection sampling - ZERO TOLERANCE for disconnected regions
REJECT_DISCONNECTED = True  # Reject samples with disconnected pairs
MAX_REJECTION_ATTEMPTS = 10  # Max attempts per sample (increased from 10)
FAIL_ON_DISCONNECTED = True  # If True, skip failed samples entirely (don't use last attempt)
                              # If False, use last attempt with warning (old behavior)

# Test sample sizes
N_TEST_SAMPLES = 10  # Number of samples for initial testing
N_FULL_SAMPLES = 100  # Number of samples for full batch

# ============================================================================
# Helper Functions
# ============================================================================

def get_config_dict():
    """Return all configuration as a dictionary."""
    return {
        # Batch sampler
        'n_samples_per_batch': N_SAMPLES_PER_BATCH,
        'coverage_factor': COVERAGE_FACTOR,
        'min_range_factor': MIN_RANGE_FACTOR,
        'main_dofs_only': MAIN_DOFS_ONLY,
        'seed': SEED,

        # SIRS
        'enable_sirs': ENABLE_SIRS,
        'target_feasibility_min': TARGET_FEASIBILITY_MIN,
        'target_feasibility_max': TARGET_FEASIBILITY_MAX,
        'num_bumps_min': NUM_BUMPS_MIN,
        'num_bumps_max': NUM_BUMPS_MAX,
        'edge_bias': EDGE_BIAS,
        'enable_rotation': ENABLE_ROTATION,
        'use_smooth_corners': USE_SMOOTH_CORNERS,
        'smoothing_k_auto': SMOOTHING_K_AUTO,
        'calibration_grid_n': CALIBRATION_GRID_N,
        'sirs_seed': SIRS_SEED,

        # Joint pairs
        'main_joint_pairs': MAIN_JOINT_PAIRS,
        'all_joint_names': ALL_JOINT_NAMES,

        # Output
        'output_dir': OUTPUT_DIR,
        'hdf5_filename': HDF5_FILENAME,
        'hdf5_compression': HDF5_COMPRESSION,
        'hdf5_compression_level': HDF5_COMPRESSION_LEVEL,

        # Validation
        'check_connectivity': CHECK_CONNECTIVITY,
        'connectivity_grid_n': CONNECTIVITY_GRID_N,
        'max_disconnected_fraction': MAX_DISCONNECTED_FRACTION,
        'n_test_samples': N_TEST_SAMPLES,
        'n_full_samples': N_FULL_SAMPLES,
    }


def print_config():
    """Print all configuration parameters."""
    print("=" * 70)
    print("SIRS-Enhanced Joint Limit Sampling Configuration")
    print("=" * 70)

    print("\n[Batch Sampler Parameters]")
    print(f"  Samples per batch: {N_SAMPLES_PER_BATCH}")
    print(f"  Coverage factor: {COVERAGE_FACTOR} (use central {COVERAGE_FACTOR*100:.0f}%)")
    print(f"  Min range factor: {MIN_RANGE_FACTOR} (at least {MIN_RANGE_FACTOR*100:.0f}% ROM)")
    print(f"  Main DOFs only: {MAIN_DOFS_ONLY}")
    print(f"  Random seed: {SEED}")

    print("\n[SIRS Parameters]")
    print(f"  Enable SIRS: {ENABLE_SIRS}")
    print(f"  Target feasibility range: [{TARGET_FEASIBILITY_MIN}, {TARGET_FEASIBILITY_MAX}]")
    print(f"  Number of bumps: [{NUM_BUMPS_MIN}, {NUM_BUMPS_MAX}]")
    print(f"  Edge bias: {EDGE_BIAS}")
    print(f"  Enable rotation: {ENABLE_ROTATION}")
    print(f"  Use smooth corners: {USE_SMOOTH_CORNERS}")
    print(f"  Calibration grid: {CALIBRATION_GRID_N}×{CALIBRATION_GRID_N}")
    print(f"  SIRS seed: {SIRS_SEED or 'same as batch sampler'}")

    print("\n[Joint Pairs]")
    print(f"  Number of pairs: {len(MAIN_JOINT_PAIRS)}")
    for i, (j1, j2) in enumerate(MAIN_JOINT_PAIRS, 1):
        print(f"    {i}. {j1} × {j2}")

    print("\n[Output Settings]")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  HDF5 filename: {HDF5_FILENAME}")
    print(f"  HDF5 compression: {HDF5_COMPRESSION} (level {HDF5_COMPRESSION_LEVEL})")
    print(f"  Visualization DPI: {VIZ_DPI}")

    print("\n[Validation Settings]")
    print(f"  Check connectivity: {CHECK_CONNECTIVITY}")
    print(f"  Max disconnected fraction: {MAX_DISCONNECTED_FRACTION*100:.0f}%")
    print(f"  Test samples: {N_TEST_SAMPLES}")
    print(f"  Full batch samples: {N_FULL_SAMPLES}")

    print("=" * 70)


if __name__ == '__main__':
    # Test configuration loading
    print_config()

    # Verify configuration dictionary
    print("\n[Testing config_dict...]")
    config_dict = get_config_dict()
    print(f"Config dictionary has {len(config_dict)} parameters")
    print("✓ Configuration loaded successfully!")
