# SIRS-Enhanced Joint Limit Sampling

Comprehensive system for generating realistic joint limit sets with pairwise coupling constraints using SIRS (Sparse Interaction RBF Shrinkage).

## Overview

This system extends the original coverage-based joint limit sampler by adding pairwise SIRS constraints to model realistic joint coupling. It generates 100 diverse samples with smooth, connected feasibility regions suitable for biomechanical simulation.

**Key Features:**
- ✅ Extends original batch sampler non-invasively
- ✅ Pairwise 2D SIRS constraints for 4 main joint pairs
- ✅ Rejection sampling eliminates disconnected regions
- ✅ Hierarchical HDF5 storage format
- ✅ Query interface for feasibility checking
- ✅ Comprehensive visualizations

## Quick Start

### 1. Generate 100 SIRS-enhanced samples

```bash
python3 generate_full_batch.py
```

This generates:
- 100 samples with box limits + SIRS bumps
- Statistics visualization: `output/sirs_sampling/batch_statistics.png`
- ~2-3 minutes runtime

### 2. Export to HDF5

```bash
python3 export_to_hdf5.py
```

Creates: `output/sirs_sampling/sirs_samples_100.h5` (~6 MB)

### 3. Query feasibility

```python
from feasibility_checker import SIRSFeasibilityChecker

# Load sample 0
checker = SIRSFeasibilityChecker('output/sirs_sampling/sirs_samples_100.h5', sample_id=0)

# Check a joint configuration
joint_config = {
    'shoulder_flexion_r': np.radians(30),
    'shoulder_abduction_r': np.radians(45),
    'shoulder_rotation_r': np.radians(0),
    'elbow_flexion_r': np.radians(90),
    # ... other joints
}

is_feasible = checker.is_feasible(joint_config)
```

## Architecture

### File Structure

```
.
├── sirs_sampling_config.py        # Configuration parameters
├── sirs_batch_sampler.py          # Main sampling logic + rejection sampling
├── validate_sirs_samples.py       # Connectivity validation
├── visualize_sirs_samples.py      # Single/multi-sample visualizations
├── create_pairplots.py            # Seaborn pairplot generation
├── generate_full_batch.py         # Generate 100 samples + statistics
├── export_to_hdf5.py              # HDF5 export + load utilities
├── feasibility_checker.py         # Query interface
├── output/sirs_sampling/          # Generated data + visualizations
│   ├── sirs_samples_100.h5
│   ├── batch_statistics.png
│   ├── pairplot_sample_*.png
│   └── multi_sample_comparison_with_rejection.png
└── sirs2d/                        # SIRS library
    ├── sirs.py                    # Core SIRS functions
    └── sampler.py                 # Bump generation + calibration
```

### Data Flow

```
1. Coverage Sampling (batch_joint_limit_sampler.py)
   └─> Box limits for 10 joints across 100 samples

2. SIRS Enhancement (sirs_batch_sampler.py)
   └─> Add pairwise bumps for 4 main pairs
   └─> Calibrate to target feasibility (0.5-0.9)
   └─> Rejection sampling (eliminate disconnected regions)

3. Storage (export_to_hdf5.py)
   └─> Hierarchical HDF5 format

4. Query (feasibility_checker.py)
   └─> Check joint configurations against SIRS constraints
```

## Configuration

All parameters in `sirs_sampling_config.py`:

### Batch Sampler Parameters

```python
N_SAMPLES_PER_BATCH = 10    # Samples per batch (10 batches = 100 total)
COVERAGE_FACTOR = 0.8       # Use central 80% of joint range
MIN_RANGE_FACTOR = 0.1      # Minimum 10% ROM
SEED = 42                   # Reproducibility
```

### SIRS Parameters

```python
ENABLE_SIRS = True
TARGET_FEASIBILITY_MIN = 0.5    # 50% minimum feasibility
TARGET_FEASIBILITY_MAX = 0.9    # 90% maximum feasibility
NUM_BUMPS_MIN = 1               # 1-4 bumps per pair
NUM_BUMPS_MAX = 4
EDGE_BIAS = True                # Bias bumps toward edges
ENABLE_ROTATION = True          # Enable rotated Gaussians
USE_SMOOTH_CORNERS = True       # Smooth box margins (C² continuity)
```

### Rejection Sampling

```python
REJECT_DISCONNECTED = True
MAX_REJECTION_ATTEMPTS = 10     # Max attempts per sample
```

**Current Performance:**
- Acceptance rate: ~41.5%
- Disconnection rate: <3% (down from 22.5% without rejection)

### Joint Pairs

4 main pairs for shoulder + elbow coupling:

```python
MAIN_JOINT_PAIRS = [
    ('shoulder_flexion_r', 'shoulder_abduction_r'),
    ('shoulder_flexion_r', 'shoulder_rotation_r'),
    ('shoulder_abduction_r', 'shoulder_rotation_r'),
    ('shoulder_abduction_r', 'elbow_flexion_r'),
]
```

## HDF5 Format

### Structure

```
/metadata/
  - n_samples (int): 100
  - joint_names (str array): 10 joint names
  - main_pairs (str array): 4 pairs as "joint1|joint2"
  - config (JSON): Full configuration

/samples/sample_000/
  /box_limits/
    - joint_names (str array)
    - lower_bounds (float array, radians)
    - upper_bounds (float array, radians)

  /sirs_bumps/pair_0/
    - joint_names (str array, size 2)
    - n_bumps (int)
    - mu (float array, shape (n_bumps, 2))      # Centers
    - ls (float array, shape (n_bumps, 2))      # Length scales
    - alpha (float array, shape (n_bumps,))     # Strengths
    - theta (float array, optional)             # Rotation angles

  /metadata/
    - sample_id (str)
    - pair_names (str array)
    - target_feasibility (float array)
    - actual_feasibility (float array)
```

### Loading Samples

```python
from export_to_hdf5 import load_sample_from_hdf5

sample = load_sample_from_hdf5('sirs_samples_100.h5', sample_id=0)

# Access data
print(sample['joint_limits'])        # Dict: {joint_name: (lower, upper)}
print(sample['sirs_bumps'])          # Dict: {(j1, j2): [bump1, bump2, ...]}
print(sample['sirs_metadata'])       # Dict: {(j1, j2): metadata}
```

## Feasibility Checker API

### Basic Usage

```python
from feasibility_checker import SIRSFeasibilityChecker

checker = SIRSFeasibilityChecker('sirs_samples_100.h5', sample_id=0)

# Simple check
is_feasible = checker.is_feasible(joint_config)

# Detailed check
result = checker.is_feasible(joint_config, return_details=True)
print(result['is_feasible'])         # Boolean
print(result['pair_results'])        # Per-pair h-values
print(result['limiting_pair'])       # Most violated constraint
print(result['min_h_value'])         # Minimum h-value across pairs
```

### Sample Feasible Configurations

```python
# Sample a random feasible configuration
config = checker.sample_feasible_config(max_attempts=10000)

if config:
    print("Found feasible configuration:")
    for joint, value in config.items():
        print(f"  {joint}: {np.degrees(value):.1f}°")
```

### Query Constraint Info

```python
# Get box limits
limits = checker.get_box_limits()  # Dict: {joint_name: (lower, upper)}

# Get pairwise constraint info
pairs = checker.get_pairwise_info()

for (j1, j2), info in pairs.items():
    print(f"{j1} × {j2}:")
    print(f"  Bumps: {info['n_bumps']}")
    print(f"  Target feasibility: {info['target_feasibility']:.1%}")
```

## Visualizations

### 1. Single Sample Pairwise Regions

```bash
python3 -c "
from visualize_sirs_samples import visualize_single_sample_pairwise
from export_to_hdf5 import load_sample_from_hdf5

sample = load_sample_from_hdf5('output/sirs_sampling/sirs_samples_100.h5', 0)
visualize_single_sample_pairwise(sample, 'output/sample_0.png')
"
```

Shows 2×2 grid with:
- Feasible region (color = h-value)
- Box outline
- Bump centers + 1σ ellipses
- h=0 contour

### 2. Multi-Sample Comparison

```bash
python3 visualize_sirs_samples.py
```

Montage showing one pair across 10 samples, demonstrating diversity.

### 3. Seaborn Pairplots

```bash
python3 create_pairplots.py
```

Generates 5 pairplots showing:
- Full 4×4 grid (all 4 main joints)
- Histograms on diagonal
- Scatter plots of feasible points
- Clear visualization of SIRS-carved regions

### 4. Batch Statistics

```bash
python3 generate_full_batch.py
```

Dashboard with:
- Target feasibility distributions
- Calibration accuracy (target vs actual)
- Bump count distributions

## Integration with SCONE

The feasibility checker can be integrated with SCONE for biomechanical simulation:

```python
from feasibility_checker import SIRSFeasibilityChecker
# from scone import ForwardKinematics  # Your SCONE interface

# Load a sample
checker = SIRSFeasibilityChecker('sirs_samples_100.h5', sample_id=0)

# Sample a feasible joint configuration
joint_config = checker.sample_feasible_config()

if joint_config:
    # Use with SCONE to compute forward kinematics
    # skeleton_pose = ForwardKinematics(joint_config)
    # ... run simulation
    pass
```

## Performance

### Generation

- **100 samples**: ~2-3 minutes
- **Rejection sampling**: 41.5% acceptance rate
- **Calibration**: Converges in 10-20 iterations
- **Connectivity check**: ~0.1s per pair

### Accuracy

- **Calibration error**: Mean 3-4% ± 2-4%
- **Target tolerance**: ±5%
- **Disconnection rate**: <3%

### File Sizes

- **HDF5 (100 samples)**: ~6 MB (gzip compression level 4)
- **Visualizations**:
  - Batch statistics: 296 KB
  - Pairplot: ~800 KB
  - Multi-sample: 636 KB

## Troubleshooting

### Low Acceptance Rate

If rejection sampling acceptance rate is too low (<30%):

1. **Increase max attempts**: `MAX_REJECTION_ATTEMPTS = 20`
2. **Relax feasibility targets**:
   ```python
   TARGET_FEASIBILITY_MIN = 0.6  # Was 0.5
   TARGET_FEASIBILITY_MAX = 0.8  # Was 0.9
   ```
3. **Reduce bump counts**:
   ```python
   NUM_BUMPS_MIN = 1
   NUM_BUMPS_MAX = 3  # Was 4
   ```

### Disconnected Regions Persist

If disconnection rate stays high (>5%):

1. **Enable smooth corners**: `USE_SMOOTH_CORNERS = True`
2. **Increase smoothing**: Edit `compute_auto_smoothing_k()` default
3. **Check connectivity grid resolution**: `CONNECTIVITY_GRID_N = 150`

### HDF5 File Too Large

To reduce file size:

1. **Increase compression**:
   ```python
   HDF5_COMPRESSION_LEVEL = 9  # Was 4
   ```
2. **Store fewer samples**:
   ```python
   N_FULL_SAMPLES = 50  # Was 100
   ```

## Advanced Usage

### Custom Joint Pairs

To add more joint pairs:

```python
# In sirs_sampling_config.py
MAIN_JOINT_PAIRS = [
    ('shoulder_flexion_r', 'shoulder_abduction_r'),
    ('shoulder_flexion_r', 'shoulder_rotation_r'),
    ('shoulder_abduction_r', 'shoulder_rotation_r'),
    ('shoulder_abduction_r', 'elbow_flexion_r'),
    # Add custom pairs
    ('elbow_flexion_r', 'shoulder_rotation_r'),
    ('shoulder_flexion_r', 'elbow_flexion_r'),
]
```

### Modify Bump Generation

Edit `sirs2d/sampler.py`:

```python
def sample_bumps(box, rng, edge_bias=True, n_bumps_range=(1, 4)):
    """
    Customize bump generation logic.
    - Change n_bumps_range for more/fewer bumps
    - Modify edge_bias sampling distribution
    - Adjust length scale (ls) sampling
    """
```

### Custom Feasibility Targets

Per-pair target feasibility:

```python
# In sirs_batch_sampler.py
target_feas_map = {
    ('shoulder_flexion_r', 'shoulder_abduction_r'): 0.7,
    ('shoulder_abduction_r', 'elbow_flexion_r'): 0.85,
    # ...
}

target_feas = target_feas_map.get(pair_key, rng.uniform(0.5, 0.9))
```

## References

### Theory

- **SIRS**: Sparse Interaction RBF Shrinkage for modeling feasibility regions
- **Smooth Corners**: Cubic polynomial smoothing (Inigo Quilez method)
- **Calibration**: Binary search on bump strength (alpha) to achieve target feasibility

### Related Files

- `sirs2d/README.md`: SIRS theory and implementation
- `sirs2d/SMOOTH_CORNERS.md`: Smooth corner approximation details
- `SIRS_INTEGRATION_PLAN.md`: Original implementation plan
- `SIRS_INTEGRATION_PROGRESS.md`: Phase-by-phase progress report

## Citation

If using this code, please cite:

```
SIRS-Enhanced Joint Limit Sampling for Biomechanical Simulation
[Your details here]
```

## License

[Your license here]

## Contact

For questions or issues:
- See inline documentation in Python files
- Check `SIRS_INTEGRATION_PROGRESS.md` for implementation notes
- Refer to `sirs2d/README.md` for SIRS theory
