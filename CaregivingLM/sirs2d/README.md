# SIRS 2D: Sparse Interaction RBF Shrinkage

A proof-of-concept system for generating configuration-dependent feasible regions by carving Gaussian "dents" inside rectangular joint limit boxes.

## Core Concept

Starting with a rectangular joint limit box `[l1, u1] × [l2, u2]`, we define a smooth feasibility function:

```
h(q) = box_margin(q) - delta(q)
```

where:
- **box_margin(q)**: Distance to nearest box edge
- **delta(q)**: Sum of Gaussian RBF "bumps" that carve dents

A configuration **q** is feasible if **h(q) ≥ 0**.

### Mathematics

Each bump k has:
- Center **μₖ** = [μ₁ₖ, μ₂ₖ]
- Lengthscales **lsₖ** = [ls₁ₖ, ls₂ₖ]
- Strength **αₖ**
- **Rotation angle θₖ** ∈ [0, 2π) (optional, for tilted ellipses)

#### Axis-Aligned (Without Rotation)
```
RBFₖ(q) = exp(-0.5 · ((q₁-μ₁ₖ)²/ls₁ₖ² + (q₂-μ₂ₖ)²/ls₂ₖ²))
```

#### Rotated (With Joint Coupling)
```
Rₖ = [[cos(θₖ), -sin(θₖ)],
      [sin(θₖ),  cos(θₖ)]]

dq_rot = Rₖᵀ · (q - μₖ)
RBFₖ(q) = exp(-0.5 · ((dq_rot₁/ls₁ₖ)² + (dq_rot₂/ls₂ₖ)²))
```

The penalty function:
```
delta(q) = Σₖ αₖ · RBFₖ(q)
```

**Key insight**: Rotation introduces **cross-joint correlation**, producing tilted elliptical dents that capture directional coupling between joints.

## Directory Structure

```
sirs2d/
├── __init__.py           # Package initialization
├── config.py             # Global parameters
├── sirs.py              # Core math (vectorized numpy)
├── sampler.py           # Box and bump sampling
├── visualize.py         # Matplotlib plotting utilities
├── experiment.py        # Main demonstrations
├── tests/
│   └── test_sirs2d.py   # Unit tests
├── outputs/             # Generated visualizations
│   ├── single_user.png
│   ├── calibration_before_after.png
│   ├── montage.png
│   └── edge_coupling.png
└── README.md            # This file
```

## Installation

No special dependencies beyond standard scientific Python:
```bash
pip install numpy matplotlib
```

## Usage

### Run All Demonstrations

```bash
python3 -m sirs2d.experiment
```

Generates five visualizations:
1. **Single-user demo**: One random box with bumps
2. **Calibration demo**: Before/after α scaling to 50% feasibility
3. **Diversity montage**: 9 random configurations
4. **Edge coupling demo**: Manual bump placement near edges/corners
5. **Edge bias comparison**: Uniform vs. edge-biased bump placement

### Run Unit Tests

```bash
python3 -m sirs2d.tests.test_sirs2d
```

Tests validate:
- Points outside box are infeasible
- K=0 bumps → feasible region = box
- Increasing α reduces feasibility
- Calibration reaches target within ±5%

### Programmatic Usage

#### Quick Start (Convenience Wrapper)

```python
import numpy as np
from sirs2d.sampler import generate_sirs_user
from sirs2d.visualize import compose_panel
import matplotlib.pyplot as plt

# Generate complete SIRS configuration with one call
rng = np.random.default_rng(42)
user = generate_sirs_user(rng, target_frac=0.6, edge_bias=True)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
compose_panel(user['box'], user['bumps'], user['X'], user['Y'], user['H'], user['M'], ax=ax)
print(f"Feasible fraction: {user['feasible_fraction']:.2%}")
plt.show()
```

#### Manual Control (Step-by-Step)

```python
import numpy as np
from sirs2d.sampler import sample_box, sample_bumps, calibrate_alpha
from sirs2d.sirs import feasible_mask_grid
from sirs2d.visualize import compose_panel
import matplotlib.pyplot as plt

# Initialize RNG
rng = np.random.default_rng(42)

# Sample box and bumps
box = sample_box(rng)
bumps = sample_bumps(box, rng, edge_bias=True)  # Optional: bias toward edges

# Calibrate to 60% feasibility (deterministic)
bumps = calibrate_alpha(box, bumps, target_frac=0.6)

# Evaluate on grid
X, Y, H, M = feasible_mask_grid(box, bumps, grid_n=400)

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
compose_panel(box, bumps, X, Y, H, M, ax=ax)
plt.show()
```

## Parameters (config.py)

### Global Joint Limits
- `Q1_RANGE = (-150, 150)` degrees
- `Q2_RANGE = (-50, 180)` degrees

### Box Sampling
- Size: 35-80% of global range
- Randomly positioned within global limits

### Bump Parameters
- Number: K ∈ [1, 4] random
- Centers: Uniform within box (or edge-biased via Beta distribution)
- Lengthscales: ~15% of box width, log-normal variation
  - **Clamped to [5%, 50%] of box width** to prevent degenerate bumps
- Strength (α): Log-normal with mean = exp(4.0) ≈ 54.6
- **Rotation angle θ**: Uniform in [0, 2π) (if `ENABLE_ROTATION=True`)
  - Enables tilted elliptical dents for cross-joint correlation

### Edge-Biased Sampling
- Optional `edge_bias=True` flag in `sample_bumps()`
- Uses **Beta(α=0.2, α=0.2)** distribution - U-shaped for edge bias
  - **α < 1**: Pushes toward edges (U-shaped distribution)
  - **α = 1**: Uniform sampling
  - **α > 1**: Pushes toward center (bell-shaped distribution)
- More realistic human-like joint coupling patterns near joint limits

### Calibration
- **Fully deterministic** binary search (same inputs → same outputs)
- ±5% tolerance, max 50 iterations
- Adaptive bound expansion for extreme target fractions

### Visualization
- Grid resolution: 300-400 points
- Colormap: viridis
- Contour: h=0 boundary in black
- Bumps: Red centers + 1σ ellipses

## Design Philosophy

Following **Linus Torvalds' "good taste" principle**:

1. **Vectorization eliminates loops**: All grid operations use numpy broadcasting
2. **No special cases**: Same math works for edge/corner/center bumps
3. **Pure functions**: No global state, explicit inputs
4. **Simple data structures**:
   - Box = `{'q1_range': (min, max), 'q2_range': (min, max)}`
   - Bump = `{'mu': [μ₁, μ₂], 'ls': [ls₁, ls₂], 'alpha': α, 'R': 2x2 matrix, 'theta': θ}`
   - Rotation handled via optional 'R' key (backward compatible)

## Validation Results

All tests pass:
```
Test 1: Outside box → infeasible ✓
Test 2: K=0 → feasible region = box ✓
Test 3: Increasing α → reduced feasibility ✓
Test 4: Calibration accuracy ✓
```

## Output Examples

Generated visualizations show:
- **Heatmap**: Feasible (green) vs infeasible (gray) regions
- **Contour**: h=0 boundary (thick black line)
- **Box outline**: Dashed blue rectangle
- **Bumps**: Red centers with 1σ ellipses (rotated if θ ≠ 0)

See `sirs2d/outputs/` for generated PNG files:
- `single_user.png` - Basic demonstration
- `calibration_before_after.png` - Alpha scaling to target
- `montage.png` - 9 diverse configurations
- `edge_coupling.png` - Manual edge/corner placement
- `edge_bias_comparison.png` - Uniform vs. edge-biased sampling

## Key Findings

1. **K=0**: Feasible region = box (100% feasibility)
2. **Small K, moderate α**: Gentle dents, smooth boundaries
3. **Large α or multiple bumps**: Strong shrinkage, possibly disconnected regions
4. **Edge/corner bumps**: Realistic curved joint-limit coupling
5. **Calibration**: Reliably reaches target feasibility ±5%
6. **Rotated bumps**: Tilted ellipses create directional coupling between joints
   - Larger lengthscale = wider RBF spread = more carving along that axis
   - Models realistic constraints like "elbow extension limited when shoulder abducted"
7. **Edge-biased sampling**: Beta(0.2, 0.2) pushes bumps near boundaries
   - Average distance to edge: ~0.00-0.10 (biased) vs ~0.25 (uniform)
   - More realistic for modeling joint-limit interactions

## Recent Bug Fixes

### Rotation Matrix Bug (Fixed)
**Issue**: The einsum operation `'...i,ij->...j'` was computing `dq @ R.T` instead of `R.T @ dq`, causing rotated ellipses to have their axes swapped (90° rotation error).

**Symptoms**:
- RBF decay was stronger along the long axis instead of the short axis
- Visual ellipses appeared correct but the carving pattern was backwards

**Fix**: Changed einsum to `'ij,...j->...i'` to correctly compute `R.T @ dq`

**Validation**: Points at 1σ distance along rotated principal axes now correctly evaluate to `exp(-0.5) ≈ 0.6065`

### Edge Distance Calculation Bug (Fixed)
**Issue**: `demo_edge_bias()` was calculating distance FROM CENTER instead of distance TO EDGE.

**Symptoms**: Printed metrics showed biased samples were farther from edges than uniform samples (backwards).

**Fix**: Changed from `min(abs(p[0]-0.5), abs(p[1]-0.5))` to `min(p[0], 1-p[0], p[1], 1-p[1])`

**Validation**: Edge-biased samples now correctly show lower avg_edge_dist values.

### Calibration Rotation Loss Bug (Fixed)
**Issue**: `_scale_bumps()` wasn't copying `theta` and `R` keys during alpha scaling, causing rotated bumps to become axis-aligned after calibration.

**Fix**: Added explicit preservation of rotation keys in `_scale_bumps()`.

## License

Part of the CaregivingLM project.
