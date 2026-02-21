# Marginal Envelope Boundary Projection

## Overview

Replace 2D slice-based boundary visualization with **marginal envelope projection** that shows the range of possible boundary positions when marginalizing over the non-displayed dimensions.

## Problem with Current Approach

**Current (2D Slices):**
```
For pair (joint_1, joint_2):
    Fix joint_3 = center, joint_4 = center
    Evaluate h(j1, j2, center, center) on grid
    Plot contour at h = 0
```

**Issues:**
1. Boundary position depends heavily on the fixed values
2. A slice might show "feasible" when most of the 4D space at that (j1, j2) is infeasible
3. Misleading representation of the true 4D geometry

## Proposed Approach: Marginal Envelope

**Concept:** For each (j1, j2) point, sample many (j3, j4) combinations and show the envelope of possible h values.

```
For each (j1, j2) grid point:
    Sample N random (j3, j4) values within bounds
    Compute h_i = h(j1, j2, j3_i, j4_i) for i = 1..N

    Store:
        h_min = min(h_i)  # Most infeasible configuration
        h_max = max(h_i)  # Most feasible configuration
        h_mean = mean(h_i)  # Average feasibility
        p_feasible = fraction where h_i > 0  # Probability of feasibility
```

**Visualization:**
- **Inner boundary** (h_max = 0): Points where NO configuration of (j3, j4) is feasible
- **Outer boundary** (h_min = 0): Points where ALL configurations of (j3, j4) are feasible
- **Band between**: Region where feasibility depends on (j3, j4)

## Visual Design

### Option A: Contour Band
```
- White dashed line: h_max = 0 (inner boundary - definitely infeasible outside)
- White solid line: h_min = 0 (outer boundary - definitely feasible inside)
- Shaded band between: Uncertain region
```

### Option B: Probability Heatmap Overlay
```
- Color intensity: p_feasible (probability of being feasible)
- Single contour at p_feasible = 0.5
```

### Option C: Combined (Recommended)
```
- Background: BALD score heatmap (existing)
- White dashed line: GT inner boundary (h_max = 0)
- White dotted line: GT outer boundary (h_min = 0)
- Red solid line: Pred inner boundary
- Red dashed line: Pred outer boundary
- Optional: Light shading in the uncertain band
```

## Implementation Plan

### Step 1: Create Projection Utility Function

**File:** `active_learning/src/boundary_projection.py`

```python
def compute_marginal_envelope(
    checker,  # LatentFeasibilityChecker or ground truth checker
    grid_j1: torch.Tensor,  # (R,) values for joint 1
    grid_j2: torch.Tensor,  # (R,) values for joint 2
    idx1: int,  # Index of joint 1 in full joint vector
    idx2: int,  # Index of joint 2 in full joint vector
    bounds: torch.Tensor,  # (n_joints, 2) full bounds
    n_samples: int = 50,  # Samples for marginalization
    device: str = 'cuda'
) -> dict:
    """
    Compute marginal envelope of feasibility boundary.

    Returns:
        {
            'h_min': (R, R) - min h over marginal samples (outer envelope)
            'h_max': (R, R) - max h over marginal samples (inner envelope)
            'h_mean': (R, R) - mean h over marginal samples
            'p_feasible': (R, R) - probability of feasibility
        }
    """
```

### Step 2: Efficient Batched Implementation

```python
def compute_marginal_envelope(...):
    R = len(grid_j1)
    n_joints = bounds.shape[0]

    # Create meshgrid for (j1, j2)
    J1, J2 = torch.meshgrid(grid_j1, grid_j2, indexing='ij')  # (R, R)

    # Sample marginal dimensions
    marginal_indices = [i for i in range(n_joints) if i not in (idx1, idx2)]

    # Initialize accumulators
    h_min = torch.full((R, R), float('inf'), device=device)
    h_max = torch.full((R, R), float('-inf'), device=device)
    h_sum = torch.zeros((R, R), device=device)
    feasible_count = torch.zeros((R, R), device=device)

    for _ in range(n_samples):
        # Sample random values for marginal dimensions
        marginal_vals = {}
        for idx in marginal_indices:
            marginal_vals[idx] = bounds[idx, 0] + torch.rand(1, device=device) * (bounds[idx, 1] - bounds[idx, 0])

        # Build full grid points: (R*R, n_joints)
        points = torch.zeros(R * R, n_joints, device=device)
        points[:, idx1] = J1.flatten()
        points[:, idx2] = J2.flatten()
        for idx in marginal_indices:
            points[:, idx] = marginal_vals[idx]

        # Evaluate h
        h = checker.logit_value(points)  # (R*R,)
        h = h.view(R, R)

        # Update accumulators
        h_min = torch.min(h_min, h)
        h_max = torch.max(h_max, h)
        h_sum += h
        feasible_count += (h > 0).float()

    return {
        'h_min': h_min,
        'h_max': h_max,
        'h_mean': h_sum / n_samples,
        'p_feasible': feasible_count / n_samples,
    }
```

### Step 3: Vectorized Version (Faster)

```python
def compute_marginal_envelope_vectorized(...):
    """Fully vectorized version - computes all samples in one batched call."""
    R = len(grid_j1)
    n_joints = bounds.shape[0]

    # Create meshgrid for (j1, j2)
    J1, J2 = torch.meshgrid(grid_j1, grid_j2, indexing='ij')  # (R, R)

    marginal_indices = [i for i in range(n_joints) if i not in (idx1, idx2)]

    # Sample all marginal values at once: (n_samples, len(marginal_indices))
    marginal_samples = torch.zeros(n_samples, len(marginal_indices), device=device)
    for i, idx in enumerate(marginal_indices):
        marginal_samples[:, i] = bounds[idx, 0] + torch.rand(n_samples, device=device) * (bounds[idx, 1] - bounds[idx, 0])

    # Build all points: (n_samples, R, R, n_joints)
    points = torch.zeros(n_samples, R, R, n_joints, device=device)
    points[:, :, :, idx1] = J1.unsqueeze(0)  # Broadcast across samples
    points[:, :, :, idx2] = J2.unsqueeze(0)
    for i, idx in enumerate(marginal_indices):
        points[:, :, :, idx] = marginal_samples[:, i].view(n_samples, 1, 1)

    # Reshape for batched evaluation: (n_samples * R * R, n_joints)
    points_flat = points.view(-1, n_joints)

    # Evaluate h
    h_flat = checker.logit_value(points_flat)  # (n_samples * R * R,)
    h = h_flat.view(n_samples, R, R)  # (n_samples, R, R)

    return {
        'h_min': h.min(dim=0).values,
        'h_max': h.max(dim=0).values,
        'h_mean': h.mean(dim=0),
        'p_feasible': (h > 0).float().mean(dim=0),
    }
```

### Step 4: Update Landscape Computation

**File:** `active_learning/test/diagnostics/run_latent_diagnosis.py`

Modify `compute_landscape_at_iter()`:

```python
def compute_landscape_at_iter(learner, iteration, res=12, selected_test=None,
                               query_history=None, use_marginal_envelope=True,
                               n_marginal_samples=50):
    """
    Compute BALD scores and projected boundary for all 6 pairs.

    If use_marginal_envelope=True, computes envelope projection instead of slice.
    """
    # ... existing setup code ...

    for pair_idx, pair in enumerate(pairs):
        # ... existing grid setup ...

        if use_marginal_envelope:
            # Compute marginal envelope for GT
            gt_envelope = compute_marginal_envelope(
                checker=learner.oracle.ground_truth_checker,
                grid_j1=x, grid_j2=y,
                idx1=idx1, idx2=idx2,
                bounds=bounds,
                n_samples=n_marginal_samples,
                device=DEVICE
            )

            # Compute marginal envelope for prediction
            pred_envelope = compute_marginal_envelope_for_decoder(
                decoder=learner.decoder,
                z=learner.get_posterior().mean,
                grid_j1=x, grid_j2=y,
                idx1=idx1, idx2=idx2,
                bounds=bounds,
                n_samples=n_marginal_samples,
                device=DEVICE
            )

            results[pair_idx] = {
                # ... existing fields ...
                'gt_h_min': gt_envelope['h_min'].cpu().numpy(),
                'gt_h_max': gt_envelope['h_max'].cpu().numpy(),
                'pred_h_min': pred_envelope['h_min'].cpu().numpy(),
                'pred_h_max': pred_envelope['h_max'].cpu().numpy(),
                'use_envelope': True,
            }
        else:
            # Existing slice-based computation
            # ...
```

### Step 5: Update Plotting

**File:** `active_learning/test/diagnostics/run_latent_diagnosis.py`

Modify `plot_master_grid()`:

```python
def plot_boundary_contours(ax, pdata):
    """Plot boundary contours with envelope if available."""

    if pdata.get('use_envelope', False):
        # Ground Truth envelope
        try:
            # Inner boundary (definitely infeasible outside)
            ax.contour(pdata['X'], pdata['Y'], pdata['gt_h_max'],
                      levels=[0], colors='white', linestyles='dashed',
                      linewidths=1.5, label='GT Inner')
            # Outer boundary (definitely feasible inside)
            ax.contour(pdata['X'], pdata['Y'], pdata['gt_h_min'],
                      levels=[0], colors='white', linestyles='dotted',
                      linewidths=1.0, label='GT Outer')
        except:
            pass

        # Prediction envelope
        try:
            ax.contour(pdata['X'], pdata['Y'], pdata['pred_h_max'],
                      levels=[0], colors='red', linestyles='solid',
                      linewidths=1.5, label='Pred Inner')
            ax.contour(pdata['X'], pdata['Y'], pdata['pred_h_min'],
                      levels=[0], colors='red', linestyles='dashed',
                      linewidths=1.0, label='Pred Outer')
        except:
            pass
    else:
        # Existing slice-based contours
        # ... existing code ...
```

### Step 6: Add Configuration

**File:** `active_learning/configs/latent.yaml`

```yaml
visualization:
  cap_joint_evolution_to_anatomical: false

  # Boundary projection settings
  use_marginal_envelope: true  # Use envelope projection instead of slices
  n_marginal_samples: 50  # Samples for marginalization (higher = smoother but slower)
  show_probability_band: false  # Show shaded uncertainty band
```

## File Changes Summary

| File | Action |
|------|--------|
| `active_learning/src/boundary_projection.py` | **CREATE** - Envelope computation utilities |
| `active_learning/test/diagnostics/run_latent_diagnosis.py` | MODIFY - Use envelope in landscape computation |
| `active_learning/configs/latent.yaml` | MODIFY - Add visualization config |

## Performance Considerations

**Current (slice):** O(R²) evaluations per pair
**Envelope:** O(R² × n_samples) evaluations per pair

For R=12, n_samples=50:
- Slice: 144 evaluations
- Envelope: 7,200 evaluations (50x more)

**Mitigations:**
1. Vectorized batched evaluation (already fast on GPU)
2. Configurable n_samples (default 50, can reduce to 20 for speed)
3. Cache envelope computation (doesn't change within iteration)

## Visual Example

```
Current (slice):
┌─────────────────┐
│     ╭───╮       │  Single line - may be misleading
│    ╱     ╲      │
│   │       │     │
│    ╲     ╱      │
│     ╰───╯       │
└─────────────────┘

Envelope (proposed):
┌─────────────────┐
│   ╭─────────╮   │  Outer envelope (h_min=0)
│  ╱  ╭───╮    ╲  │  Inner envelope (h_max=0)
│ │  ╱     ╲    │ │  Band = uncertainty region
│  ╲  ╲   ╱    ╱  │
│   ╰──╰─╯───╯    │
└─────────────────┘
```

## Testing

1. Run diagnostic with envelope enabled:
   ```bash
   python run_latent_diagnosis.py --strategy bald --budget 20
   ```

2. Compare visually with slice-based (set `use_marginal_envelope: false`)

3. Verify envelope bands are sensible:
   - Inner should always be inside outer
   - Band width indicates sensitivity to marginal dimensions
   - Tight band = boundary position stable across (j3, j4)
   - Wide band = boundary highly dependent on (j3, j4)
