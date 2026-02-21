# Phase 3: Vectorize Projected SVGD

## Overview

**Goal**: Eliminate Python loops in `ProjectedSVGD.step()` by processing all slice directions simultaneously.

**Current bottleneck**: The `step()` method has two Python loops:
- Phase 3 (lines 138-154): Loop over `r` eigenvector directions
- Phase 4 (lines 157-173): Loop over `n_random` random directions

Each iteration calls `_compute_1d_svgd()`, which launches multiple GPU kernels. With `n_slices=20` (default), this causes ~100+ kernel launches per SVGD step.

**Expected improvement**: 3-5x speedup on GPU by reducing kernel launch overhead.

---

## High-Level Changes

### 1. Create `_compute_1d_svgd_batched()`
Process all S slices in a single batched operation instead of S separate calls.

**Why**: The current `_compute_1d_svgd()` is called once per slice. Each call does:
- Pairwise difference computation
- Distance squared computation
- Median heuristic (if no fixed width)
- Kernel evaluation
- Term1 and term2 computation

All of these can be batched across slices.

### 2. Build Combined Direction Matrix
Stack eigenvector directions and random directions into a single `(D, S)` matrix.

**Why**: Allows single matrix multiplication for projection (`particles @ directions`) instead of S separate dot products.

### 3. Build Weight Vector
Create a weight vector of length S where:
- First `r` entries: eigenvalue weights (for eigenvector directions)
- Remaining `n_random` entries: 1.0 (for random directions)

**Why**: Enables element-wise weighting after computing all phi_1d values.

### 4. Single Matrix Multiply for Reconstruction
Use `phi_1d @ directions.T` to project back to D-space.

**Why**: Replaces S calls to `torch.outer()` with one matrix multiply.

---

## Tensor Shape Reference

| Tensor | Current Shape | Batched Shape | Notes |
|--------|---------------|---------------|-------|
| `particles` | (K, D) | (K, D) | Unchanged |
| `log_prob_grad` | (K, D) | (K, D) | Unchanged |
| `directions` | (D,) per slice | (D, S) | All directions stacked |
| `s_proj` | (K,) per slice | (K, S) | Projected particles |
| `g_proj` | (K,) per slice | (K, S) | Projected gradients |
| `diff_1d` | (K, K) per slice | (K, K, S) | Pairwise differences |
| `dist_sq_1d` | (K, K) per slice | (K, K, S) | Squared distances |
| `h_val` | scalar per slice | (S,) | Bandwidth per slice |
| `k_mat` | (K, K) per slice | (K, K, S) | Kernel matrices |
| `term1_1d` | (K,) per slice | (K, S) | Gradient term |
| `term2_1d` | (K,) per slice | (K, S) | Repulsion term |
| `phi_1d` | (K,) per slice | (K, S) | Weighted update per slice |
| `weights` | scalar per slice | (S,) | Eigenvalue or uniform weight |

Where: K = num_particles, D = dimension (32), S = n_slices (20)

---

## Low-Level Implementation Steps

### Step 1: Add Toggle and Batched Method Signature

Add module-level toggle at top of file:
```python
_USE_VECTORIZED_PSVGD = True
```

Add new method signature:
```python
def _compute_1d_svgd_batched(self, s_proj, g_proj, K, h, device):
    """
    Compute 1D SVGD for multiple slices simultaneously.

    Args:
        s_proj: (K, S) projected particles for all slices
        g_proj: (K, S) projected gradients for all slices
        K: number of particles
        h: fixed bandwidth squared, or None for median heuristic
        device: torch device

    Returns:
        term1_1d: (K, S) gradient terms
        term2_1d: (K, S) repulsion terms
        h_vals: (S,) bandwidth values used
    """
```

### Step 2: Implement Batched Pairwise Computation

```python
# s_proj: (K, S)
# Compute pairwise differences for all slices at once
diff_1d = s_proj.unsqueeze(1) - s_proj.unsqueeze(0)  # (K, K, S)
dist_sq_1d = diff_1d ** 2  # (K, K, S)
```

### Step 3: Implement Batched Median Heuristic

The median must be computed per-slice:
```python
S = s_proj.shape[1]

if h is None:
    # Reshape to (S, K*K) to compute median per slice
    dist_sq_flat = dist_sq_1d.permute(2, 0, 1).reshape(S, -1)  # (S, K*K)
    median_dist = torch.median(dist_sq_flat, dim=1).values  # (S,)

    log_K_plus_1 = torch.log(torch.tensor(K + 1.0, device=device))
    h_vals = median_dist / log_K_plus_1  # (S,)

    # Handle zero median (degenerate case)
    h_vals = torch.where(h_vals == 0, torch.ones_like(h_vals), h_vals)
else:
    h_vals = torch.full((S,), h, device=device)
```

### Step 4: Implement Batched Kernel Computation

```python
# h_vals: (S,) -> need to broadcast to (K, K, S)
h_broadcast = h_vals.view(1, 1, S)  # (1, 1, S)
scaled_dist = dist_sq_1d / h_broadcast  # (K, K, S)

if self.kernel_type == 'imq':
    base = 1.0 + scaled_dist
    k_mat = base.pow(-0.5)  # (K, K, S)
    grad_factor = -(1.0 / h_broadcast) * base.pow(-1.5)  # (K, K, S)
else:  # rbf
    k_mat = torch.exp(-scaled_dist)  # (K, K, S)
    grad_factor = -(2.0 / h_broadcast) * k_mat  # (K, K, S)
```

### Step 5: Implement Batched Term Computation

```python
# term1: k_mat.T @ g_proj for each slice
# k_mat: (K, K, S) indexed as (i, j, s), g_proj: (K, S)
# Need: for each slice s, sum_j k_mat[j,i,s] * g_proj[j,s] (the transpose)
# Use einsum: 'jik,jk->ik' - reading first dim of k_mat as 'j' gives us the transpose
term1_1d = torch.einsum('jik,jk->ik', k_mat, g_proj) / K  # (K, S)

# term2: sum over dim 0 of (grad_factor * diff_1d)
# grad_factor: (K, K, S), diff_1d: (K, K, S)
term2_1d = (grad_factor * diff_1d).sum(dim=0) / K  # (K, S)
```

**Note on einsum**: The pattern `'jik,jk->ik'` treats the first dimension of k_mat as `j` (summed over), second as `i` (output), third as `k` (output). This effectively computes the transpose-multiply for each slice. Since RBF/IMQ kernels are symmetric, `'ijk,jk->ik'` would also work, but `'jik'` is semantically correct.

### Step 6: Modify `step()` to Build Direction Matrix

Replace Phase 3 and Phase 4 loops with:

```python
# --- Build combined direction matrix ---
S = self.n_slices
directions_list = []

# Eigenvector directions (first r columns)
if r > 0:
    eigen_dirs = eigenvectors[:, :r]  # (D, r)
    directions_list.append(eigen_dirs)

# Random directions (remaining n_random columns)
if n_random > 0:
    random_dirs = torch.randn(D, n_random, device=device)
    # Normalize each column to unit length
    random_dirs = random_dirs / random_dirs.norm(dim=0, keepdim=True)
    directions_list.append(random_dirs)

# Stack into (D, S) matrix
all_directions = torch.cat(directions_list, dim=1)  # (D, S)
```

### Step 7: Modify `step()` to Build Weight Vector

```python
# --- Build weight vector ---
weights_list = []

if r > 0:
    weights_list.append(eigenvalue_weights)  # (r,)

if n_random > 0:
    weights_list.append(torch.ones(n_random, device=device))  # (n_random,)

all_weights = torch.cat(weights_list)  # (S,)
```

### Step 8: Modify `step()` for Batched Projection

```python
# --- Project particles and gradients ---
s_proj = particles @ all_directions  # (K, D) @ (D, S) -> (K, S)
g_proj = log_prob_grad @ all_directions  # (K, S)
```

### Step 9: Modify `step()` for Batched SVGD Computation

```python
# --- Compute batched 1D SVGD ---
term1_1d, term2_1d, h_vals = self._compute_1d_svgd_batched(
    s_proj, g_proj, K, h_fixed, device
)

# Apply weights
phi_1d = (term1_1d + term2_1d) * all_weights.unsqueeze(0)  # (K, S)
```

### Step 10: Modify `step()` for Batched Reconstruction

```python
# --- Reconstruct in D-space ---
# phi_1d: (K, S), all_directions: (D, S)
# Want: sum over slices of outer(phi_1d[:,s], directions[:,s])
# Equivalent to: phi_1d @ all_directions.T
phi = phi_1d @ all_directions.T  # (K, S) @ (S, D) -> (K, D)

# Normalize
phi /= S
```

### Step 11: Update Diagnostics Collection

For diagnostics, compute aggregate statistics from batched results:

```python
if return_diagnostics:
    h_values_list = h_vals.tolist()
    term1_1d_norms = term1_1d.norm(dim=0).tolist()  # norm per slice
    term2_1d_norms = term2_1d.norm(dim=0).tolist()

    # Rest of diagnostics unchanged...
```

### Step 12: Add Dispatch Logic

In `step()`, add dispatch based on toggle:

```python
if _USE_VECTORIZED_PSVGD:
    return self._step_vectorized(particles, log_prob_grad, return_diagnostics)
else:
    return self._step_loop(particles, log_prob_grad, return_diagnostics)
```

Or integrate directly with fallback flag check.

---

## Edge Cases to Handle

1. **r = 0**: No eigenvector directions (degenerate score covariance)
   - `directions_list` only contains random directions
   - `eigenvalue_weights` is empty, so `weights_list` only contains ones

2. **r = n_slices**: All eigenvector directions, no random
   - `random_dirs` is empty (n_random = 0)
   - Only eigenvector directions in matrix

3. **K = 1**: Single particle
   - Pairwise distances are all zero
   - Median is zero, h_vals fallback to 1.0
   - term2 (repulsion) is zero

4. **Fixed kernel width**: Skip median computation
   - `h_vals = torch.full((S,), h_fixed, device=device)`

---

## Testing Plan

### Test 1: Numerical Equivalence
```python
def test_equivalence():
    """Vectorized produces same results as loop version."""
    # Create test inputs
    K, D = 50, 32
    particles = torch.randn(K, D, device='cuda')
    log_prob_grad = torch.randn(K, D, device='cuda')

    optimizer = ProjectedSVGD(n_slices=20)

    # Seed RNG for reproducible random directions
    torch.manual_seed(42)
    _USE_VECTORIZED_PSVGD = False
    phi_loop = optimizer.step(particles, log_prob_grad)

    torch.manual_seed(42)
    _USE_VECTORIZED_PSVGD = True
    phi_vec = optimizer.step(particles, log_prob_grad)

    max_diff = (phi_loop - phi_vec).abs().max()
    assert max_diff < 1e-5, f"Max diff: {max_diff}"
```

### Test 2: Gradient Equivalence
Verify gradients through particles and log_prob_grad match.

### Test 3: Edge Cases
- r=0 (no eigenvector directions)
- r=n_slices (all eigenvector directions)
- K=1 (single particle)
- Fixed kernel width

### Test 4: Benchmark
Compare timing of loop vs vectorized on GPU.

---

## Potential Issues and Mitigations

### Issue 1: Memory Usage
**Problem**: Batched version creates (K, K, S) tensors instead of (K, K).
**Impact**: With K=50, S=20, this is 50*50*20 = 50K floats = 200KB (negligible).
**Mitigation**: None needed for typical sizes. For very large K, could process in chunks.

### Issue 2: Random Direction Reproducibility
**Problem**: Random directions generated differently (all at once vs one-by-one).
**Impact**: Results differ between loop and vectorized even with same seed.
**Solution**: Generate random directions one-by-one in a loop, then stack. This ensures identical RNG consumption order on both CPU and CUDA:
```python
if n_random > 0:
    random_dirs_list = []
    for _ in range(n_random):
        r_dir = torch.randn(D, device=device)
        r_dir = r_dir / r_dir.norm()
        random_dirs_list.append(r_dir)
    random_dirs = torch.stack(random_dirs_list, dim=1)  # (D, n_random)
```
The overhead of this small loop is negligible compared to the batched SVGD computation gains.

### Issue 3: Median Computation
**Problem**: `torch.median` on large tensors can be slow.
**Impact**: Minimal - we're computing median of 2500 values (K*K) per slice.
**Mitigation**: None needed. If it becomes bottleneck, could use approximate median.

---

## Files to Modify

1. **`active_learning/src/svgd/projected_svgd_optimizer.py`**
   - Add `_USE_VECTORIZED_PSVGD` toggle
   - Add `_compute_1d_svgd_batched()` method
   - Refactor `step()` to use batched computation

2. **`infer_params/tests/test_vectorized_projected_svgd.py`** (new)
   - Numerical equivalence tests
   - Gradient tests
   - Edge case tests
   - Benchmark

---

## Success Criteria

1. All tests pass with max difference < 1e-5
2. Gradients match within 1e-4
3. GPU speedup of 2x+ on `step()` method
4. No increase in memory usage beyond expected (K, K, S) tensor
