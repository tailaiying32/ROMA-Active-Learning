# Phase 2: Vectorize Level Set Evaluation

## Objective
Eliminate the Python for loop over 6 joint pairs in `evaluate_level_set_batched` to reduce GPU kernel launches from ~6 per call to ~1.

---

## Current Implementation Analysis

### File: `infer_params/training/level_set_torch.py`

### Current Code (lines 140-239)

```python
def evaluate_level_set_batched(
    points: torch.Tensor,      # (N, 4) - grid points
    lower: torch.Tensor,       # (B, 4) - box lower bounds
    upper: torch.Tensor,       # (B, 4) - box upper bounds
    weights: torch.Tensor,     # (B, 4) - box weights
    presence: torch.Tensor,    # (B, K) where K=18 - blob presence
    blob_params: torch.Tensor, # (B, K, 6) - blob parameters
    alpha: float = 10.0,
) -> torch.Tensor:             # Returns (B, N)
```

**The loop (lines 195-234):**
```python
joint_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]  # 6 pairs
num_slots = 18  # presence.shape[1]
slots_per_pair = 3  # 18 / 6

penalty = torch.zeros(B, N, device=device)

for pair_idx, (i, j) in enumerate(joint_pairs):
    slot_start = pair_idx * slots_per_pair  # 0, 3, 6, 9, 12, 15
    slot_end = slot_start + slots_per_pair  # 3, 6, 9, 12, 15, 18

    # Extract 2D points for this joint pair: (B, N, 2)
    x_2d = torch.stack([points_batched[:, :, i], points_batched[:, :, j]], dim=-1)

    # Get params for this pair's slots: (B, 3, ...)
    pair_centers = centers[:, slot_start:slot_end, :]   # (B, 3, 2)
    pair_sigmas = sigmas[:, slot_start:slot_end, :]     # (B, 3, 2)
    pair_amps = amplitudes[:, slot_start:slot_end]      # (B, 3)
    pair_rots = rotations[:, slot_start:slot_end]       # (B, 3)
    pair_pres = presence[:, slot_start:slot_end]        # (B, 3)

    # Compute diff: (B, N, 1, 2) - (B, 1, 3, 2) -> (B, N, 3, 2)
    diff = x_2d.unsqueeze(2) - pair_centers.unsqueeze(1)

    # Rotation
    cos_t = torch.cos(-pair_rots).unsqueeze(1)  # (B, 1, 3)
    sin_t = torch.sin(-pair_rots).unsqueeze(1)
    diff_rot_x = cos_t * diff[..., 0] - sin_t * diff[..., 1]
    diff_rot_y = sin_t * diff[..., 0] + cos_t * diff[..., 1]

    # Mahalanobis distance
    pair_sigmas_exp = pair_sigmas.unsqueeze(1)  # (B, 1, 3, 2)
    mahal_sq = (diff_rot_x**2 / pair_sigmas_exp[..., 0] +
                diff_rot_y**2 / pair_sigmas_exp[..., 1])

    # Gaussian blob values
    blob_vals = pair_amps.unsqueeze(1) * torch.exp(-0.5 * mahal_sq)  # (B, N, 3)

    # Mask by presence and sum over slots
    pair_penalty = (blob_vals * pair_pres.unsqueeze(1)).sum(dim=-1)  # (B, N)
    penalty = penalty + pair_penalty
```

---

## Vectorization Strategy

### Key Insight
Each joint pair (i, j) extracts 2 columns from the 4D points. Instead of looping, we can:
1. Pre-compute which columns to extract for all 6 pairs
2. Use advanced indexing to extract all at once
3. Reshape blob params from (B, 18, ...) to (B, 6, 3, ...)
4. Compute all pair penalties in parallel

### Tensor Shape Transformations

| Tensor | Current Shape | Vectorized Shape | Notes |
|--------|---------------|------------------|-------|
| `points_batched` | (B, N, 4) | (B, N, 4) | Unchanged |
| `x_2d` | (B, N, 2) per pair | (B, N, 6, 2) all pairs | Stack all 6 extractions |
| `centers` | (B, 18, 2) | (B, 6, 3, 2) | Reshape by pairs |
| `sigmas` | (B, 18, 2) | (B, 6, 3, 2) | Reshape by pairs |
| `amplitudes` | (B, 18) | (B, 6, 3) | Reshape by pairs |
| `rotations` | (B, 18) | (B, 6, 3) | Reshape by pairs |
| `presence` | (B, 18) | (B, 6, 3) | Reshape by pairs |
| `diff` | (B, N, 3, 2) per pair | (B, N, 6, 3, 2) all pairs | Full tensor |
| `blob_vals` | (B, N, 3) per pair | (B, N, 6, 3) all pairs | Full tensor |
| `penalty` | (B, N) | (B, N) | Sum over dims 2,3 |

### Implementation Steps

#### Step 1: Define pair indices as constants
```python
# At module level for efficiency
_PAIR_I = None  # Will be tensor([0, 0, 0, 1, 1, 2])
_PAIR_J = None  # Will be tensor([1, 2, 3, 2, 3, 3])

def _get_pair_indices(device):
    global _PAIR_I, _PAIR_J
    if _PAIR_I is None or _PAIR_I.device != device:
        _PAIR_I = torch.tensor([0, 0, 0, 1, 1, 2], device=device)
        _PAIR_J = torch.tensor([1, 2, 3, 2, 3, 3], device=device)
    return _PAIR_I, _PAIR_J
```

#### Step 2: Extract all 2D points at once
```python
pair_i, pair_j = _get_pair_indices(device)

# points_batched: (B, N, 4)
# points_batched[:, :, pair_i] -> (B, N, 6) - first coord of each pair
# points_batched[:, :, pair_j] -> (B, N, 6) - second coord of each pair
x_2d_all = torch.stack([
    points_batched[:, :, pair_i],  # (B, N, 6)
    points_batched[:, :, pair_j],  # (B, N, 6)
], dim=-1)  # (B, N, 6, 2)
```

#### Step 3: Reshape blob params by pairs
```python
num_pairs = 6
slots_per_pair = 3

# (B, 18, 2) -> (B, 6, 3, 2)
centers_by_pair = centers.view(B, num_pairs, slots_per_pair, 2)
sigmas_by_pair = sigmas.view(B, num_pairs, slots_per_pair, 2)

# (B, 18) -> (B, 6, 3)
amps_by_pair = amplitudes.view(B, num_pairs, slots_per_pair)
rots_by_pair = rotations.view(B, num_pairs, slots_per_pair)
pres_by_pair = presence.view(B, num_pairs, slots_per_pair)
```

#### Step 4: Compute diff for all pairs
```python
# x_2d_all: (B, N, 6, 2)
# centers_by_pair: (B, 6, 3, 2)
# We need: (B, N, 6, 3, 2)

# Expand dims for broadcasting:
# x_2d_all: (B, N, 6, 1, 2)
# centers:  (B, 1, 6, 3, 2)
diff = x_2d_all.unsqueeze(3) - centers_by_pair.unsqueeze(1)  # (B, N, 6, 3, 2)
```

#### Step 5: Apply rotation (vectorized)
```python
# rots_by_pair: (B, 6, 3)
# Need: (B, 1, 6, 3) for broadcasting with diff (B, N, 6, 3, 2)
cos_t = torch.cos(-rots_by_pair).unsqueeze(1)  # (B, 1, 6, 3)
sin_t = torch.sin(-rots_by_pair).unsqueeze(1)  # (B, 1, 6, 3)

# diff[..., 0] and diff[..., 1] are (B, N, 6, 3)
diff_rot_x = cos_t * diff[..., 0] - sin_t * diff[..., 1]  # (B, N, 6, 3)
diff_rot_y = sin_t * diff[..., 0] + cos_t * diff[..., 1]  # (B, N, 6, 3)
```

#### Step 6: Compute Mahalanobis distance
```python
# sigmas_by_pair: (B, 6, 3, 2) -> (B, 1, 6, 3, 2)
sigmas_exp = sigmas_by_pair.unsqueeze(1)

# (B, N, 6, 3) / (B, 1, 6, 3) -> (B, N, 6, 3)
mahal_sq = (diff_rot_x**2 / sigmas_exp[..., 0] +
            diff_rot_y**2 / sigmas_exp[..., 1])
```

#### Step 7: Compute blob values and penalty
```python
# amps_by_pair: (B, 6, 3) -> (B, 1, 6, 3)
amps_exp = amps_by_pair.unsqueeze(1)

# blob_vals: (B, N, 6, 3)
blob_vals = amps_exp * torch.exp(-0.5 * mahal_sq)

# pres_by_pair: (B, 6, 3) -> (B, 1, 6, 3)
pres_exp = pres_by_pair.unsqueeze(1)

# Mask and sum over slots (dim=3) and pairs (dim=2)
penalty = (blob_vals * pres_exp).sum(dim=(2, 3))  # (B, N)
```

---

## Complete Vectorized Function

```python
def evaluate_level_set_batched_vectorized(
    points: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weights: torch.Tensor,
    presence: torch.Tensor,
    blob_params: torch.Tensor,
    alpha: float = 10.0,
) -> torch.Tensor:
    """Fully vectorized level-set evaluation.

    Eliminates Python loop over joint pairs for GPU efficiency.
    Same interface and output as evaluate_level_set_batched.
    """
    B = lower.shape[0]
    N = points.shape[0]
    device = lower.device

    # Constants
    NUM_PAIRS = 6
    SLOTS_PER_PAIR = presence.shape[1] // NUM_PAIRS  # 3

    # Pair indices (cached)
    pair_i, pair_j = _get_pair_indices(device)

    # Expand points: (N, 4) -> (B, N, 4)
    points_batched = points.unsqueeze(0).expand(B, -1, -1)

    # === BOX DISTANCE (unchanged) ===
    d_box = weighted_box_distance(points_batched, lower, upper, weights, alpha)

    # === VECTORIZED BLOB PENALTY ===

    # Extract blob parameters
    centers = blob_params[:, :, 0:2]  # (B, 18, 2)
    sigmas = blob_params[:, :, 2:4].abs().clamp(min=1e-6)  # (B, 18, 2)
    amplitudes = blob_params[:, :, 4].abs().clamp(min=0.01, max=0.5)  # (B, 18)
    rotations = blob_params[:, :, 5]  # (B, 18)

    # Reshape by pairs: (B, 18, ...) -> (B, 6, 3, ...)
    centers = centers.view(B, NUM_PAIRS, SLOTS_PER_PAIR, 2)
    sigmas = sigmas.view(B, NUM_PAIRS, SLOTS_PER_PAIR, 2)
    amplitudes = amplitudes.view(B, NUM_PAIRS, SLOTS_PER_PAIR)
    rotations = rotations.view(B, NUM_PAIRS, SLOTS_PER_PAIR)
    presence = presence.view(B, NUM_PAIRS, SLOTS_PER_PAIR)

    # Extract 2D points for all pairs: (B, N, 6, 2)
    x_2d = torch.stack([
        points_batched[:, :, pair_i],
        points_batched[:, :, pair_j],
    ], dim=-1)

    # Compute diff: (B, N, 6, 1, 2) - (B, 1, 6, 3, 2) -> (B, N, 6, 3, 2)
    diff = x_2d.unsqueeze(3) - centers.unsqueeze(1)

    # Rotation: (B, 1, 6, 3)
    cos_t = torch.cos(-rotations).unsqueeze(1)
    sin_t = torch.sin(-rotations).unsqueeze(1)

    diff_rot_x = cos_t * diff[..., 0] - sin_t * diff[..., 1]
    diff_rot_y = sin_t * diff[..., 0] + cos_t * diff[..., 1]

    # Mahalanobis: (B, N, 6, 3)
    sigmas_exp = sigmas.unsqueeze(1)  # (B, 1, 6, 3, 2)
    mahal_sq = diff_rot_x**2 / sigmas_exp[..., 0] + diff_rot_y**2 / sigmas_exp[..., 1]

    # Blob values: (B, N, 6, 3)
    blob_vals = amplitudes.unsqueeze(1) * torch.exp(-0.5 * mahal_sq)

    # Mask and sum: (B, N)
    penalty = (blob_vals * presence.unsqueeze(1)).sum(dim=(2, 3))

    return d_box - penalty
```

---

## Integration Plan

### Option A: Replace in-place (simpler)
Replace the loop in `evaluate_level_set_batched` with vectorized code.

### Option B: Add new function + toggle (safer)
1. Add `evaluate_level_set_batched_vectorized` as new function
2. Add module-level toggle `_USE_VECTORIZED = True`
3. Have `evaluate_level_set_batched` dispatch to vectorized version

**Recommendation: Option B** - allows easy rollback and A/B testing.

---

## Testing Strategy

### Test 1: Numerical equivalence
```python
def test_vectorized_equivalence():
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    B, N = 50, 100
    points = torch.randn(N, 4, device=device)
    lower = torch.randn(B, 4, device=device)
    upper = lower + torch.rand(B, 4, device=device) + 0.1
    weights = torch.rand(B, 4, device=device) + 0.5
    presence = torch.rand(B, 18, device=device)
    blob_params = torch.randn(B, 18, 6, device=device)

    result_loop = evaluate_level_set_batched(points, lower, upper, weights, presence, blob_params)
    result_vec = evaluate_level_set_batched_vectorized(points, lower, upper, weights, presence, blob_params)

    assert torch.allclose(result_loop, result_vec, atol=1e-5, rtol=1e-4), \
        f"Max diff: {(result_loop - result_vec).abs().max()}"
```

### Test 2: Gradient equivalence
```python
def test_gradient_equivalence():
    # Same setup with requires_grad=True on blob_params
    blob_params = torch.randn(B, 18, 6, device=device, requires_grad=True)

    result_loop = evaluate_level_set_batched(..., blob_params)
    result_loop.sum().backward()
    grad_loop = blob_params.grad.clone()

    blob_params.grad = None
    result_vec = evaluate_level_set_batched_vectorized(..., blob_params)
    result_vec.sum().backward()
    grad_vec = blob_params.grad.clone()

    assert torch.allclose(grad_loop, grad_vec, atol=1e-5)
```

### Test 3: Performance benchmark
```python
def benchmark():
    # Warmup
    for _ in range(10):
        evaluate_level_set_batched(...)
        evaluate_level_set_batched_vectorized(...)

    torch.cuda.synchronize()

    # Benchmark loop version
    start = time.perf_counter()
    for _ in range(100):
        evaluate_level_set_batched(...)
    torch.cuda.synchronize()
    time_loop = time.perf_counter() - start

    # Benchmark vectorized
    start = time.perf_counter()
    for _ in range(100):
        evaluate_level_set_batched_vectorized(...)
    torch.cuda.synchronize()
    time_vec = time.perf_counter() - start

    print(f"Loop: {time_loop*10:.2f}ms, Vectorized: {time_vec*10:.2f}ms")
    print(f"Speedup: {time_loop/time_vec:.2f}x")
```

---

## Potential Issues

1. **Memory usage**: The vectorized version creates a (B, N, 6, 3, 2) tensor for `diff`, which is 6x larger than the loop version's (B, N, 3, 2). For B=50, N=100, this is 50×100×6×3×2×4 = 720KB - negligible.

2. **Numerical precision**: The order of operations differs slightly. Should be within floating point tolerance.

3. **Edge cases**:
   - Empty points (N=0): Should return (B, 0) tensor
   - Single sample (B=1): Should work with broadcasting
   - K != 18: Current code assumes 18 slots. Need to handle dynamically.

---

## Rollout Plan

1. Implement vectorized function
2. Add toggle flag, default OFF
3. Run test suite with both versions
4. Enable vectorized on GPU only
5. Benchmark and verify speedup
6. Make vectorized the default
