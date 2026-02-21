# GPU Optimization Implementation Plan

## Executive Summary

The active learning pipeline currently runs slower on GPU than CPU due to:
1. **Python for loops inside hot paths** causing excessive kernel launches
2. **Redundant decoder calls** in the SVGD inner loop
3. **Small tensor operations** that don't amortize GPU overhead
4. **Potential CPU-GPU data transfers** in test history handling

This plan addresses each issue systematically to achieve **3-10× speedup on GPU**.

---

## Phase 1: Vectorize Level Set Evaluation (HIGH IMPACT)

### Problem
`evaluate_level_set_batched` in `infer_params/training/level_set_torch.py` has a Python for loop over 6 joint pairs (lines 195-234). Each iteration launches ~10 CUDA kernels.

**Current code:**
```python
for pair_idx, (i, j) in enumerate(joint_pairs):
    # ~10 tensor operations per iteration
    x_2d = torch.stack([points_batched[:, :, i], points_batched[:, :, j]], dim=-1)
    diff = x_2d.unsqueeze(2) - pair_centers.unsqueeze(1)
    # ... more operations
    penalty = penalty + pair_penalty
```

**Called:** ~900 times per active learning iteration (50 SVGD steps + 100 BALD steps × decay)

### Solution
Vectorize across all 6 joint pairs simultaneously.

### File: `infer_params/training/level_set_torch.py`

#### Step 1.1: Add vectorized helper function

Add after line 137 (after `gaussian_blob_value`):

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
    """Fully vectorized level-set evaluation - no Python loops.

    Same interface as evaluate_level_set_batched but ~6x faster on GPU
    by eliminating the joint-pair loop.
    """
    B = lower.shape[0]
    N = points.shape[0]
    device = lower.device

    # Expand points for batched evaluation: (N, 4) -> (B, N, 4)
    points_batched = points.unsqueeze(0).expand(B, -1, -1)

    # Compute box distance (already vectorized)
    d_box = weighted_box_distance(points_batched, lower, upper, weights, alpha)

    # === VECTORIZED BLOB PENALTY ===
    # Joint pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Pre-compute all pair indices
    pair_i = torch.tensor([0, 0, 0, 1, 1, 2], device=device)
    pair_j = torch.tensor([1, 2, 3, 2, 3, 3], device=device)
    num_pairs = 6
    slots_per_pair = presence.shape[1] // num_pairs  # 3

    # Extract blob parameters: (B, K, 6) where K=18
    centers = blob_params[:, :, 0:2]  # (B, K, 2)
    sigmas = blob_params[:, :, 2:4].abs().clamp(min=1e-6)  # (B, K, 2)
    amplitudes = blob_params[:, :, 4].abs().clamp(min=0.01, max=0.5)  # (B, K)
    rotations = blob_params[:, :, 5]  # (B, K)

    # Reshape for per-pair processing: (B, num_pairs, slots_per_pair, ...)
    centers = centers.view(B, num_pairs, slots_per_pair, 2)
    sigmas = sigmas.view(B, num_pairs, slots_per_pair, 2)
    amplitudes = amplitudes.view(B, num_pairs, slots_per_pair)
    rotations = rotations.view(B, num_pairs, slots_per_pair)
    presence = presence.view(B, num_pairs, slots_per_pair)

    # Extract 2D points for ALL pairs at once
    # points_batched: (B, N, 4)
    # We need (B, N, num_pairs, 2) where last dim is (points[:,:,i], points[:,:,j])
    x_2d = torch.stack([
        points_batched[:, :, pair_i],  # (B, N, 6)
        points_batched[:, :, pair_j],  # (B, N, 6)
    ], dim=-1)  # (B, N, 6, 2)

    # Compute diff: (B, N, num_pairs, 1, 2) - (B, 1, num_pairs, slots_per_pair, 2)
    # -> (B, N, num_pairs, slots_per_pair, 2)
    diff = x_2d.unsqueeze(3) - centers.unsqueeze(1)

    # Apply rotation for all pairs/slots at once
    cos_t = torch.cos(-rotations).unsqueeze(1)  # (B, 1, num_pairs, slots_per_pair)
    sin_t = torch.sin(-rotations).unsqueeze(1)

    diff_rot_x = cos_t * diff[..., 0] - sin_t * diff[..., 1]  # (B, N, 6, 3)
    diff_rot_y = sin_t * diff[..., 0] + cos_t * diff[..., 1]

    # Mahalanobis distance
    sigmas_exp = sigmas.unsqueeze(1)  # (B, 1, 6, 3, 2)
    mahal_sq = (diff_rot_x ** 2 / sigmas_exp[..., 0] +
                diff_rot_y ** 2 / sigmas_exp[..., 1])  # (B, N, 6, 3)

    # Gaussian blob values
    amps_exp = amplitudes.unsqueeze(1)  # (B, 1, 6, 3)
    blob_vals = amps_exp * torch.exp(-0.5 * mahal_sq)  # (B, N, 6, 3)

    # Mask by presence and sum over slots and pairs
    pres_exp = presence.unsqueeze(1)  # (B, 1, 6, 3)
    penalty = (blob_vals * pres_exp).sum(dim=(-1, -2))  # (B, N)

    return d_box - penalty
```

#### Step 1.2: Add toggle to use vectorized version

Modify `evaluate_level_set_batched` to optionally use the vectorized version:

```python
# At module level, add flag
_USE_VECTORIZED = True

def evaluate_level_set_batched(...):
    if _USE_VECTORIZED:
        return evaluate_level_set_batched_vectorized(
            points, lower, upper, weights, presence, blob_params, alpha
        )
    # ... existing code as fallback
```

#### Step 1.3: Add torch.compile decorator (PyTorch 2.0+)

```python
# At module level after imports
import sys

# Only compile if PyTorch 2.0+ and not in debug mode
_COMPILE_ENABLED = (
    hasattr(torch, 'compile') and
    not getattr(sys, 'gettrace', lambda: None)()
)

if _COMPILE_ENABLED:
    evaluate_level_set_batched_vectorized = torch.compile(
        evaluate_level_set_batched_vectorized,
        mode="reduce-overhead"  # Optimizes for small tensors
    )
```

### Expected Impact
- **Kernel launches**: 6× reduction (900 → 150 per iteration)
- **GPU speedup**: 3-5× for level set evaluation
- **Overall iteration speedup**: 1.5-2×

### Testing
```python
# Test equivalence
points = torch.randn(100, 4, device='cuda')
lower = torch.randn(50, 4, device='cuda')
# ... other params

result_loop = evaluate_level_set_batched(...)
result_vec = evaluate_level_set_batched_vectorized(...)
assert torch.allclose(result_loop, result_vec, atol=1e-5)
```

---

## Phase 2: Cache Decoded Parameters in SVGD (MEDIUM IMPACT)

### Problem
The SVGD inner loop calls `log_likelihood` at every iteration, which internally calls `batched_logit_values`, which decodes the particles. But particles only move slightly between iterations.

**Current code in `svgd_vi.py` lines 278-283:**
```python
for i in range(self.max_iters):
    p_in = particles.detach().requires_grad_(True)
    ll = self.log_likelihood(test_history, p_in, iteration=iteration)  # DECODES EVERY TIME
```

### Solution
Decode once at the start, cache the decoded parameters, and use them throughout the inner loop. Only pass through the decoder for gradient computation.

### File: `active_learning/src/svgd/svgd_vi.py`

#### Step 2.1: Add cached likelihood computation method

Add after `log_likelihood` method (around line 198):

```python
def log_likelihood_from_decoded(
    self,
    test_history: TestHistory,
    particles: torch.Tensor,
    decoded_params: tuple,
    iteration: int = None
) -> torch.Tensor:
    """
    Compute log likelihood using pre-decoded parameters.

    This avoids re-decoding particles when they haven't changed much.
    The decoded_params should be computed with gradients enabled if
    you need to backprop through particles.

    Args:
        test_history: Test history
        particles: (K, D) particles (not used directly, for shape only)
        decoded_params: Tuple from decode_latent_params (lower, upper, weights, pres_logits, blob_params)
        iteration: Current iteration for tau scheduling

    Returns:
        (K,) log likelihood per particle
    """
    results = test_history.get_all()
    if not results:
        return torch.zeros(particles.shape[0], device=particles.device)

    # Stack test points and outcomes
    test_points = torch.stack([r.test_point for r in results]).to(particles.device)
    outcomes = torch.tensor([r.outcome for r in results], device=particles.device).unsqueeze(0)

    # Use pre-decoded params
    pred_logits = LatentFeasibilityChecker.evaluate_from_decoded(test_points, decoded_params)

    # Scale and compute BCE
    current_tau = self._get_current_tau(iteration)
    scaled_logits = pred_logits / current_tau
    targets_expanded = outcomes.expand_as(scaled_logits)

    neg_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        scaled_logits, targets_expanded, reduction='none'
    )

    return -neg_bce.sum(dim=1)
```

#### Step 2.2: Modify update_posterior to use caching

Replace the inner loop section (lines 278-316) with:

```python
        # Pre-decode for cached likelihood (with gradients for first iter)
        # We'll re-decode periodically or when movement is large
        REDECODE_INTERVAL = 10  # Re-decode every N iterations
        MOVEMENT_THRESHOLD = 0.1  # Re-decode if particles move more than this

        particles_at_last_decode = particles.clone()

        for i in range(self.max_iters):
            # 1. Detach particles and enable grad
            p_in = particles.detach().requires_grad_(True)

            # 2. Decide whether to re-decode
            movement = (particles - particles_at_last_decode).norm(dim=-1).mean()
            should_redecode = (
                i == 0 or
                i % REDECODE_INTERVAL == 0 or
                movement > MOVEMENT_THRESHOLD
            )

            if should_redecode:
                # Decode with gradients
                decoded_params = LatentFeasibilityChecker.decode_latent_params(
                    self.decoder, p_in
                )
                particles_at_last_decode = particles.clone()

            # 3. Compute Log Joint using cached decode
            ll = self.log_likelihood_from_decoded(test_history, p_in, decoded_params, iteration)
            lp = self.log_prior(p_in)

            # Compute gradients
            ll_grad = torch.autograd.grad(ll.sum(), p_in, retain_graph=True)[0]
            lp_grad = torch.autograd.grad(lp.sum(), p_in)[0]
            log_prob_grad = ll_grad + kw * lp_grad

            # ... rest of loop unchanged
```

**Note:** This is a simplified version. For full correctness, you need to ensure gradients flow through `decoded_params` when computing `ll_grad`. The above approach re-decodes periodically which maintains gradient correctness at those points.

#### Step 2.3: Alternative - Decode once, approximate gradients

For maximum speed with slight approximation:

```python
        # Decode ONCE at the start with gradients
        p_init = particles.detach().requires_grad_(True)
        decoded_params = LatentFeasibilityChecker.decode_latent_params(self.decoder, p_init)

        for i in range(self.max_iters):
            p_in = particles.detach().requires_grad_(True)

            # Use cached decode (approximate - ignores decoder Jacobian)
            ll = self.log_likelihood_from_decoded(test_history, p_in, decoded_params, iteration)
            lp = self.log_prior(p_in)

            # For approximate gradients, we only differentiate through evaluate_from_decoded
            # This misses d(decoder)/d(particles) but particles move little per step
            ll_grad = torch.autograd.grad(ll.sum(), p_in, retain_graph=True)[0]
            lp_grad = torch.autograd.grad(lp.sum(), p_in)[0]
            log_prob_grad = ll_grad + kw * lp_grad

            # ... rest unchanged
```

This is an approximation but should work well in practice since:
1. Particles move by small amounts each step
2. The decoder is relatively smooth
3. The prior gradient (which is exact) provides regularization

### Expected Impact
- **Decoder calls**: 50× reduction (from 50 to 1-5 per update)
- **GPU speedup for SVGD**: 2-3×
- **Overall iteration speedup**: 1.3-1.5×

---

## Phase 3: Ensure All Data Stays on GPU (LOW-MEDIUM IMPACT)

### Problem
Test history tensors may be created on CPU and transferred each `log_likelihood` call.

### File: `active_learning/src/test_history.py`

#### Step 3.1: Store device reference and keep tensors on GPU

```python
class TestHistory:
    '''Stores and manages test history.'''

    def __init__(self, joint_names: list[str], device: str = 'cpu'):
        '''
        Initialize empty history with joint names.
        '''
        self.joint_names = joint_names
        self._results: list[TestResult] = []
        self.device = device

        # Cached tensors for efficient access
        self._test_points_cache: Optional[torch.Tensor] = None
        self._outcomes_cache: Optional[torch.Tensor] = None
        self._cache_valid = False

    def add(self, test_point: torch.Tensor, outcome: float, ...):
        # Ensure tensor is on correct device
        test_point = test_point.to(self.device)
        result = TestResult(
            test_point=test_point.clone().detach(),
            ...
        )
        self._results.append(result)
        self._cache_valid = False  # Invalidate cache
        return result

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached test points and outcomes tensors."""
        if not self._cache_valid and self._results:
            self._test_points_cache = torch.stack(
                [r.test_point for r in self._results]
            ).to(self.device)
            self._outcomes_cache = torch.tensor(
                [r.outcome for r in self._results],
                device=self.device
            )
            self._cache_valid = True
        return self._test_points_cache, self._outcomes_cache
```

### File: `active_learning/src/svgd/svgd_vi.py`

#### Step 3.2: Use cached tensors in log_likelihood

```python
def log_likelihood(self, test_history: TestHistory, particles: torch.Tensor, ...):
    results = test_history.get_all()
    if not results:
        return torch.zeros(particles.shape[0], device=particles.device)

    # Use cached tensors instead of stacking each time
    test_points, outcomes = test_history.get_tensors()
    test_points = test_points.to(particles.device)
    outcomes = outcomes.to(particles.device).unsqueeze(0)

    # ... rest unchanged
```

### Expected Impact
- **CPU-GPU transfers**: Eliminated after first call
- **Tensor allocation**: Reduced by caching
- **Overall speedup**: 1.1-1.2×

---

## Phase 4: Enable GPU and Verify (REQUIRED)

### File: `active_learning/src/config.py`

#### Step 4.1: Enable GPU by default

```python
# Line 15-16, change:
# DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
```

#### Step 4.2: Add GPU memory monitoring (optional)

```python
def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
```

---

## Phase 5: Apply torch.compile to Hot Functions (EASY WIN)

### File: `active_learning/src/svgd/svgd_optimizer.py`

```python
import torch

class SVGD:
    def __init__(self, ...):
        ...
        # Compile the step function for faster execution
        if hasattr(torch, 'compile'):
            self._step_compiled = torch.compile(self._step_impl, mode="reduce-overhead")
        else:
            self._step_compiled = self._step_impl

    def _step_impl(self, particles, log_prob_grad):
        # Move existing step() code here
        ...

    def step(self, particles, log_prob_grad, return_diagnostics=False):
        if return_diagnostics:
            # Can't compile with dynamic return
            return self._step_impl_with_diagnostics(particles, log_prob_grad)
        return self._step_compiled(particles, log_prob_grad)
```

### File: `active_learning/src/svgd/sliced_svgd_optimizer.py`

Same pattern for `SlicedSVGD.step()`.

---

## Implementation Order

| Phase | Effort | Impact | Dependencies |
|-------|--------|--------|--------------|
| **Phase 4** (Enable GPU) | 5 min | Required | None |
| **Phase 1** (Vectorize level set) | 2-3 hrs | HIGH | None |
| **Phase 5** (torch.compile) | 30 min | MEDIUM | Phase 1 |
| **Phase 3** (GPU tensors) | 1 hr | LOW-MEDIUM | None |
| **Phase 2** (Cache decode) | 2 hrs | MEDIUM | Phase 1 |

**Recommended order:** 4 → 1 → 5 → 3 → 2

---

## Verification Checklist

After each phase:

1. **Correctness test:**
   ```python
   # Run with CPU, save results
   DEVICE = 'cpu'
   result_cpu = run_iteration()

   # Run with GPU
   DEVICE = 'cuda'
   result_gpu = run_iteration()

   # Compare
   assert torch.allclose(result_cpu.particles, result_gpu.particles, atol=1e-4)
   ```

2. **Performance test:**
   ```python
   import time

   # Warmup
   for _ in range(3):
       run_iteration()

   # Benchmark
   torch.cuda.synchronize()
   start = time.perf_counter()
   for _ in range(10):
       run_iteration()
   torch.cuda.synchronize()
   elapsed = time.perf_counter() - start

   print(f"Time per iteration: {elapsed/10*1000:.1f}ms")
   ```

3. **Memory test:**
   ```python
   torch.cuda.reset_peak_memory_stats()
   run_iteration()
   peak_mb = torch.cuda.max_memory_allocated() / 1024**2
   print(f"Peak GPU memory: {peak_mb:.1f}MB")
   ```

---

## Expected Final Performance

| Metric | Current (CPU) | After Optimization (GPU) |
|--------|---------------|--------------------------|
| Iteration time | ~3000ms | ~300-500ms |
| BALD select_test | ~1500ms | ~150-250ms |
| SVGD update | ~1400ms | ~150-250ms |
| Kernel launches/iter | ~900 | ~100-150 |

**Target: 6-10× speedup on GPU vs current CPU baseline.**

---

## Fallback Options

If GPU is still slower after optimizations:

1. **CUDA Graphs** (advanced): Capture entire SVGD loop as a graph
   ```python
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       run_svgd_step()
   # Replay without Python overhead
   for _ in range(max_iters):
       g.replay()
   ```

2. **Mixed precision** (easy): Use float16 for decoder
   ```python
   with torch.autocast(device_type='cuda', dtype=torch.float16):
       decoded = decoder(particles)
   ```

3. **Batch across iterations** (medium): Compute multiple SVGD steps in parallel
