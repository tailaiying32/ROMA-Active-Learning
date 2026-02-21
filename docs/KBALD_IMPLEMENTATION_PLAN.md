# k-BALD Batch Test Selection Strategy

## Overview

Implement a **k-BALD** (greedy BatchBALD approximation) test selection strategy that selects tests in batches of k points, ensuring diversity through conditional information gain.

**Key Difference from G-BALD**: Instead of geometric diversity heuristics, k-BALD uses information-theoretic diversity - each subsequent point in the batch is selected to maximize information gain *given* the previous selections.

## Problem with Current Approaches

1. **Pure BALD**: Selects one test at a time. Can be myopic - multiple queries may target the same uncertain region.

2. **G-BALD**: Uses geometric diversity (ellipsoid-based), but the boundary penalty actually encourages clustering near the query center.

3. **Existing BatchBALD**: Pre-selects all queries upfront from the prior, doesn't adapt to posterior updates.

## k-BALD Algorithm

### Core Idea
Greedily select k points that jointly maximize mutual information:

```
I(y₁, y₂, ..., yₖ ; θ | x₁, x₂, ..., xₖ)
```

**Greedy Approximation**:
1. Select x₁ = argmax BALD(x)
2. Select x₂ = argmax BALD(x | x₁ selected)
3. Select x₃ = argmax BALD(x | x₁, x₂ selected)
4. ... repeat until k points selected

### Conditional BALD Approximation

For the j-th selection given previous selections S = {x₁, ..., x_{j-1}}:

```
BALD_conditional(x | S) ≈ BALD(x) - redundancy(x, S)
```

**Redundancy term** (correlation-based):
```
redundancy(x, S) = λ · max_{s ∈ S} |corr(p(y|x,θ), p(y|s,θ))|
```

Where:
- `p(y|x,θ)` is the prediction distribution at x across posterior samples
- `corr` measures how similar the predictions are (if they agree, querying both is redundant)
- `λ` controls diversity strength

### Batch Workflow

```
for each batch:
    1. Sample θ₁, ..., θₘ from posterior (or use SVGD particles)
    2. Compute p(y|x,θᵢ) for all candidates x and all samples i
    3. Greedy k-BALD selection:
       - Select x₁ with highest BALD
       - For j = 2 to k:
         - Compute conditional BALD for remaining candidates
         - Select xⱼ with highest conditional BALD
    4. Query oracle for all k points
    5. Update posterior with all k observations
    6. Repeat until budget exhausted
```

## Implementation Plan

### Step 1: Create `kbald.py`

**File**: `active_learning/src/kbald.py`

```python
class KBaldStrategy:
    """
    k-BALD: Greedy BatchBALD approximation for batch test selection.

    Selects k tests per batch by greedily maximizing conditional BALD,
    which accounts for redundancy between selected points.
    """

    def __init__(self, decoder, posterior, config):
        # k-BALD specific config
        self.batch_size = config.get('kbald', {}).get('batch_size', 5)
        self.n_candidates = config.get('kbald', {}).get('n_candidates', 5000)
        self.diversity_weight = config.get('kbald', {}).get('diversity_weight', 0.5)
        self.tau = config.get('kbald', {}).get('tau', 0.1)

    def compute_score(self, test_point, zs=None, iteration=None):
        """Compute standard BALD score (for compatibility)."""

    def select_batch(self, bounds, zs=None):
        """
        Select k points using greedy conditional BALD.

        Returns:
            List of k test points
        """

    def _compute_bald_scores(self, candidates, probs):
        """Compute BALD scores for all candidates."""

    def _compute_redundancy(self, candidate_probs, selected_probs):
        """Compute redundancy between candidate and selected points."""

    def select_test(self, bounds, **kwargs):
        """
        Interface for LatentActiveLearner.

        Returns next test from current batch, or selects new batch if empty.
        """
```

### Step 2: Batch State Management

The strategy needs to maintain state for the current batch:

```python
class KBaldStrategy:
    def __init__(self, ...):
        self.current_batch = []  # Remaining points in current batch
        self.batch_index = 0     # Which point in batch we're on

    def select_test(self, bounds, **kwargs):
        # If batch empty, select new batch
        if not self.current_batch:
            self.current_batch = self.select_batch(bounds)
            self.batch_index = 0

        # Return next point from batch
        test_point = self.current_batch.pop(0)
        return test_point, self._compute_bald_score(test_point), []
```

### Step 3: Register in Factory

**File**: `active_learning/src/factory.py`

Add to `_build_strategy()`:
```python
elif strategy_type == 'kbald':
    from active_learning.src.kbald import KBaldStrategy
    return KBaldStrategy(decoder=decoder, posterior=posterior, config=config)
```

Update error message to include 'kbald'.

### Step 4: Add Configuration

**File**: `active_learning/configs/latent.yaml`

```yaml
# ===== k-BALD Batch Selection Settings =====
kbald:
  batch_size: 5              # Number of tests to select per batch
  n_candidates: 5000         # Candidate pool size for selection
  diversity_weight: 0.5      # Weight for redundancy penalty (0 = pure BALD)
  tau: 0.1                   # Sigmoid temperature
  tau_schedule:              # Optional tau annealing
    start: 0.3
    end: 0.1
    duration: 100
    schedule: linear
```

### Step 5: Update CLI

**Files**:
- `active_learning/test/diagnostics/run_latent_diagnosis.py`
- `active_learning/test/latent/compare_all.py`

Add 'kbald' to strategy choices and factory mappings.

## Algorithm Details

### Candidate Generation

```python
def _generate_candidates(self, bounds, n_candidates):
    """Generate random candidates within bounds."""
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return lower + torch.rand(n_candidates, len(bounds)) * (upper - lower)
```

### BALD Score Computation

```python
def _compute_bald_scores(self, candidates, zs):
    """
    Compute BALD scores for all candidates.

    Args:
        candidates: (N, n_joints)
        zs: (K, latent_dim) posterior samples

    Returns:
        bald_scores: (N,)
        probs: (K, N) - predictions for redundancy computation
    """
    logits = LatentFeasibilityChecker.batched_logit_values(
        self.decoder, zs, candidates
    )
    probs = torch.sigmoid(logits / self.tau)  # (K, N)

    # BALD = H(E[p]) - E[H(p)]
    mean_probs = probs.mean(dim=0)  # (N,)
    entropy_of_mean = binary_entropy(mean_probs)
    mean_of_entropies = binary_entropy(probs).mean(dim=0)
    bald_scores = entropy_of_mean - mean_of_entropies

    return bald_scores, probs
```

### Redundancy Computation

```python
def _compute_redundancy(self, candidate_probs, selected_probs_list):
    """
    Compute redundancy between candidates and already-selected points.

    Args:
        candidate_probs: (K, N) predictions for candidates
        selected_probs_list: List of (K,) predictions for selected points

    Returns:
        redundancy: (N,) max correlation with any selected point
    """
    if not selected_probs_list:
        return torch.zeros(candidate_probs.shape[1])

    max_redundancy = torch.zeros(candidate_probs.shape[1])

    for selected_probs in selected_probs_list:
        # Correlation between candidate and selected predictions
        # Both are (K,) vectors of predictions across posterior samples
        # High correlation = redundant information

        # Normalize to zero mean
        cand_centered = candidate_probs - candidate_probs.mean(dim=0, keepdim=True)
        sel_centered = selected_probs - selected_probs.mean()

        # Correlation: (N,)
        corr = (cand_centered * sel_centered.unsqueeze(1)).mean(dim=0)
        corr = corr / (cand_centered.std(dim=0) * sel_centered.std() + 1e-8)

        max_redundancy = torch.max(max_redundancy, corr.abs())

    return max_redundancy
```

### Greedy Batch Selection

```python
def select_batch(self, bounds, zs=None):
    """Select k points using greedy conditional BALD."""

    # Generate candidates
    candidates = self._generate_candidates(bounds, self.n_candidates)

    # Get posterior samples
    if zs is None:
        if hasattr(self.posterior, 'get_particles'):
            zs = self.posterior.get_particles()
        else:
            zs = self.posterior.sample(32)

    # Compute BALD scores and prediction probs
    bald_scores, probs = self._compute_bald_scores(candidates, zs)

    selected_indices = []
    selected_probs = []

    for _ in range(self.batch_size):
        # Compute conditional BALD
        redundancy = self._compute_redundancy(probs, selected_probs)
        conditional_bald = bald_scores - self.diversity_weight * redundancy

        # Mask already selected
        for idx in selected_indices:
            conditional_bald[idx] = -float('inf')

        # Select best
        best_idx = conditional_bald.argmax().item()
        selected_indices.append(best_idx)
        selected_probs.append(probs[:, best_idx])

    return [candidates[i] for i in selected_indices]
```

## Comparison with Existing Strategies

| Strategy | Selection | Diversity | Posterior Updates |
|----------|-----------|-----------|-------------------|
| BALD | 1 at a time | None (myopic) | After each query |
| Multi-Stage Warmup | Batch (boundary targeting) | Farthest point sampling | After each batch |
| G-BALD | 1 at a time | Geometric (ellipsoid) | After each query |
| **k-BALD** | Batch of k | Information-theoretic (redundancy) | After each batch |

## Expected Benefits

1. **Information-theoretic diversity**: Unlike G-BALD's geometric approach, k-BALD ensures selected points provide complementary information about the posterior.

2. **Batch efficiency**: Selecting k points at once is more efficient than k sequential BALD queries, especially when posterior updates are expensive.

3. **No geometric assumptions**: Doesn't require ellipsoid fitting or boundary penalties that can cause clustering.

4. **Adapts to posterior**: Diversity is measured in prediction space (how particles disagree), not geometric space.

## Files to Create/Modify

| File | Action |
|------|--------|
| `active_learning/src/kbald.py` | **CREATE** - Main k-BALD strategy |
| `active_learning/src/factory.py` | MODIFY - Register 'kbald' strategy |
| `active_learning/configs/latent.yaml` | MODIFY - Add kbald config section |
| `active_learning/test/diagnostics/run_latent_diagnosis.py` | MODIFY - Add 'kbald' to CLI choices |
| `active_learning/test/latent/compare_all.py` | MODIFY - Add 'kbald' to CLI and factory mappings |

## Testing

1. Run diagnostic script with k-BALD:
   ```bash
   python run_latent_diagnosis.py --strategy kbald --budget 40
   ```

2. Compare against BALD and multi-stage-warmup:
   ```bash
   python compare_all.py --strategies kbald bald multi-stage-warmup --trials 5 --budget 40
   ```

3. Verify batch selection produces diverse points by inspecting query history visualization.
