# G-BALD Implementation Plan

## Overview

**Goal**: Implement Geometric BALD (G-BALD) as a new test selection strategy that combines uncertainty-based acquisition (BALD) with geometric diversity via ellipsoid-based core-set selection.

**Strategy Name**: `gbald` (registered in factory alongside `bald`, `random`, etc.)

---

## Part 1: Mathematical Foundation

### 1.1 Standard BALD Recap

BALD selects points that maximize mutual information between the prediction and model parameters:

```
BALD(x) = I(y; θ | x) = H[y | x] - E_θ[H[y | x, θ]]
```

For binary classification (feasibility):
```
BALD(x) = H(p̄) - (1/K) Σₖ H(pₖ)
```

Where:
- `p̄ = (1/K) Σₖ pₖ` is the mean prediction across K particles/samples
- `H(p) = -p log(p) - (1-p) log(1-p)` is binary entropy
- `pₖ = σ(f(x; θₖ))` is the feasibility probability under particle k

**Problem with BALD**: It only considers uncertainty, ignoring the spatial distribution of queries. This can lead to:
1. **Redundant queries**: Repeatedly selecting nearby high-uncertainty points
2. **Poor coverage**: Missing informative regions that happen to have lower uncertainty
3. **Boundary artifacts**: Selecting extreme/outlier points

### 1.2 Core-Set Intuition

The core-set approach ensures selected points "cover" the input space. Given a set of already-queried points S, we want new queries to be far from S:

```
diversity(x) = min_{s ∈ S} d(x, s)
```

This is the **k-Center problem**: select points to minimize the maximum distance from any point to its nearest selected point.

**Standard approach uses Euclidean distance**:
```
d(x, s) = ||x - s||₂
```

### 1.3 Why Ellipsoid Geometry?

**Problem with spherical/Euclidean distance**: Treats all directions equally, but:
- The data distribution may be anisotropic (stretched in some directions)
- Points at the "boundary" of the distribution may appear diverse but are outliers
- Doesn't account for the natural geometry of the feasible region

**Ellipsoid solution**: Use Mahalanobis distance based on the covariance of queried points:

```
d_ellipsoid(x, s) = √[(x - s)ᵀ A (x - s)]
```

Where `A = Σ⁻¹` is the inverse covariance matrix of the query history.

**Geometric interpretation**:
- The ellipsoid `{x : (x - μ)ᵀ A (x - μ) ≤ 1}` represents the "typical" region
- Distance is measured along the ellipsoid's principal axes
- Points in low-variance directions contribute more to diversity

### 1.4 Boundary Penalty

G-BALD adds a penalty for points near the ellipsoid boundary:

```
boundary_penalty(x) = σ(-η · (d_center(x) - 1))
```

Where:
- `d_center(x) = (x - μ)ᵀ A (x - μ)` is squared Mahalanobis distance to center
- `η` controls penalty sharpness
- Points with `d_center > 1` (outside the ellipsoid) get penalized

**Why?** Points at distribution boundaries are often:
- Less representative of the true feasible region
- More likely to be noisy/outliers
- Less informative for learning the core structure

### 1.5 G-BALD Acquisition Function

The final acquisition combines uncertainty and geometric diversity:

```
G-BALD(x) = BALD(x) · diversity(x) · boundary_penalty(x)
```

Or in expanded form:

```
G-BALD(x) = [H(p̄) - (1/K) Σₖ H(pₖ)] · [min_{s ∈ S} d_ellipsoid(x, s)] · [σ(-η · (d_center(x) - 1))]
            \_________________________/   \____________________________/   \__________________________/
                   Uncertainty                    Core-set diversity              Boundary penalty
```

---

## Part 2: Why G-BALD Will Be Better

### 2.1 Addressing BALD's Weaknesses

| BALD Weakness | G-BALD Solution |
|---------------|-----------------|
| Redundant queries in high-uncertainty regions | Core-set diversity term forces spatial spread |
| No notion of coverage | Ellipsoid distance ensures coverage of the "typical" region |
| May select outlier points | Boundary penalty discourages extreme points |
| Ignores query history geometry | Ellipsoid adapts to the shape of queried region |

### 2.2 Theoretical Advantages

1. **Tighter error bounds**: G-BALD paper proves that ellipsoid geodesic search achieves tighter lower bounds on generalization error than spherical search.

2. **Robustness to noise**: By penalizing boundary points, G-BALD is less affected by noisy labels at extreme configurations.

3. **Adaptive geometry**: The ellipsoid is fitted to query history, so it adapts as learning progresses.

### 2.3 Expected Benefits for Your Pipeline

**Current situation**:
- SVGD particles may cluster in certain latent regions
- BALD selects based on particle disagreement alone
- May repeatedly query the same boundary segment

**With G-BALD**:
- Queries will spread across the entire feasible region
- Boundary learning will be more uniform
- Fewer queries needed to achieve same IoU (hypothesis)

---

## Part 3: Implementation Plan

### 3.1 File Structure

```
active_learning/src/
├── gbald.py                    # NEW: G-BALD implementation
├── ellipsoid.py                # NEW: Ellipsoid fitting utilities
├── factory.py                  # MODIFY: Register 'gbald' strategy
└── latent_active_learner.py    # NO CHANGE (uses factory)
```

### 3.2 Step-by-Step Implementation

#### Step 1: Create `ellipsoid.py` - Ellipsoid Utilities

```python
# active_learning/src/ellipsoid.py
"""
Ellipsoid geometry utilities for G-BALD.

Provides:
- Minimum-volume enclosing ellipsoid (MVEE) fitting
- Mahalanobis distance computation
- Boundary penalty calculation
"""

import torch
from typing import Tuple, Optional


class AdaptiveEllipsoid:
    """
    Incrementally-updated ellipsoid for core-set diversity.

    Uses online covariance estimation for efficiency.
    Falls back to spherical distance when insufficient points.
    """

    def __init__(self, dim: int, device: str = 'cuda',
                 regularization: float = 1e-4,
                 min_points_for_ellipsoid: int = None):
        """
        Args:
            dim: Dimensionality of the space (e.g., 4 for joint space)
            device: Torch device
            regularization: Ridge regularization for covariance inversion
            min_points_for_ellipsoid: Minimum points before using ellipsoid
                                      Default: dim + 1
        """
        self.dim = dim
        self.device = device
        self.reg = regularization
        self.min_points = min_points_for_ellipsoid or (dim + 1)

        # Online statistics
        self.n = 0
        self.mean = torch.zeros(dim, device=device)
        self.M2 = torch.zeros(dim, dim, device=device)  # For Welford's algorithm

        # Cached ellipsoid parameters
        self._A = None  # Precision matrix (inverse covariance)
        self._A_valid = False

    def update(self, point: torch.Tensor):
        """
        Update ellipsoid with a new point using Welford's online algorithm.

        Args:
            point: New point tensor of shape (dim,)
        """
        self.n += 1
        delta = point - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = point - self.mean
        self.M2 = self.M2 + torch.outer(delta, delta2)
        self._A_valid = False  # Invalidate cache

    def get_precision_matrix(self) -> Optional[torch.Tensor]:
        """
        Get the precision matrix A = Σ⁻¹.

        Returns:
            A: Precision matrix of shape (dim, dim), or None if insufficient points
        """
        if self.n < self.min_points:
            return None

        if not self._A_valid:
            # Covariance from Welford's M2
            cov = self.M2 / (self.n - 1)

            # Regularized inverse
            cov_reg = cov + self.reg * torch.eye(self.dim, device=self.device)
            self._A = torch.linalg.inv(cov_reg)
            self._A_valid = True

        return self._A

    def mahalanobis_distance_sq(self, x: torch.Tensor,
                                 y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute squared Mahalanobis distance.

        Args:
            x: Points of shape (N, dim) or (dim,)
            y: Reference points of shape (M, dim) or (dim,), or None for center

        Returns:
            If y is None: distances to center, shape (N,) or scalar
            If y is given: pairwise distances, shape (N, M)
        """
        A = self.get_precision_matrix()

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if y is None:
            # Distance to ellipsoid center
            diff = x - self.mean  # (N, dim)
            # (x - μ)ᵀ A (x - μ)
            return torch.einsum('ni,ij,nj->n', diff, A, diff)
        else:
            if y.dim() == 1:
                y = y.unsqueeze(0)
            # Pairwise distances
            diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, dim)
            return torch.einsum('nmi,ij,nmj->nm', diff, A, diff)

    def euclidean_distance_sq(self, x: torch.Tensor,
                               y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fallback Euclidean distance when ellipsoid not available.

        Same signature as mahalanobis_distance_sq.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if y is None:
            diff = x - self.mean
            return (diff ** 2).sum(dim=-1)
        else:
            if y.dim() == 1:
                y = y.unsqueeze(0)
            diff = x.unsqueeze(1) - y.unsqueeze(0)
            return (diff ** 2).sum(dim=-1)

    def distance_sq(self, x: torch.Tensor,
                    y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute squared distance, using ellipsoid if available, else Euclidean.
        """
        if self.get_precision_matrix() is not None:
            return self.mahalanobis_distance_sq(x, y)
        else:
            return self.euclidean_distance_sq(x, y)

    @property
    def center(self) -> torch.Tensor:
        """Return ellipsoid center (mean of points)."""
        return self.mean

    @property
    def is_ellipsoid_valid(self) -> bool:
        """Check if we have enough points for ellipsoid geometry."""
        return self.n >= self.min_points
```

#### Step 2: Create `gbald.py` - Main G-BALD Strategy

```python
# active_learning/src/gbald.py
"""
Geometric BALD (G-BALD) test selection strategy.

Combines:
1. BALD uncertainty (particle disagreement)
2. Ellipsoid-based core-set diversity
3. Boundary penalty to avoid outlier selections

Reference: "Bayesian Active Learning by Disagreements: A Geometric Perspective"
           Cao & Tsang, 2021 (arXiv:2105.02543)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any

from active_learning.src.ellipsoid import AdaptiveEllipsoid
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.utils import binary_entropy


class GeometricBALD:
    """
    G-BALD: Geometric Bayesian Active Learning by Disagreement.

    Acquisition function:
        G-BALD(x) = BALD(x) · diversity(x) · boundary_penalty(x)

    Where:
        - BALD(x) = H(p̄) - E[H(p)] (standard uncertainty)
        - diversity(x) = min_{s ∈ history} d_ellipsoid(x, s)
        - boundary_penalty(x) = σ(-η · (d_center(x) - 1))
    """

    def __init__(self, decoder, posterior, config: dict):
        """
        Args:
            decoder: LevelSetDecoder model
            posterior: SVGDPosterior or LatentUserDistribution
            config: Configuration dict with 'gbald' section
        """
        self.decoder = decoder
        self.posterior = posterior
        self.config = config

        # G-BALD specific config
        gbald_config = config.get('gbald', {})
        self.tau = gbald_config.get('tau', 0.1)  # Temperature for sigmoid
        self.n_samples = gbald_config.get('n_samples', 32)
        self.eta = gbald_config.get('eta', 2.0)  # Boundary penalty sharpness
        self.diversity_weight = gbald_config.get('diversity_weight', 1.0)
        self.boundary_weight = gbald_config.get('boundary_weight', 1.0)
        self.use_log_scale = gbald_config.get('use_log_scale', True)

        # Ellipsoid for diversity (in test/joint space)
        n_joints = config['prior'].get('n_joints', 4)
        device = config.get('device', 'cuda')
        self.ellipsoid = AdaptiveEllipsoid(dim=n_joints, device=device)

        # Query history (for core-set distance)
        self.query_history: List[torch.Tensor] = []

    def compute_score(self, test_points: torch.Tensor,
                      zs: Optional[torch.Tensor] = None,
                      iteration: int = 0) -> torch.Tensor:
        """
        Compute G-BALD acquisition scores for candidate test points.

        Args:
            test_points: Candidate points, shape (N, n_joints)
            zs: Optional pre-sampled latent codes, shape (K, latent_dim)
            iteration: Current iteration (for diagnostics)

        Returns:
            scores: G-BALD scores, shape (N,)
        """
        device = test_points.device

        # 1. Sample from posterior if not provided
        if zs is None:
            if hasattr(self.posterior, 'get_particles'):
                zs = self.posterior.get_particles()  # SVGD
            else:
                zs = self.posterior.sample(self.n_samples)  # VI

        # 2. Compute BALD scores (uncertainty component)
        bald_scores = self._compute_bald(test_points, zs)

        # 3. Compute diversity scores (core-set component)
        diversity_scores = self._compute_diversity(test_points)

        # 4. Compute boundary penalty
        boundary_scores = self._compute_boundary_penalty(test_points)

        # 5. Combine scores
        if self.use_log_scale:
            # Log-scale combination (more numerically stable)
            # G-BALD = exp(log(BALD) + λ_d·log(diversity) + λ_b·log(boundary))
            log_bald = torch.log(bald_scores + 1e-10)
            log_div = torch.log(diversity_scores + 1e-10)
            log_bound = torch.log(boundary_scores + 1e-10)

            log_scores = (log_bald +
                         self.diversity_weight * log_div +
                         self.boundary_weight * log_bound)
            scores = torch.exp(log_scores)
        else:
            # Multiplicative combination
            scores = bald_scores * (diversity_scores ** self.diversity_weight) * (boundary_scores ** self.boundary_weight)

        return scores

    def _compute_bald(self, test_points: torch.Tensor,
                      zs: torch.Tensor) -> torch.Tensor:
        """
        Compute standard BALD scores: I(y; θ | x) = H(p̄) - E[H(p)]

        Args:
            test_points: (N, n_joints)
            zs: (K, latent_dim)

        Returns:
            bald_scores: (N,)
        """
        # Get feasibility probabilities for each particle
        # probs shape: (K, N)
        logits = LatentFeasibilityChecker.batched_logit_values(
            self.decoder, zs, test_points
        )
        probs = torch.sigmoid(logits / self.tau)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)

        # Mean probability across particles: p̄ = (1/K) Σ pₖ
        mean_probs = probs.mean(dim=0)  # (N,)

        # Entropy of mean: H(p̄)
        entropy_of_mean = binary_entropy(mean_probs)  # (N,)

        # Mean of entropies: (1/K) Σ H(pₖ)
        entropies = binary_entropy(probs)  # (K, N)
        mean_of_entropies = entropies.mean(dim=0)  # (N,)

        # BALD = H(p̄) - E[H(p)]
        bald_scores = entropy_of_mean - mean_of_entropies

        # Ensure non-negative (numerical stability)
        bald_scores = F.relu(bald_scores)

        return bald_scores

    def _compute_diversity(self, test_points: torch.Tensor) -> torch.Tensor:
        """
        Compute core-set diversity: min distance to query history.

        diversity(x) = min_{s ∈ S} √[(x - s)ᵀ A (x - s)]

        Args:
            test_points: (N, n_joints)

        Returns:
            diversity_scores: (N,) - higher means more diverse
        """
        N = test_points.shape[0]
        device = test_points.device

        if len(self.query_history) == 0:
            # No history yet - all points equally diverse
            return torch.ones(N, device=device)

        # Stack history into tensor
        history = torch.stack(self.query_history)  # (H, n_joints)

        # Compute pairwise distances (squared)
        # Shape: (N, H)
        dist_sq = self.ellipsoid.distance_sq(test_points, history)

        # Min distance to any historical point
        min_dist_sq = dist_sq.min(dim=1).values  # (N,)

        # Return sqrt for actual distance
        diversity = torch.sqrt(min_dist_sq + 1e-10)

        # Normalize to [0, 1] range for stability
        if diversity.max() > 0:
            diversity = diversity / (diversity.max() + 1e-10)

        return diversity

    def _compute_boundary_penalty(self, test_points: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary penalty: penalize points far from ellipsoid center.

        boundary_penalty(x) = σ(-η · (d²_center(x) - 1))

        Points inside ellipsoid (d² < 1): penalty ≈ 1
        Points outside ellipsoid (d² > 1): penalty → 0

        Args:
            test_points: (N, n_joints)

        Returns:
            boundary_penalty: (N,) - values in (0, 1)
        """
        N = test_points.shape[0]
        device = test_points.device

        if not self.ellipsoid.is_ellipsoid_valid:
            # Not enough points for ellipsoid - no penalty
            return torch.ones(N, device=device)

        # Squared Mahalanobis distance to center
        dist_sq_to_center = self.ellipsoid.mahalanobis_distance_sq(test_points)

        # Sigmoid penalty: σ(-η · (d² - 1))
        # When d² < 1 (inside): -η·(d²-1) > 0, σ > 0.5
        # When d² > 1 (outside): -η·(d²-1) < 0, σ < 0.5
        penalty = torch.sigmoid(-self.eta * (dist_sq_to_center - 1))

        return penalty

    def update(self, query_point: torch.Tensor, outcome: float):
        """
        Update G-BALD state after a query.

        Args:
            query_point: The queried test point, shape (n_joints,)
            outcome: Query outcome (not used for ellipsoid, but kept for interface)
        """
        # Add to history
        self.query_history.append(query_point.detach().clone())

        # Update ellipsoid
        self.ellipsoid.update(query_point.detach())

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about G-BALD state."""
        return {
            'n_queries': len(self.query_history),
            'ellipsoid_valid': self.ellipsoid.is_ellipsoid_valid,
            'ellipsoid_center': self.ellipsoid.center.cpu().numpy() if self.ellipsoid.n > 0 else None,
            'eta': self.eta,
            'diversity_weight': self.diversity_weight,
            'boundary_weight': self.boundary_weight,
        }
```

#### Step 3: Create G-BALD Learner Wrapper

```python
# active_learning/src/gbald_learner.py
"""
G-BALD Active Learner - wraps GeometricBALD with optimization loop.
"""

import torch
from typing import Optional

from active_learning.src.gbald import GeometricBALD
from active_learning.src.config import DEVICE


class GBALDLearner:
    """
    Active learner using G-BALD for test selection.

    Combines:
    - GeometricBALD for acquisition scoring
    - Gradient-based optimization to find best test point
    - SVGD/VI posterior updates (delegated to parent learner)
    """

    def __init__(self, decoder, posterior, bounds: torch.Tensor, config: dict):
        """
        Args:
            decoder: LevelSetDecoder
            posterior: SVGDPosterior or LatentUserDistribution
            bounds: Test space bounds, shape (n_joints, 2)
            config: Configuration dict
        """
        self.decoder = decoder
        self.posterior = posterior
        self.bounds = bounds
        self.config = config

        # Initialize G-BALD
        self.gbald = GeometricBALD(decoder, posterior, config)

        # Optimization config
        opt_config = config.get('gbald', {}).get('optimization', {})
        self.n_restarts = opt_config.get('n_restarts', 10)
        self.n_steps = opt_config.get('n_steps', 50)
        self.lr = opt_config.get('lr', 0.1)
        self.n_candidates = opt_config.get('n_candidates', 1000)

    def select_test_point(self, iteration: int = 0) -> torch.Tensor:
        """
        Select next test point by maximizing G-BALD score.

        Uses multi-start gradient ascent with random initialization.

        Returns:
            best_point: Selected test point, shape (n_joints,)
        """
        device = self.bounds.device
        n_joints = self.bounds.shape[0]

        # Sample from posterior once for all optimizations
        if hasattr(self.posterior, 'get_particles'):
            zs = self.posterior.get_particles()
        else:
            zs = self.posterior.sample(self.gbald.n_samples)

        best_score = -float('inf')
        best_point = None

        for restart in range(self.n_restarts):
            # Initialize from random point in bounds
            x = self.bounds[:, 0] + torch.rand(n_joints, device=device) * (
                self.bounds[:, 1] - self.bounds[:, 0]
            )
            x = x.requires_grad_(True)

            optimizer = torch.optim.Adam([x], lr=self.lr)

            for step in range(self.n_steps):
                optimizer.zero_grad()

                # Compute G-BALD score
                score = self.gbald.compute_score(
                    x.unsqueeze(0), zs=zs, iteration=iteration
                ).squeeze()

                # Maximize score (minimize negative)
                loss = -score
                loss.backward()
                optimizer.step()

                # Project back to bounds
                with torch.no_grad():
                    x.clamp_(self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate final score
            with torch.no_grad():
                final_score = self.gbald.compute_score(
                    x.unsqueeze(0), zs=zs, iteration=iteration
                ).squeeze().item()

            if final_score > best_score:
                best_score = final_score
                best_point = x.detach().clone()

        return best_point

    def update(self, query_point: torch.Tensor, outcome: float):
        """Update G-BALD state after query."""
        self.gbald.update(query_point, outcome)
```

#### Step 4: Register in Factory

```python
# In active_learning/src/factory.py

# Add import
from active_learning.src.gbald import GeometricBALD

# In build_learner function, add case for 'gbald':
elif strategy == 'gbald':
    # G-BALD: Geometric BALD with ellipsoid diversity
    from active_learning.src.gbald import GeometricBALD

    bald_calculator = GeometricBALD(decoder, posterior, config)

    learner = LatentActiveLearner(
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        vi=vi,
        bald_calculator=bald_calculator,
        bounds=bounds,
        config=config,
        embeddings=embeddings
    )
```

#### Step 5: Update LatentActiveLearner to Handle G-BALD Updates

The existing `LatentActiveLearner.step()` method needs to call `bald_calculator.update()` if the calculator has an update method:

```python
# In LatentActiveLearner.step(), after getting outcome:

# Update G-BALD state (if applicable)
if hasattr(self.bald_calculator, 'update'):
    self.bald_calculator.update(test_point, outcome)
```

---

## Part 4: Configuration

### 4.1 Config Schema

Add to `configs/latent.yaml`:

```yaml
gbald:
  # Temperature for feasibility sigmoid
  tau: 0.1

  # Number of posterior samples for BALD computation
  n_samples: 32

  # Boundary penalty sharpness (higher = sharper cutoff)
  eta: 2.0

  # Weight for diversity term (0 = pure BALD)
  diversity_weight: 1.0

  # Weight for boundary penalty (0 = no penalty)
  boundary_weight: 1.0

  # Use log-scale combination (more stable)
  use_log_scale: true

  # Optimization settings
  optimization:
    n_restarts: 10
    n_steps: 50
    lr: 0.1
```

### 4.2 Hyperparameter Guidance

| Parameter | Range | Effect |
|-----------|-------|--------|
| `eta` | 0.5 - 5.0 | Higher = stricter boundary avoidance |
| `diversity_weight` | 0.5 - 2.0 | Higher = more emphasis on coverage |
| `boundary_weight` | 0.5 - 2.0 | Higher = more penalty for outliers |
| `tau` | 0.05 - 0.2 | Lower = sharper feasibility decisions |

---

## Part 5: Testing Plan

### 5.1 Unit Tests

```python
# tests/test_gbald.py

def test_ellipsoid_update():
    """Test that ellipsoid statistics update correctly."""
    ellipsoid = AdaptiveEllipsoid(dim=4, device='cpu')

    # Add points
    for _ in range(10):
        point = torch.randn(4)
        ellipsoid.update(point)

    assert ellipsoid.n == 10
    assert ellipsoid.is_ellipsoid_valid
    assert ellipsoid.get_precision_matrix() is not None

def test_gbald_score_shape():
    """Test G-BALD output shape."""
    # ... setup ...
    scores = gbald.compute_score(test_points)
    assert scores.shape == (len(test_points),)

def test_diversity_increases_with_distance():
    """Test that diversity score increases with distance from history."""
    # Add a point to history
    gbald.update(torch.zeros(4), 1.0)

    # Score nearby point
    near = torch.tensor([0.1, 0.1, 0.1, 0.1])
    # Score far point
    far = torch.tensor([1.0, 1.0, 1.0, 1.0])

    div_near = gbald._compute_diversity(near.unsqueeze(0))
    div_far = gbald._compute_diversity(far.unsqueeze(0))

    assert div_far > div_near
```

### 5.2 Integration Test

Run diagnostic comparison:
```bash
python active_learning/test/latent/compare_all.py \
    --strategies bald gbald \
    --trials 5 \
    --budget 50 \
    --metrics iou f1 boundary_accuracy
```

### 5.3 Expected Results

- G-BALD should achieve same IoU as BALD with fewer queries
- Query points should be more spatially distributed
- Boundary accuracy should improve (more uniform boundary coverage)

---

## Part 6: Summary

### Implementation Checklist

- [ ] Create `ellipsoid.py` with `AdaptiveEllipsoid` class
- [ ] Create `gbald.py` with `GeometricBALD` class
- [ ] Register `gbald` strategy in `factory.py`
- [ ] Update `LatentActiveLearner` to call `bald_calculator.update()`
- [ ] Add `gbald` section to `latent.yaml`
- [ ] Write unit tests
- [ ] Run comparison experiments

### Key Equations Reference

```
BALD(x) = H(p̄) - (1/K) Σₖ H(pₖ)

diversity(x) = min_{s ∈ S} √[(x - s)ᵀ A (x - s)]

boundary(x) = σ(-η · [(x - μ)ᵀ A (x - μ) - 1])

G-BALD(x) = BALD(x) · diversity(x)^λ_d · boundary(x)^λ_b
```

### Why This Will Work

1. **Uncertainty** (BALD): Targets informative points where particles disagree
2. **Diversity** (Core-set): Ensures queries spread across the feasible region
3. **Boundary penalty**: Keeps queries within the "typical" region, avoiding outliers
4. **Adaptive geometry**: Ellipsoid shape learned from data, not assumed spherical
