# Active Learning Pipeline: Architecture & Implementation Guide

**Version:** 3.0
**Last Updated:** January 28, 2026
**Purpose:** Comprehensive reference for understanding the entire active learning codebase

---

## Table of Contents

1. [Overview](#1-overview)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Codebase Architecture](#3-codebase-architecture)
4. [Core Components](#4-core-components)
5. [Acquisition Strategies](#5-acquisition-strategies)
6. [Variational Inference Methods](#6-variational-inference-methods)
7. [Unified Pipeline](#7-unified-pipeline)
8. [Configuration System](#8-configuration-system)
9. [Metrics & Diagnostics](#9-metrics--diagnostics)
10. [API Reference](#10-api-reference)
11. [File Dependencies](#11-file-dependencies)
12. [Usage Examples](#12-usage-examples)
13. [Extending the Codebase](#13-extending-the-codebase)
14. [Test Suite](#14-test-suite)
15. [Performance Considerations](#15-performance-considerations)
16. [Common Pitfalls & Debugging](#16-common-pitfalls--debugging)

---

## 1. Overview

### 1.1 What This Codebase Does

This active learning system **adaptively discovers a user's kinematic feasibility boundary** (joint range-of-motion limits) by intelligently selecting test queries to minimize the number of oracle queries needed.

**Problem:** Given a disabled user with unknown joint constraints, determine which poses are feasible/infeasible.

**Solution:** Use Bayesian active learning in a learned latent space to efficiently explore the feasibility boundary.

### 1.2 Key Innovation: Latent Space Learning

Instead of modeling joint limits directly in parameter space (high-dimensional, constrained), we:
1. **Train a decoder** that maps low-dimensional latent codes ($z \in \mathbb{R}^{32}$) to RBF-based level-set functions
2. **Perform active learning in latent space** where the manifold of realistic constraints is better structured
3. **Use BALD** (Bayesian Active Learning by Disagreement) to select maximally informative test points

### 1.3 Unified Architecture with Two Orthogonal Axes

All strategies are handled by a **single learner class** (`LatentActiveLearner`) with swappable components injected by the factory. The system has **two orthogonal configuration axes**:

**Axis 1: Test Selection Strategy** (`acquisition.strategy`)
- Determines how to select informative test points
- Options: `bald`, `random`, `quasi_random`, `prior_boundary`, `multi_stage_warmup` (NEW), `canonical`, `gp`, `grid`, `heuristic`, `version_space`

**Axis 2: Posterior Inference Method** (`posterior.method`)
- Determines how to represent and update beliefs
- Options: `vi` (single Gaussian), `full_cov_vi` (full covariance), `ensemble` (K Gaussians), `svgd` (K particles), `sliced_svgd` (1D projections), `projected_svgd` (NEW - eigenvector-informed)

**Key Innovation:** Any strategy can be combined with any posterior method. BALD auto-detects the posterior type and uses the appropriate variant (LatentBALD, EnsembleBALD, or ParticleBALD).

| Posterior Method | Representation | VI Method | BALD Variant | Key Files |
|------------------|----------------|-----------|--------------|-----------|
| **`vi`** (default) | Single Gaussian ($\mu, \sigma$) | `LatentVariationalInference` | `LatentBALD` | `latent_bald.py`, `latent_variational_inference.py` |
| **`full_cov_vi`** | Full covariance Gaussian | `LatentVariationalInference` (full cov) | `LatentBALD` | `latent_variational_inference.py` |
| **`ensemble`** | K independent Gaussians | K × `LatentVariationalInference` | `EnsembleBALD` | `ensemble/ensemble_bald.py` |
| **`svgd`** | K interacting particles | `SVGDVariationalInference` | `ParticleBALD` | `svgd/svgd_vi.py`, `svgd/svgd_optimizer.py` |
| **`sliced_svgd`** | K particles (1D projections) | `SlicedSVGDVariationalInference` | `ParticleBALD` | `svgd/sliced_svgd_vi.py`, `svgd/sliced_svgd_optimizer.py` |
| **`projected_svgd`** (NEW) | K particles (eigenvector-informed) | `ProjectedSVGDVariationalInference` | `ParticleBALD` | `svgd/projected_svgd_vi.py`, `svgd/projected_svgd_optimizer.py` |

All configurations are assembled inside `factory.py` and returned as a single `LatentActiveLearner` instance.

---

## 2. Theoretical Foundation

### 2.1 Bayesian Active Learning

**Goal:** Minimize expected posterior uncertainty by selecting optimal queries.

**Framework:**
```
Posterior:  p(z|D) ~ p(D|z) * p(z)
Query:      x* = argmax_{x} I(z; y|x, D)
Update:     D <- D U {(x*, y*)}
```

Where:
- $z$ = latent code
- $D$ = test history (queries and outcomes)
- $I(z; y|x, D)$ = mutual information (BALD score)

### 2.2 BALD (Bayesian Active Learning by Disagreement)

**Formula:**
```
BALD(x) = H(E_{z}[p(y|x,z)]) - E_{z}[H(p(y|x,z))]
        = Entropy of mean prediction - Mean entropy of predictions
        = Information gain about parameters
```

**Interpretation:** Points with high BALD are:
- Uncertain in expectation (high $H(\bar{p})$)
- But certain for individual parameter samples (low $E[H(p)]$)
- Thus maximally informative about parameter disagreement

### 2.3 Latent Space Representation

**Level-Set Function:**
```
f(q; z) = d_box(q; lower(z), upper(z), weights(z))
          - Sum_k presence_k(z) * RBF_k(q; blob_params_k(z))
```

Where:
- $d_{box}$ = distance to box boundaries (joint limits)
- $\text{RBF}_k$ = Gaussian "bump" penalties (injury constraints)
- Decoder: $z \rightarrow (lower, upper, weights, presence, blob\_params)$

**Feasibility:** $q$ is feasible if $f(q; z) \geq 0$

### 2.4 Variational Inference Objectives

#### Standard VI (Gaussian Posterior)
**Objective:** Maximize ELBO
```
ELBO(mu, sigma) = E_{z~N(mu,sigma)}[log p(D|z)] - KL(N(mu,sigma) || N(mu_prior, sigma_prior))
               = Likelihood - Regularization
```

**Update:** Adam on $(mu, \log sigma)$

#### SVGD (Particle Posterior)
**Objective:** Minimize KL divergence via Stein operator
```
phi*(x) = E_{x'~q}[k(x',x) grad_log p(x'|D) + grad_{x'} k(x',x)]
```

Where:
- $k(x',x)$ = RBF kernel (similarity)
- $\nabla \log p(x'|D)$ = score function (attraction to high prob)
- $\nabla_{x'} k(x',x)$ = kernel gradient (repulsion for diversity)

**Update:** $x_i \leftarrow x_i + \epsilon \phi^*(x_i)$

#### Ensemble (Multiple Independent Gaussians)
**Objective:** K independent ELBO optimizations
```
For each member k: maximize ELBO_k(mu_k, sigma_k)
```

**Uncertainty:** Disagreement between ensemble members

---

## 3. Codebase Architecture

### 3.1 Directory Structure

```
active_learning/
|-- src/
|   |-- Core Components (shared):
|   |   |-- config.py                      # Configuration loader, DEVICE, bounds
|   |   |-- utils.py                       # binary_entropy, KL weight, adaptive params
|   |   |-- latent_user_distribution.py    # Gaussian latent distribution
|   |   |-- latent_feasibility_checker.py  # Level-set evaluator
|   |   |-- latent_oracle.py               # Ground truth queries
|   |   |-- latent_prior_generation.py     # Prior initialization
|   |   |-- test_history.py                # Query history storage
|   |   |-- diagnostics.py                 # Iteration diagnostics
|   |   +-- metrics.py                     # IoU/Accuracy/F1 metrics
|   |
|   |-- Active Learning Pipeline:
|   |   |-- latent_active_learning.py      # UNIFIED learner (all strategies)
|   |   |-- latent_bald.py                 # BALD acquisition + test optimization
|   |   +-- latent_variational_inference.py # Standard VI optimizer
|   |
|   |-- Ensemble Components:
|   |   +-- ensemble/
|   |       |-- __init__.py
|   |       +-- ensemble_bald.py           # K-member BALD acquisition
|   |
|   |-- SVGD Components:
|   |   +-- svgd/
|   |       |-- particle_user_distribution.py  # Particle posterior
|   |       |-- particle_bald.py               # Particle BALD acquisition
|   |       |-- svgd_vi.py                     # SVGD VI optimizer
|   |       +-- svgd_optimizer.py              # Stein update kernel
|   |
|   |-- Baselines:
|   |   +-- baselines/
|   |       |-- random_strategy.py         # Uniform random sampling
|   |       |-- quasi_random_strategy.py   # Sobol sequences + BALD
|   |       |-- gp_strategy.py             # Gaussian Process + Straddle
|   |       |-- grid_strategy.py           # Grid-based exhaustive search
|   |       |-- heuristic_strategy.py      # Dense banking (candidate evaluation)
|   |       +-- version_space_strategy.py  # Greedy maximin hypothesis pruning
|   |
|   |-- Hybrid Strategies:
|   |   +-- canonical_acquisition.py       # Canonical queries + BALD
|   |
|   +-- Factory:
|       +-- factory.py                     # SINGLE entry point: build_learner()
|
|-- configs/
|   +-- latent.yaml                        # Configuration file
|
|-- test/
|   |-- test_refactored_pipeline.py        # Comprehensive pytest suite (110 tests)
|   |-- test_latent_active_learning.py     # Additional integration tests
|   |-- latent/                            # Comparison scripts
|   +-- diagnostics/
|       +-- run_latent_diagnosis.py        # Diagnostic runner
|
+-- docs/
    |-- README.md                          # Documentation index
    |-- ARCHITECTURE.md                    # This file
    |-- QUICK_START.md                     # Quick onboarding
    +-- VISUAL_REFERENCE.md               # Visual diagrams
```

### 3.2 Design Patterns

#### Factory Pattern (Central)
**Purpose:** Construct the unified learner with appropriate components based on two orthogonal configuration axes.

```python
# factory.py -- Clean separation of concerns via helper functions
def build_learner(decoder, prior, posterior, oracle, bounds, config, embeddings=None):
    """
    Two-axis factory:
      1. config['posterior']['method'] → posterior representation
      2. config['acquisition']['strategy'] → test selection method

    Any strategy works with any posterior method.
    """
    # 1. Build posterior based on method (vi, ensemble, svgd)
    posterior = _build_posterior(decoder, prior, posterior, config)
    #   - vi: pass-through single Gaussian
    #   - ensemble: create K perturbed Gaussians
    #   - svgd: create ParticleUserDistribution

    # 2. Build strategy (auto-detects BALD variant for bald/quasi_random/canonical)
    strategy = _build_strategy(decoder, prior, posterior, config, bounds, oracle, embeddings)
    #   - Calls _build_bald() which auto-detects posterior type:
    #     * list → EnsembleBALD
    #     * has get_particles → ParticleBALD
    #     * else → LatentBALD

    # 3. Build VI (auto-detects based on posterior method)
    vi = _build_vi(decoder, prior, posterior, config)
    #   - ensemble: list of K LatentVariationalInference
    #   - svgd: SVGDVariationalInference
    #   - vi: single LatentVariationalInference

    # 4. Construct learner (ALWAYS returns LatentActiveLearner)
    return LatentActiveLearner(decoder, prior, posterior, oracle, bounds, config,
                                acquisition_strategy=strategy, vi=vi)
```

**Key insights:**
1. The factory cleanly separates the two axes via helper functions
2. BALD auto-detection allows any strategy to work with any posterior method
3. Callers always get back a `LatentActiveLearner` with a uniform API

#### Strategy Pattern
**Purpose:** Plug different acquisition functions into the learner.

All strategies implement:
```python
select_test(bounds, test_history, iteration, diagnostics, ...) -> (test_point, score[, stats])
```

Some strategies additionally implement:
```python
post_query_update(test_point, outcome, history)  # For stateful strategies (GP, VersionSpace)
```

#### Composition over Inheritance
The learner uses **composition** to support all three posterior modes:
- `self.posterior` is either a `LatentUserDistribution` or `List[LatentUserDistribution]`
- `self.vi` is either a single VI or `List[VI]`
- `self._is_ensemble = isinstance(self.posterior, list)` determines ensemble behavior

---

## 4. Core Components

### 4.1 LatentUserDistribution

**File:** `latent_user_distribution.py`
**Purpose:** Represents a Gaussian distribution in latent space

**Interface:**
```python
class LatentUserDistribution:
    def __init__(latent_dim, decoder, mean, log_std, device)
    def sample(n_samples, temperature=1.0, generator=None) -> (N, D)

    # Properties
    @property mean -> (D,)
    @property log_std -> (D,)
```

**Reparameterization Trick:**
```python
std = exp(log_std) * temperature
eps = randn(n_samples, latent_dim)
samples = mean + std * eps
```

**Variants:**
- **Standard:** Optimized mean/log_std parameters
- **Ensemble:** K independent instances with perturbed means
- **Particle:** Non-parametric (see `ParticleUserDistribution`)
- **Full Covariance (VI):** Supports full covariance matrix via Cholesky factor (`cov_cholesky`)

### 4.2 LatentFeasibilityChecker

**File:** `latent_feasibility_checker.py`
**Purpose:** Evaluates feasibility of test points given latent code

**Key Methods:**
```python
@staticmethod
decode_latent_params(decoder, zs: (B, D)) -> (lower, upper, weights, pres, blobs)

@staticmethod
evaluate_from_decoded(test_points: (N, J), decoded_params) -> logits: (B, N)

@staticmethod
batched_logit_values(decoder, zs: (B, D), test_points: (N, J)) -> (B, N)
```

**Usage Pattern:**
```python
# Pre-decode once (efficient for BALD optimization)
decoded = LatentFeasibilityChecker.decode_latent_params(decoder, zs)

# Evaluate many test points against same decoded params
for test_point in candidates:
    logit = LatentFeasibilityChecker.evaluate_from_decoded(test_point, decoded)
```

### 4.3 LatentOracle

**File:** `latent_oracle.py`
**Purpose:** Provides ground truth feasibility queries

**Interface:**
```python
class LatentOracle:
    def __init__(decoder, ground_truth_z, n_joints)
    def query(test_point: (J,)) -> float  # Signed distance (logit)
    def get_history() -> TestHistory
    def reset()
```

**Ground Truth:** Uses a fixed latent code $z_{GT}$ to evaluate feasibility.

### 4.4 TestHistory

**File:** `test_history.py`
**Purpose:** Stores query history for VI updates

**Interface:**
```python
@dataclass
class TestResult:
    test_point: Tensor      # (J,)
    outcome: float          # Signed distance
    timestamp: int          # Query index
    h_value: Optional[float]
    metadata: Optional[dict]

class TestHistory:
    def add(test_point, outcome, ...) -> TestResult
    def get_all() -> List[TestResult]
```

### 4.5 Utility Functions

**File:** `utils.py`
**Purpose:** Shared mathematical utilities

```python
def binary_entropy(p: Tensor) -> Tensor
    """H(p) = -p*log(p) - (1-p)*log(1-p), with eps clamping."""

def calculate_kl_weight(iteration: int, annealing_config: dict, default_weight: float) -> float
    """KL weight from annealing schedule (linear, cosine, logistic, step)."""

def get_adaptive_param(schedule: dict, iteration: int, default: float) -> float
    """Interpolate parameter value from schedule config."""

class AcquisitionStrategy:
    """Protocol class defining the strategy interface."""
    def select_test(self, bounds, **kwargs): ...
    def post_query_update(self, test_point, outcome, history): ...  # default no-op
```

### 4.6 Configuration System

**File:** `config.py`
**Config File:** `configs/latent.yaml`

**Key Functions:**
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

load_config(config_path: str) -> dict
get_bounds_from_config(config: dict, device) -> Tensor (J, 2)
create_generator(config: dict, device) -> Generator or None
```

---

## 5. Acquisition Strategies

The codebase supports 10 acquisition strategies, all routed through the factory.

### 5.1 LatentBALD (Standard)

**File:** `latent_bald.py`
**Theory:** Selects points maximizing information gain about latent code

**Score Computation:**
```python
# Sample from posterior
zs = posterior.sample(n_samples)

# Compute predictions
logits = evaluate_level_set(zs, test) / tau
probs = sigmoid(logits)

# BALD = H(mean) - mean(H)
p_mean = probs.mean(dim=0)
entropy_of_mean = H(p_mean)
mean_of_entropies = H(probs).mean()
bald_score = entropy_of_mean - mean_of_entropies
```

**Test Optimization:** Multi-restart gradient ascent with Adam -> SGD switching, pre-decoded RBF parameters, optional diversity bonus.

**Adaptive Schedules:** `tau_schedule` (sigmoid temperature), `weighted_bald_sigma_schedule` (gate width).

### 5.2 EnsembleBALD

**File:** `ensemble/ensemble_bald.py`
**Theory:** Measures disagreement between K independent posteriors

```python
# Sample from each ensemble member
for posterior_k in posteriors:
    zs_k = posterior_k.sample(n_samples_per_member)
    p_k = mean(sigmoid(evaluate(zs_k, test) / tau))

# Ensemble BALD = disagreement
p_bar = mean(member_probs)
score = H(p_bar) - mean([H(p_k) for p_k in member_probs])
```

### 5.3 ParticleBALD

**File:** `svgd/particle_bald.py`
**Theory:** Uses all SVGD particles directly (no sampling)

```python
class ParticleBALD(LatentBALD):
    def compute_score(test, zs=None, ...):
        if zs is None:
            zs = self.posterior.get_particles()  # All K particles
        return super().compute_score(test, zs=zs, ...)
```

### 5.4 Random Baseline

**File:** `baselines/random_strategy.py`

Uniform random sampling within bounds.

### 5.5 Quasi-Random Baseline

**File:** `baselines/quasi_random_strategy.py`

Phase 1 (first N queries): Sobol low-discrepancy sequence. Phase 2: Delegates to BALD.

### 5.6 CanonicalAcquisition (Hybrid)

**File:** `canonical_acquisition.py`

Phase 1 (first N queries): Pre-computed canonical queries from `.npz` file. Phase 2: Delegates to BALD.

### 5.7 GP Strategy (Gaussian Process)

**File:** `baselines/gp_strategy.py`
**Theory:** Fits a GP to observed feasibility, selects via Straddle heuristic.

**Stateful:** Implements `post_query_update()` to add observations to GP model. Requires `scikit-learn`.

### 5.8 Grid Strategy

**File:** `baselines/grid_strategy.py`

Exhaustive grid search over joint space. Evaluates all grid points and selects the most uncertain.

### 5.9 Heuristic Strategy

**File:** `baselines/heuristic_strategy.py`

Dense banking: evaluates all candidate embeddings against a query grid, selects the candidate that best halves the query set. Requires embeddings parameter in factory.

### 5.10 Version Space Strategy

**File:** `baselines/version_space_strategy.py`

Greedy maximin hypothesis pruning. Maintains a valid hypothesis mask and selects queries to maximally prune the version space.

**Stateful:** Implements `post_query_update()` to prune hypotheses after each observation.

### 5.11 Prior Boundary Strategy

**File:** `baselines/prior_boundary_strategy.py`

**Theory:** Uses a two-phase approach to efficiently find the feasibility boundary.

1.  **Warmup Phase:** Selects points that lie on the decision boundary of the *prior mean* (where $p(feasible \mid z_{mean}) \approx 0.5$). Uses farthest-point sampling to cover the boundary evenly.
2.  **BALD Phase:** Once the warmup budget is exhausted, switches to standard BALD acquisition.

**Use Case:** Good for "jump-starting" the search when the prior is reasonably good but the boundary needs refinement.

### 5.12 Multi-Stage Warmup Strategy (NEW)

**File:** `baselines/multi_stage_warmup_strategy.py`

**Theory:** An advanced adaptive warmup strategy that iteratively queries the **current posterior's** decision boundary, not just the prior's.

**Algorithm:**
```
For each stage (up to n_stages):
    1. Sample N_candidates uniform in anatomical bounds
    2. Evaluate p(feasible) by averaging over posterior samples
    3. Select candidates where p_mean ≈ 0.5 (decision boundary)
    4. Use farthest-point sampling for spatial diversity
    5. Query stage_budget points from this stage's boundary

    After each query:
    6. Track outcomes in rolling window
    7. Compute rolling entropy: H = -p*log(p) - (1-p)*log(1-p)
    8. If H >= entropy_threshold (adaptive stopping):
       Switch to final_phase (bald, posterior_boundary, or stop)

    If max_warmup reached without adaptive stop:
       Switch to final_phase
```

**Key Differences from Prior Boundary:**
- **Posterior-adaptive:** Uses current posterior, not prior mean
- **Multi-stage:** Recomputes boundary at each stage as posterior updates
- **Entropy-based stopping:** Automatically detects when boundary is well-discovered
- **Configurable final phase:** Can continue with BALD, posterior boundary, or stop

**Configuration:**
```yaml
multi_stage_warmup:
  n_stages: 5                    # Max stages (acts as cap if adaptive_stopping=true)
  queries_per_stage: 5           # Queries per stage
  n_candidates: 5000             # Candidate pool for boundary search
  boundary_percentile: 0.05      # Top 5% closest to p=0.5
  use_farthest_point: true       # GPU-accelerated diversity sampling
  final_phase: bald              # 'bald', 'posterior_boundary', or 'stop'
  adaptive_stopping: true        # Enable entropy-based stopping
  entropy_threshold: 0.9         # H=0.9 ≈ ratio in [0.35, 0.65]
  window_size: 15                # Rolling window for entropy
  min_warmup_queries: 10         # Minimum before checking entropy
```

**Use Case:** When you want efficient exploration that adapts to the posterior's uncertainty, then refines with BALD.

---

## 6. Variational Inference Methods

### 6.1 Standard VI (Gaussian)

**File:** `latent_variational_inference.py`

**Algorithm:**
```
For iter in 1..max_iters:
    1. Sample: z ~ N(mu, exp(2*log_std))
    2. Likelihood: LL = Sum log p(y_i | z, x_i)
    3. KL: KL = KL(N(mu,sigma) || N(mu_prior, sigma_prior))
    4. ELBO = LL - w*KL
    5. Backprop: -ELBO
    6. Adam step on (mu, log_std)
    7. Clamp: log_std >= log(min_std)
    8. Early stop if no improvement for patience iterations
```

**KL Annealing:**
```python
kl_weight(t) = start + (end - start) * schedule(t)
# Schedules: linear, cosine, logistic, step
```

### 6.2 SVGD (Particle)

**File:** `svgd/svgd_vi.py`, `svgd/svgd_optimizer.py`

**Algorithm:**
```
For iter in 1..max_iters:
    1. Detach particles and enable gradients
    2. Compute log p(D|z) for each particle
    3. Compute log p(z) for each particle
    4. log_joint = log_likelihood + log_prior
    5. Backward: get grad_log p(z|D)
    6. SVGD step: phi = SVGD_update(particles, gradients)
    7. Update: particles += lr * phi
```

**SVGD Update (Stein Forces):**
```python
# Kernel matrix (RBF)
h = median(pairwise_distances) / log(K+1)
k(x_i, x_j) = exp(-||x_i - x_j||^2 / h)

# Attraction to high probability + Repulsion for diversity
phi(x_i) = (1/K) * Sum_j [k(x_j,x_i) * grad_log_p(x_j)] + (1/K) * Sum_j [grad_{x_j} k(x_j,x_i)]
```

### 6.3 Full Covariance VI

**File:** `latent_variational_inference.py`

**Algorithm:**
Similar to Standard VI, but instead of optimizing a diagonal covariance (log_std), it optimizes a full lower-triangular Cholesky factor `L`.

**Closed-Form KL:**
Uses a closed-form expression for $KL(N(\mu_q, \Sigma_q) \parallel N(\mu_p, \Sigma_p))$, supporting both diagonal and full-covariance priors.

### 6.4 Sliced SVGD (NEW)

**File:** `svgd/sliced_svgd_optimizer.py`, `svgd/sliced_svgd_vi.py`

**Theory:** Projects particles onto random 1D directions and computes SVGD updates in these 1D spaces, then projects back. This reduces the computational cost of kernel computations in high dimensions.

**Algorithm:**
```
For each of n_slices directions:
    1. Sample random unit direction v ~ N(0, I), normalize
    2. Project particles: s_proj = particles @ v
    3. Project gradients: g_proj = gradients @ v
    4. Compute 1D pairwise distances and kernel
    5. Compute 1D SVGD update: term1 (attraction) + term2 (repulsion)
    6. Project back: phi += outer(phi_1d, v)

Normalize: phi /= n_slices
```

**Configuration:**
```yaml
sliced_svgd:
  step_size: 0.1
  n_slices: 20              # Number of random directions
  kernel_type: rbf          # 'rbf' or 'imq'
```

### 6.5 Projected SVGD (NEW - Recommended)

**File:** `svgd/projected_svgd_optimizer.py`, `svgd/projected_svgd_vi.py`

**Theory:** Combines eigenvector-informed direction selection with random slicing. Uses the score covariance matrix to identify directions where the posterior varies most, then focuses updates on those directions.

**Innovation:** Instead of purely random directions (Sliced SVGD), Projected SVGD:
1. Computes score covariance: C = grad.T @ grad / K
2. Extracts top eigenvectors capturing `variance_threshold` of variance
3. Uses eigenvector directions (with eigenvalue weighting) for high-variance dims
4. Uses random directions for remaining low-variance dimensions

**Algorithm:**
```
Phase 1: Score covariance eigendecomposition
    C = log_prob_grad.T @ log_prob_grad / K
    eigenvalues, eigenvectors = eigh(C)  # ascending order
    Flip to descending order

Phase 2: Adaptive r selection
    cumulative = cumsum(eigenvalues) / total_variance
    r = first index where cumulative >= variance_threshold
    r = min(r, n_slices)
    n_random = n_slices - r

Phase 3: Projected directions (r eigenvector slices)
    For each eigenvector v_k in top r:
        Compute 1D SVGD with eigenvalue weighting:
        phi_1d = (term1 + term2) * w_k
        Where w_k = eigenvalue_k^eigen_smoothing (normalized)

Phase 4: Random directions (n_random slices)
    Standard Sliced SVGD for low-variance dimensions

Phase 5: Combine and normalize
    phi /= n_slices
```

**Key Parameters:**
- `n_slices`: Total directions (eigenvector + random)
- `variance_threshold`: Fraction of variance to capture (0.95 = top 95%)
- `eigen_smoothing`: Weight exponent (0=uniform, 0.5=sqrt, 1.0=linear)
- `max_eigenweight`: Clamp to prevent needle updates from peaked spectrum
- `kernel_type`: 'rbf' (Gaussian) or 'imq' (inverse multi-quadric)

**Configuration:**
```yaml
projected_svgd:
  step_size: 0.1
  n_slices: 20
  variance_threshold: 0.95
  max_eigenweight: 3.0
  eigen_smoothing: 0.5
  kernel_type: imq
```

**Vectorized Implementation:**
The `_compute_1d_svgd_batched()` method processes all slices simultaneously for GPU efficiency. Toggle via `_USE_VECTORIZED_PSVGD = True` (default).

### 6.6 Comparison Matrix

| Feature | Standard VI | Full Cov VI | SVGD | Sliced SVGD | Projected SVGD | Ensemble |
|---------|------------|-------------|------|-------------|----------------|----------|
| **Posterior** | 1 Gaussian (diag) | 1 Gaussian (full) | K particles | K particles | K particles | K Gaussians |
| **Parameters** | 2D | D + D(D+1)/2 | K×D | K×D | K×D | K×2D |
| **Objective** | ELBO | ELBO | log p(z\|D) | log p(z\|D) | log p(z\|D) | K ELBOs |
| **Kernel** | N/A | N/A | D-dim RBF | 1D RBF | Eigenvector-weighted 1D | N/A |
| **Memory** | O(D) | O(D²) | O(K²×D) | O(K²×S) | O(K²×S) | O(K×D) |
| **Speed** | Fast | Medium | Slow (high D) | Faster | **Fastest SVGD** | K× slower |
| **Best For** | Prototyping | Correlations | Complex | High-dim | **Recommended** | Stable |

---

## 7. Unified Pipeline

### 7.1 LatentActiveLearner

**File:** `latent_active_learning.py`

This is the **single learner class** used by all strategies. It handles single-posterior, ensemble (list of posteriors), and particle-based modes via composition.

**Constructor:**
```python
class LatentActiveLearner:
    def __init__(
        self,
        decoder,
        prior: LatentUserDistribution,
        posterior,  # LatentUserDistribution OR List[LatentUserDistribution]
        oracle: LatentOracle,
        bounds: Tensor,
        config: dict,
        acquisition_strategy=None,  # Injected strategy
        vi=None  # Injected VI (single or List)
    ):
        self._is_ensemble = isinstance(self.posterior, list)
```

**Main Loop (`step()`):**
```python
def step(verbose=False) -> LatentIterationResult:
    iteration = len(self.results)

    # 1. Select test (epsilon-greedy + strategy)
    test_point, score, stats = self.strategy.select_test(
        bounds=self.bounds,
        test_history=past_tests,
        iteration=iteration,
        diagnostics=self.diagnostics
    )

    # 2. Query oracle
    outcome = self.oracle.query(test_point)

    # 2b. Strategy-specific update (GP, VersionSpace)
    if hasattr(self.strategy, 'post_query_update'):
        self.strategy.post_query_update(test_point, outcome, history)

    # 3. Update posterior via VI
    kl_weight = self._calculate_kl_weight(iteration)
    if isinstance(self.vi, list):
        # Ensemble: update each VI independently
        for vi_opt in self.vi:
            vi_opt.update_posterior(history, kl_weight=kl_weight)
    else:
        self.vi.update_posterior(history, kl_weight=kl_weight)

    # 4. Log diagnostics + store result
    ...
    return LatentIterationResult(iteration, test_point, outcome, bald_score, elbo, ...)
```

**Stopping Criteria:**
- **Budget:** Hard limit on queries
- **BALD threshold:** Stop if score < threshold for N consecutive iterations
- **ELBO plateau:** Stop if ELBO range < threshold over window
- **Uncertainty:** Stop if mean posterior std < threshold

### 7.2 How the Factory Assembles Components

```
build_learner(decoder, prior, posterior, oracle, bounds, config)
    |
    |-- strategy_type = config['acquisition']['strategy']
    |
    |-- IF 'ensemble_bald':
    |       Create K posteriors (prior.mean + noise each)
    |       Create K LatentVariationalInference optimizers
    |       Create EnsembleBALD strategy
    |       -> LatentActiveLearner(posterior=List[K], vi=List[K], strategy=EnsembleBALD)
    |
    |-- IF 'svgd':
    |       Create ParticleUserDistribution
    |       Create SVGDVariationalInference
    |       Create ParticleBALD strategy
    |       -> LatentActiveLearner(posterior=Particle, vi=SVGDVI, strategy=ParticleBALD)
    |
    |-- OTHERWISE (bald, random, quasi_random, canonical, gp, grid, heuristic, version_space):
    |       Create appropriate strategy via _build_strategy()
    |       Create LatentVariationalInference
    |       -> LatentActiveLearner(posterior=single, vi=LatentVI, strategy=...)
    |
    +-- ALWAYS returns LatentActiveLearner
```

---

## 8. Configuration System

### 8.1 Config Structure (latent.yaml)

```yaml
# BALD Acquisition
bald:
  tau: 1.0                      # Sigmoid temperature
  n_mc_samples: 100             # MC samples for BALD
  sampling_temperature: 1.0     # Posterior sampling temperature
  use_weighted_bald: false      # Enable Gaussian gate
  weighted_bald_sigma: 0.1      # Gate width
  epsilon: 0.0                  # Epsilon-greedy exploration
  epsilon_decay: 0.95
  diversity_weight: 0.0         # Sequential diversity bonus

  # Adaptive schedules
  tau_schedule:
    start: 2.0
    end: 1.0
    duration: 50
    schedule: linear            # linear, cosine, logistic, step

  weighted_bald_sigma_schedule:
    start: 0.25
    end: 0.05
    duration: 100
    schedule: linear

# BALD Test Optimization
bald_optimization:
  n_restarts: 10                # Random restarts
  n_iters_per_restart: 50       # Iterations per restart
  lr_adam: 0.05                 # Adam learning rate
  lr_sgd: 0.01                 # SGD learning rate
  switch_to_sgd_at: 0.8        # Fraction before switch

# Variational Inference
vi:
  n_mc_samples: 100             # MC samples for ELBO
  learning_rate: 0.058          # Adam learning rate
  max_iters: 500                # Max VI iterations
  convergence_tol: 0.0005       # ELBO improvement threshold
  patience: 10                  # Early stopping patience
  grad_clip: 1.0                # Gradient clipping
  noise_std: 1.0                # Gaussian likelihood noise std

  # KL Annealing
  kl_annealing:
    enabled: true
    start_weight: 0.286
    end_weight: 0.286
    duration: 10
    schedule: linear

# Acquisition Strategy
acquisition:
  strategy: bald                # bald, ensemble_bald, svgd, random,
                                # quasi_random, canonical, gp, grid,
                                # heuristic, version_space
  n_canonical: 5
  n_quasi_random: 10
  grid_strategy_resolution: 16

# Stopping Criteria
stopping:
  budget: 20
  bald_enabled: false
  bald_threshold: 0.1
  bald_patience: 3
  elbo_plateau_enabled: false
  uncertainty_enabled: false

# Prior Configuration
prior:
  mean_noise_std: 0.3
  init_std: 1.8
  use_anatomical_limit_prior: true
  units: degrees
  joint_names:
    - shoulder_flexion_r
    - shoulder_abduction_r
    - shoulder_rotation_r
    - elbow_flexion_r
  anatomical_limits:
    shoulder_flexion_r: [-15, 195]
    shoulder_abduction_r: [-175, 145]
    shoulder_rotation_r: [-145, 200]
    elbow_flexion_r: [-30, 175]

# Latent Space
latent:
  latent_dim: 32
  model_path: ../models/best_model.pt
  dataset_path: ../models/training_data.npz

# Ensemble (for ensemble_bald)
ensemble:
  ensemble_size: 5
  init_noise_std: 0.4

# SVGD (for svgd/sliced_svgd/projected_svgd)
posterior:
  method: svgd           # Options: vi, full_cov_vi, ensemble, svgd, sliced_svgd, projected_svgd
  n_particles: 50

svgd:
  step_size: 0.1
  max_iters: 100

# Sliced SVGD (NEW)
sliced_svgd:
  step_size: 0.1
  n_slices: 20
  kernel_type: rbf       # 'rbf' or 'imq'

# Projected SVGD (NEW - Recommended)
projected_svgd:
  step_size: 0.1
  n_slices: 20                     # Total directions (eigenvector + random)
  variance_threshold: 0.95         # Capture 95% score variance with eigenvectors
  max_eigenweight: 3.0             # Clamp to prevent needle updates
  eigen_smoothing: 0.5             # 0=uniform, 0.5=sqrt, 1.0=linear
  kernel_type: imq                 # 'rbf' or 'imq' (inverse multi-quadric)

# Multi-Stage Warmup (NEW)
multi_stage_warmup:
  n_stages: 5                      # Max warmup stages
  queries_per_stage: 5             # Queries per boundary stage
  n_candidates: 5000               # Candidate pool for boundary search
  boundary_percentile: 0.05        # Top 5% closest to p=0.5
  use_farthest_point: true         # GPU-accelerated diversity sampling
  final_phase: bald                # 'bald', 'posterior_boundary', or 'stop'
  adaptive_stopping: true          # Enable entropy-based stopping
  entropy_threshold: 0.9           # H=0.9 ≈ ratio in [0.35, 0.65]
  window_size: 15                  # Rolling window for entropy
  min_warmup_queries: 10           # Min queries before checking entropy

seed: null
```

### 8.2 Config Access Patterns

```python
# Load config
config = load_config("configs/latent.yaml")

# Access nested values with defaults
bald_cfg = config.get('bald', {})
tau = bald_cfg.get('tau', 1.0)

# Get bounds (converts degrees to radians automatically)
bounds = get_bounds_from_config(config, device)

# Create seeded generator
generator = create_generator(config, device)  # None if seed=null
```

---

## 9. Metrics & Diagnostics

### 9.1 Reachability Metrics

**File:** `metrics.py`

```python
compute_reachability_metrics(
    decoder, ground_truth_params, posterior_mean, test_grid
) -> (iou, accuracy, f1)
```

### 9.2 Diagnostic Tracking

**File:** `diagnostics.py`

**DiagnosticSnapshot (per iteration):**
```python
@dataclass
class DiagnosticSnapshot:
    iteration: int
    prior_z_coverage: float            # % GT dims within 2 sigma of prior
    posterior_z_coverage: float        # % GT dims within 2 sigma of posterior
    query_distance_to_boundary: float  # |h(query)|
    query_is_near_boundary: bool       # distance < 0.1
    grad_norm: float
    posterior_movement: float          # ||mu_post - mu_prev||
    latent_std_mean: float
    vi_converged: bool
    vi_iterations: int
    elbo_history: List[float]
    likelihood: float
    kl_divergence: float
    bald_opt_stats: List[Dict]         # Per-restart diagnostics
```

**Usage:**
```python
diagnostics = Diagnostics(true_z=ground_truth_z)
diagnostics.log_iteration(iteration=i, prior=prior, posterior=posterior, ...)
diagnostics.print_final_report()
```

---

## 10. API Reference

### 10.1 Core API: Active Learner

```python
class LatentActiveLearner:
    def step(verbose: bool = False) -> LatentIterationResult
    def run(n_iterations: int, verbose: bool = True) -> List[LatentIterationResult]
    def check_stopping_criteria() -> (bool, str)
    def get_posterior() -> LatentUserDistribution
    def get_history() -> TestHistory
    def ensemble_predict_probs(test_points) -> Tensor  # Only for ensemble mode
```

### 10.2 Acquisition Strategy API

All strategies implement:
```python
def select_test(
    bounds: Tensor,             # (J, 2)
    verbose: bool = False,
    test_history: List = None,
    iteration: int = None,
    diagnostics = None,
    **kwargs
) -> Union[
    Tuple[Tensor, float],              # (test_point, score)
    Tuple[Tensor, float, List[Dict]]   # (test_point, score, stats)
]
```

Stateful strategies additionally implement:
```python
def post_query_update(self, test_point, outcome, history):
    """Called after oracle query to update internal state."""
```

### 10.3 Variational Inference API

```python
class VariationalInference:
    def likelihood(test_history: TestHistory) -> Tensor
    def regularizer(kl_weight: float = None) -> Tensor
    def update_posterior(
        test_history: TestHistory,
        kl_weight: float = None,
        diagnostics = None,
        iteration: int = None
    ) -> VIResult
```

### 10.4 Factory API

```python
def build_learner(
    decoder,
    prior: LatentUserDistribution,
    posterior: LatentUserDistribution,
    oracle: LatentOracle,
    bounds: Tensor,
    config: dict,
    embeddings: Tensor = None  # Required for heuristic, version_space
) -> LatentActiveLearner  # ALWAYS returns LatentActiveLearner
```

---

## 11. File Dependencies

### 11.1 Dependency Graph

```
+-------------------------------------------------------------+
|                    Application Layer                          |
+-------------------------------------------------------------+
|  run_latent_diagnosis.py                                     |
|    | imports                                                 |
|  factory.py                                                  |
|    | constructs                                              |
|  LatentActiveLearner (with injected components)              |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                    Learner Layer                              |
+-------------------------------------------------------------+
|  LatentActiveLearner (SINGLE unified learner)                |
|    |-- strategy: (any of the 10 strategies)                  |
|    |-- vi: LatentVI | List[LatentVI] | SVGDVI               |
|    |-- posterior: LatentUserDist | List | Particle            |
|    |-- oracle: LatentOracle                                  |
|    |-- diagnostics: Diagnostics                              |
|    +-- config: dict                                          |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                    Strategy Layer                             |
+-------------------------------------------------------------+
|  LatentBALD, EnsembleBALD, ParticleBALD                     |
|  RandomStrategy, QuasiRandomStrategy, CanonicalAcquisition  |
|  GPStrategy, GridStrategy, HeuristicStrategy                |
|  VersionSpaceStrategy                                        |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                    Core Utilities                             |
+-------------------------------------------------------------+
|  LatentUserDistribution, LatentFeasibilityChecker            |
|  TestHistory, Config, Metrics, Utils                         |
+-------------------------------------------------------------+
                              |
+-------------------------------------------------------------+
|                    External Models                            |
+-------------------------------------------------------------+
|  infer_params.training.model.LevelSetDecoder                 |
|  infer_params.training.level_set_torch                       |
+-------------------------------------------------------------+
```

### 11.2 Import Modularity

**Good practices followed:**
1. **No circular imports:** Dependencies form a DAG
2. **Lazy imports in factory:** Strategy modules only imported when that strategy is selected
3. **Single learner class:** No learner inheritance hierarchy to maintain
4. **Config-driven:** Factory routes based on config, not hard-coded

---

## 12. Orchestrating Experiments & Customizing Baselines

This section shows how to use the two-axis configuration system to orchestrate experiments and compare strategies × posterior methods.

### 12.1 Basic Pipeline

```python
from active_learning.src.config import load_config, get_bounds_from_config
from active_learning.src.factory import build_learner
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_user_distribution import LatentUserDistribution

# 1. Load configuration
config = load_config("configs/latent.yaml")

# 2. Configure two axes
config['acquisition']['strategy'] = 'bald'       # Axis 1: test selection
config['posterior']['method'] = 'vi'             # Axis 2: posterior method

# 3. Load decoder model
decoder = ...  # Load LevelSetDecoder

# 4. Setup prior and posterior
prior_gen = LatentPriorGenerator(config, decoder)
prior = prior_gen.get_prior()
posterior = LatentUserDistribution(
    latent_dim=32, decoder=decoder,
    mean=prior.mean.clone(), log_std=prior.log_std.clone()
)

# 5. Create oracle and bounds
oracle = LatentOracle(decoder, ground_truth_z, n_joints=4)
bounds = get_bounds_from_config(config, device='cuda')

# 6. Build learner (factory returns LatentActiveLearner)
learner = build_learner(decoder, prior, posterior, oracle, bounds, config)

# 7. Run
results = learner.run(n_iterations=50, verbose=True)
```

### 12.2 Cross-Combination Experiments

**Key insight:** Any strategy can be combined with any posterior method. Just modify the two config keys:

```python
# Example 1: BALD with Ensemble posterior
config['acquisition']['strategy'] = 'bald'
config['posterior']['method'] = 'ensemble'
config['ensemble']['ensemble_size'] = 5
learner = build_learner(decoder, prior, posterior, oracle, bounds, config)
results = learner.run(verbose=True)

# Example 2: Random selection with SVGD posterior
config['acquisition']['strategy'] = 'random'
config['posterior']['method'] = 'svgd'
config['posterior']['n_particles'] = 20
learner = build_learner(decoder, prior, posterior, oracle, bounds, config)
results = learner.run(verbose=True)

# Example 3: GP with Ensemble posterior
config['acquisition']['strategy'] = 'gp'
config['posterior']['method'] = 'ensemble'
learner = build_learner(decoder, prior, posterior, oracle, bounds, config)
results = learner.run(verbose=True)

# Example 4: Grid search with VI posterior (default)
config['acquisition']['strategy'] = 'grid'
config['posterior']['method'] = 'vi'  # or omit (vi is default)
learner = build_learner(decoder, prior, posterior, oracle, bounds, config)
results = learner.run(verbose=True)

# Same API regardless of configuration
final_posterior = learner.get_posterior()
history = learner.get_history()
```

### 12.3 Comparative Study Template

Compare multiple strategy × posterior combinations:

```python
import itertools

strategies = ['bald', 'random', 'gp', 'grid']
posterior_methods = ['vi', 'ensemble', 'svgd']

results_table = {}

for strategy, posterior_method in itertools.product(strategies, posterior_methods):
    # Configure
    config['acquisition']['strategy'] = strategy
    config['posterior']['method'] = posterior_method

    # Setup fresh prior/posterior for each trial
    prior = prior_gen.get_prior()
    posterior = LatentUserDistribution(
        latent_dim=32, decoder=decoder,
        mean=prior.mean.clone(), log_std=prior.log_std.clone()
    )

    # Build and run
    learner = build_learner(decoder, prior, posterior, oracle, bounds, config)
    results = learner.run(n_iterations=50, verbose=False)

    # Collect metrics
    results_table[(strategy, posterior_method)] = {
        'final_elbo': results[-1].elbo,
        'mean_bald': np.mean([r.bald_score for r in results]),
        'convergence_iter': len(results)
    }

# Analyze
import pandas as pd
df = pd.DataFrame(results_table).T
print(df)
```

### 12.4 Diagnostic Runner with Two Axes

```bash
# Standard BALD with default VI posterior
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy bald --posterior-method vi --budget 40 --seed 42

# BALD with Ensemble posterior
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy bald --posterior-method ensemble --ensemble-size 5 --budget 40

# BALD with SVGD posterior
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy bald --posterior-method svgd --n-particles 20 --budget 40

# Random with Ensemble posterior (baseline)
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy random --posterior-method ensemble --budget 100

# GP with SVGD posterior
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy gp --posterior-method svgd --n-particles 15 --budget 40

# Grid search with VI (exhaustive + fast posterior)
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy grid --posterior-method vi --budget 100
```

### 12.5 Customizing Strategy Hyperparameters

All strategies share the same posterior update mechanism, but have different selection hyperparameters:

```python
# BALD-specific settings
config['bald']['tau'] = 1.0                          # Sigmoid temperature
config['bald']['n_mc_samples'] = 100                 # MC samples for BALD
config['bald_optimization']['n_restarts'] = 10       # Gradient restarts
config['bald_optimization']['lr_adam'] = 0.05        # Learning rate

# Quasi-random settings
config['acquisition']['n_quasi_random'] = 10         # Initial Sobol points

# Canonical settings
config['acquisition']['n_canonical'] = 5             # Number of canonical queries
config['acquisition']['canonical_path'] = 'models/canonical_queries.npz'

# GP settings
config['acquisition']['gp_n_candidates'] = 5000      # Candidate pool size

# Grid settings
config['acquisition']['grid_strategy_resolution'] = 16  # Grid resolution

# Build learner with custom hyperparameters
learner = build_learner(decoder, prior, posterior, oracle, bounds, config)
```

### 12.6 Customizing Posterior Method Hyperparameters

```python
# Ensemble configuration
config['posterior']['method'] = 'ensemble'
config['ensemble']['ensemble_size'] = 10             # Number of members
config['ensemble']['init_noise_std'] = 0.4           # Initialization perturbation

# SVGD configuration
config['posterior']['method'] = 'svgd'
config['posterior']['n_particles'] = 50              # Number of particles
config['svgd']['step_size'] = 0.1                    # SVGD step size
config['svgd']['max_iters'] = 100                    # SVGD optimization iters

# VI configuration (shared by all methods)
config['vi']['learning_rate'] = 0.058                # Adam LR
config['vi']['max_iters'] = 500                      # Max VI iterations
config['vi']['patience'] = 10                        # Early stopping
config['vi']['grad_clip'] = 1.0                      # Gradient clipping

# KL annealing (all posterior methods)
config['vi']['kl_annealing']['enabled'] = True
config['vi']['kl_annealing']['start_weight'] = 0.1
config['vi']['kl_annealing']['end_weight'] = 0.3
config['vi']['kl_annealing']['duration'] = 10

learner = build_learner(decoder, prior, posterior, oracle, bounds, config)
```

### 12.7 Batch Experiment Script

```python
# run_experiment_sweep.py
import yaml
from pathlib import Path

base_config = yaml.safe_load(open('configs/latent.yaml'))

experiment_configs = [
    # (name, strategy, posterior_method, custom_overrides)
    ('bald_vi', 'bald', 'vi', {}),
    ('bald_ensemble_k5', 'bald', 'ensemble', {'ensemble': {'ensemble_size': 5}}),
    ('bald_ensemble_k10', 'bald', 'ensemble', {'ensemble': {'ensemble_size': 10}}),
    ('bald_svgd_p20', 'bald', 'svgd', {'posterior': {'n_particles': 20}}),
    ('random_vi', 'random', 'vi', {}),
    ('random_ensemble', 'random', 'ensemble', {}),
    ('gp_vi', 'gp', 'vi', {}),
    ('gp_svgd', 'gp', 'svgd', {}),
]

for name, strategy, posterior_method, overrides in experiment_configs:
    config = deepcopy(base_config)
    config['acquisition']['strategy'] = strategy
    config['posterior']['method'] = posterior_method
    config.update(overrides)

    # Run experiment
    learner = build_learner(decoder, prior, posterior, oracle, bounds, config)
    results = learner.run(n_iterations=50, verbose=True)

    # Save results
    torch.save({
        'config': config,
        'results': results,
        'diagnostics': learner.diagnostics,
    }, f'outputs/{name}.pt')

    print(f"✓ {name}: {len(results)} iterations, final ELBO={results[-1].elbo:.2f}")
```

---

## 13. Extending the Codebase

This section is the definitive guide for adding new components. The architecture is designed for easy extension.

### 13.1 Adding a New Acquisition Strategy

This is the most common extension. Follow these steps:

**Step 1: Create the strategy class**

Create a new file in `active_learning/src/baselines/` (or `src/` for core strategies):

```python
# active_learning/src/baselines/my_strategy.py

class MyStrategy:
    """My custom acquisition strategy."""

    def __init__(self, config: dict = None, **kwargs):
        self.config = config or {}
        # Initialize any state needed

    def select_test(
        self,
        bounds,              # Tensor (n_joints, 2) with [lower, upper]
        verbose=False,
        test_history=None,   # List of past test point Tensors
        iteration=None,      # Current iteration index
        diagnostics=None,    # Diagnostics object (optional)
        **kwargs
    ):
        """
        Select the next test point.

        Returns:
            (test_point, score) - Tensor of shape (n_joints,) and float
            OR
            (test_point, score, stats) - with optional diagnostics list
        """
        # Your selection logic here
        test_point = ...  # Tensor of shape (n_joints,)
        score = ...       # float
        return test_point, score

    # OPTIONAL: implement if your strategy has state that updates after queries
    def post_query_update(self, test_point, outcome, history):
        """Called by learner after oracle query. Use for stateful strategies."""
        pass
```

**Key requirements:**
- `select_test()` must accept `bounds` as first positional arg and `**kwargs` to absorb extra arguments
- Return at minimum `(test_point, score)` where test_point is shape `(n_joints,)`
- `post_query_update()` is optional -- only needed for stateful strategies (like GP or VersionSpace)
- **Don't worry about posterior type** -- your strategy automatically works with vi, ensemble, and svgd!

**Step 2: Register in the factory**

Edit `active_learning/src/factory.py`, add a new `elif` in `_build_strategy()`:

```python
def _build_strategy(decoder, prior, posterior, config, bounds, oracle=None, embeddings=None):
    ...
    elif strategy_type == 'my_strategy':
        from active_learning.src.baselines.my_strategy import MyStrategy
        return MyStrategy(config=config)
    ...
```

**Step 3: Add to config comments**

In `configs/latent.yaml`, add your strategy name to the comments:

```yaml
acquisition:
  # Available strategies:
  #   ...
  #   my_strategy     - Description of your strategy
  strategy: bald
```

**Step 4: Add tests**

Add test cases in `test/test_refactored_pipeline.py`:

```python
# Test the strategy itself
def test_my_strategy_select_test(self, ...):
    strategy = MyStrategy(config=mock_config)
    result = strategy.select_test(mock_bounds)
    test_point = result[0]
    assert test_point.shape == (N_JOINTS,)

# Test cross-combinations (optional but recommended)
def test_my_strategy_with_ensemble(self, ...):
    config['acquisition']['strategy'] = 'my_strategy'
    config['posterior']['method'] = 'ensemble'
    learner = build_learner(...)
    result = learner.step()
    assert isinstance(result, LatentIterationResult)

def test_my_strategy_with_svgd(self, ...):
    config['acquisition']['strategy'] = 'my_strategy'
    config['posterior']['method'] = 'svgd'
    learner = build_learner(...)
    result = learner.step()
    assert isinstance(result, LatentIterationResult)
```

That's it. The factory handles wiring it into `LatentActiveLearner` automatically, and your strategy works with all posterior methods (vi, ensemble, svgd) without any extra code.

### 13.2 Adding a New Posterior Method

To add a new posterior inference method (like a fourth option beyond vi, ensemble, svgd):

**Step 1: Create posterior distribution class**

```python
# active_learning/src/my_posterior_distribution.py

class MyPosteriorDistribution:
    """Custom posterior representation."""

    def __init__(self, latent_dim, decoder, init_params, device='cpu'):
        self.latent_dim = latent_dim
        self.decoder = decoder
        self.device = device
        # Your custom parameters
        self.params = init_params

    def sample(self, n_samples, temperature=1.0, generator=None):
        """Sample latent codes. Must return (N, latent_dim) tensor."""
        ...

    # Optionally implement for compatibility
    @property
    def mean(self):
        """Representative mean (for diagnostics)."""
        ...

    @property
    def log_std(self):
        """Representative log_std (for diagnostics)."""
        ...
```

**Step 2: Create corresponding VI method**

```python
# active_learning/src/my_vi.py

from dataclasses import dataclass

@dataclass
class MyVIResult:
    converged: bool
    n_iterations: int
    final_elbo: float
    final_grad_norm: float
    elbo_history: list
    grad_norm_history: list

class MyVariationalInference:
    def __init__(self, decoder, prior, posterior, config):
        self.decoder = decoder
        self.prior = prior
        self.posterior = posterior  # MyPosteriorDistribution instance
        self.config = config
        self.kl_weight = config.get('vi', {}).get('kl_annealing', {}).get('end_weight', 1.0)

    def likelihood(self, test_history) -> torch.Tensor:
        """Compute log-likelihood of observations under posterior."""
        ...

    def regularizer(self, kl_weight=None) -> torch.Tensor:
        """Compute KL regularization term."""
        ...

    def update_posterior(self, test_history, kl_weight=None,
                         diagnostics=None, iteration=None) -> MyVIResult:
        """Run VI optimization and return result."""
        ...
```

**Step 3: Create BALD variant (optional)**

If your posterior needs a custom BALD implementation:

```python
# active_learning/src/my_bald.py

class MyBALD:
    def __init__(self, decoder, posterior, config, prior):
        self.decoder = decoder
        self.posterior = posterior  # MyPosteriorDistribution instance
        self.config = config
        self.prior = prior

    def compute_score(self, test_point, **kwargs):
        """Compute BALD score for your posterior type."""
        ...

    def select_test(self, bounds, **kwargs):
        """Optimize to find max BALD test point."""
        ...
```

**Step 4: Register in factory**

Edit `factory.py`:

```python
# In _build_posterior():
def _build_posterior(decoder, prior, posterior, config):
    method = config.get('posterior', {}).get('method', 'vi')
    ...
    elif method == 'my_method':
        from active_learning.src.my_posterior_distribution import MyPosteriorDistribution
        init_params = ...  # Extract from config
        return MyPosteriorDistribution(prior.latent_dim, decoder, init_params, device=DEVICE)
    ...

# In _build_vi():
def _build_vi(decoder, prior, posterior, config):
    method = config.get('posterior', {}).get('method', 'vi')
    ...
    elif method == 'my_method':
        from active_learning.src.my_vi import MyVariationalInference
        return MyVariationalInference(decoder, prior, posterior, config)
    ...

# In _build_bald() (if needed):
def _build_bald(decoder, prior, posterior, config):
    ...
    elif hasattr(posterior, 'my_custom_marker'):
        from active_learning.src.my_bald import MyBALD
        return MyBALD(decoder, posterior, config, prior)
    ...
```

**Step 5: Update config**

In `configs/latent.yaml`:

```yaml
posterior:
  method: my_method  # Options: vi, ensemble, svgd, my_method
  my_method_param_1: value
  my_method_param_2: value
```

**Step 6: Add tests**

Test your posterior with multiple strategies:

```python
def test_my_method_with_bald(self, ...):
    config['posterior']['method'] = 'my_method'
    config['acquisition']['strategy'] = 'bald'
    learner = build_learner(...)
    assert hasattr(learner.posterior, 'my_custom_marker')

def test_my_method_with_random(self, ...):
    config['posterior']['method'] = 'my_method'
    config['acquisition']['strategy'] = 'random'
    learner = build_learner(...)
    result = learner.step()
    assert isinstance(result, LatentIterationResult)
```

### 13.3 Adding a New Posterior Type

If your method needs a different posterior representation (beyond Gaussian or particles):

1. Create a class compatible with `LatentUserDistribution` interface (must have `mean`, `log_std`, `sample()`, `device`)
2. Pass it as `posterior` to `LatentActiveLearner`
3. If it's a list (like ensemble), the learner automatically detects `_is_ensemble = True`

### 13.4 Architecture Extension Points Summary

| What to Add | Files to Modify | Required Interface |
|-------------|-----------------|-------------------|
| **Strategy** | New file + `factory.py` `_build_strategy()` | `select_test(bounds, **kwargs) -> tuple` |
| **VI Method** | New file + `factory.py` `_build_vi()` | `likelihood()`, `regularizer()`, `update_posterior()` |
| **Posterior** | New file + `factory.py` `build_learner()` | `mean`, `log_std`, `sample()`, `device` |
| **Metric** | `metrics.py` | Any callable |
| **Stopping criterion** | `latent_active_learning.py` `check_stopping_criteria()` | Add to existing method |

---

## 14. Test Suite

### 14.1 Overview

**File:** `test/test_refactored_pipeline.py`
**Framework:** pytest
**Count:** 119+ tests across 15 test classes (including cross-combination tests)

```bash
# Run the full test suite
cd active_learning/
mamba run -n active_learning python -m pytest test/test_refactored_pipeline.py -v

# Run specific test class
mamba run -n active_learning python -m pytest test/test_refactored_pipeline.py::TestFactory -v

# Run cross-combination tests
mamba run -n active_learning python -m pytest test/test_refactored_pipeline.py::TestCrossCombinations -v
```

### 14.1.1 All Test Scripts and Entry Points

| Script | Purpose | Usage |
|--------|---------|-------|
| `test/test_refactored_pipeline.py` | Main pytest suite (119+ tests) | `pytest test/test_refactored_pipeline.py -v` |
| `test/diagnostics/run_latent_diagnosis.py` | Main experiment runner with full diagnostics | See Section 14.6 |
| `test/param_search/svgd_search.py` | Optuna hyperparameter search for SVGD | See Section 14.7 |
| `test/param_search/projected_svgd_search.py` | **NEW**: Hyperparameter search for Projected SVGD | See Section 14.7 |
| `test/profile_pipeline.py` | **NEW**: Performance profiling | See Section 14.8 |
| `test/profile_pipeline_detailed.py` | **NEW**: Detailed component profiling | See Section 14.8 |
| `test/test_batched_bald.py` | **NEW**: BALD batching tests | `python test/test_batched_bald.py` |

### 14.2 Test Classes

| Class | Tests | Covers |
|-------|-------|--------|
| `TestUtilityFunctions` | 17 | `binary_entropy`, `calculate_kl_weight`, `get_adaptive_param` |
| `TestMockDecoder` | 6 | Mock decoder shapes, gradients, determinism |
| `TestLatentUserDistribution` | 4 | Sampling, temperature, reproducibility |
| `TestLatentFeasibilityChecker` | 6 | Batched logits, decode, evaluate |
| `TestStrategyInterface` | 13 | All strategies: `select_test()` returns correct types |
| `TestFactory` | 12 | `build_learner()` for all configurations, error cases |
| `TestLearnerPipeline` | 10 | `step()`, `run()`, bounds checking, epsilon-greedy |
| `TestCrossCombinations` | 10 | **NEW**: Strategy × posterior cross-combinations |
| `TestStoppingCriteria` | 5 | Budget, BALD, ELBO plateau, uncertainty |
| `TestEnsembleSupport` | 8 | K posteriors, K VIs, `ensemble_predict_probs` |
| `TestSVGDSupport` | 4 | Particle posterior, particle count, step |
| `TestConfigIntegration` | 4 | Config loading, bounds, units conversion |
| `TestKLAnnealingIntegration` | 3 | Weight schedules, disabled annealing |
| `TestDiagnosticsIntegration` | 3 | Diagnostics creation, population |
| `TestStrategySpecificBehavior` | 15 | Per-strategy behavioral tests |

### 14.3 Cross-Combination Tests (NEW)

**Purpose:** Verify that any strategy works with any posterior method.

The `TestCrossCombinations` class tests the two-axis architecture by running combinations like:

- `random` + `ensemble`
- `random` + `svgd`
- `grid` + `ensemble`
- `grid` + `svgd`
- `gp` + `ensemble`
- `bald` + `ensemble`
- `bald` + `svgd`

**Example test:**
```python
def test_random_ensemble_step(self, ...):
    """Random selection + ensemble posterior produces valid result."""
    config['acquisition']['strategy'] = 'random'
    config['posterior']['method'] = 'ensemble'
    config['ensemble'] = {'ensemble_size': 3, 'init_noise_std': 0.2}
    learner = build_learner(...)
    result = learner.step()
    assert isinstance(result, LatentIterationResult)
    assert learner._is_ensemble
    assert isinstance(learner.vi, list)
```

**Why this matters:** These tests validate the clean separation of concerns and ensure that adding a new strategy automatically makes it compatible with all posterior methods (and vice versa).

### 14.4 MockDecoder

Tests use a lightweight `MockDecoder` (an `nn.Module`) that produces correct output shapes without needing a checkpoint file. It uses simple linear projections with `softplus` to guarantee `upper > lower`.

### 14.5 Main Diagnostic Runner

**File:** `test/diagnostics/run_latent_diagnosis.py`

The primary entry point for running experiments with full diagnostics and visualization.

**Command-Line Interface:**
```bash
# Basic usage with config defaults
python run_latent_diagnosis.py --budget 40

# Two-axis configuration
python run_latent_diagnosis.py \
    --strategy bald \                    # Axis 1: test selection
    --posterior-method ensemble \        # Axis 2: posterior inference
    --ensemble-size 5 \
    --budget 40

# New Projected SVGD with all parameters
python run_latent_diagnosis.py \
    --strategy bald \
    --posterior-method projected-svgd \
    --n-particles 50 \
    --budget 40

# New Multi-Stage Warmup
python run_latent_diagnosis.py \
    --strategy multi-stage-warmup \
    --posterior-method projected-svgd \
    --n-particles 30 \
    --budget 60
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--budget` | Max queries | 20 |
| `--strategy` | Test selection strategy | config |
| `--posterior-method` | Posterior inference method | config |
| `--ensemble-size` | K for ensemble | 5 |
| `--n-particles` | K for SVGD variants | 50 |
| `--seed` | Random seed | None |

**Outputs:**
- `images/diagnostics/latent/performance_metrics.png` - 4x2 grid of metrics
- `images/diagnostics/latent/latent_evolution.png` - Posterior std over time
- `images/diagnostics/latent/vi_metrics.png` - ELBO and gradient norms
- `images/diagnostics/latent/weighted_bald_gate.png` - BALD gate activation (if enabled)
- `diagnostics/latent/*.csv` - CSV logs for SVGD diagnostics

### 14.6 Hyperparameter Search Scripts

**File:** `test/param_search/projected_svgd_search.py` (NEW)

Optuna-based hyperparameter optimization for Projected SVGD.

**Usage:**
```bash
# Basic search
python -m active_learning.test.param_search.projected_svgd_search \
    --n-trials 50 --budget 20

# Resume existing study with persistent storage
python -m active_learning.test.param_search.projected_svgd_search \
    --n-trials 20 \
    --study-name projected_svgd_search_20260128 \
    --storage sqlite:///active_learning/optuna_studies.db
```

**Hyperparameters Searched:**
- `bald.tau_schedule.{start, end}`
- `projected_svgd.step_size`
- `projected_svgd.step_decay.power`
- `projected_svgd.n_slices`
- `projected_svgd.eigen_smoothing`
- `projected_svgd.kl_annealing.{start_weight, end_weight}`

**Objective:** Maximize AUC of IoU curve (area under IoU-vs-iteration).

**File:** `test/param_search/svgd_search.py`

Same structure for standard SVGD hyperparameter search.

### 14.7 Profiling Scripts (NEW)

**File:** `test/profile_pipeline.py`

Performance profiling with timing breakdown.

**Usage:**
```bash
python active_learning/test/profile_pipeline.py --iterations 10
```

**Output:** Timing report showing:
- Decoder forward pass time
- BALD optimization time
- VI update time
- Total iteration time
- Breakdown by component

**File:** `test/profile_pipeline_detailed.py`

More granular profiling with nested timing contexts.

**Usage:**
```bash
python active_learning/test/profile_pipeline_detailed.py --iterations 5
```

**Output:** Hierarchical timing report with per-component breakdown.

### 14.8 BALD Batching Tests (NEW)

**File:** `test/test_batched_bald.py`

Tests for BALD computation batching performance.

**Usage:**
```bash
python active_learning/test/test_batched_bald.py
```

---

## 15. Performance Considerations

### 15.1 Computational Bottlenecks

**BALD Score Computation:**
- **Optimization:** Pre-decode RBF parameters once per restart, reuse for all gradient steps

**VI Update:**
- **Standard VI:** T=500, N=100 MC samples, fast per iteration
- **SVGD:** T=100, K particles, more expensive per iteration (kernel computations)
- **Ensemble:** K x (Standard VI cost)

**Test Optimization:**
- Multi-restart gradient ascent: R=10 restarts, I=50 iterations each
- Adam -> SGD switching for stability near convergence

### 15.2 Memory Usage

| Component | Memory |
|-----------|--------|
| Decoder | ~10-50 MB (network params) |
| Single Posterior | 2D x 4 bytes (mean + log_std) |
| Ensemble (K=5) | 10D x 4 bytes |
| SVGD (K=50) | 50D x 4 bytes |
| Samples (BALD) | N x D x 4 bytes |

---

## 16. Common Pitfalls & Debugging

### 16.1 Posterior Collapse

**Symptom:** Posterior std -> 0, BALD scores -> 0

**Fix:** Increase `min_std`, reduce `kl_weight`, increase `tau`.

### 16.2 Vanishing Gradients

**Symptom:** ELBO flat, grad_norm < 0.01

**Fix:** Use `tau_schedule` starting high (2.0+), reduce VI learning rate if oscillating.

### 16.3 SVGD Particle Collapse

**Symptom:** All particles converge to single point

**Fix:** Reduce `step_size`, increase `n_particles`, or fix kernel bandwidth.

### 16.4 Debugging Checklist

```python
# 1. Check posterior health
std = torch.exp(learner.posterior.log_std) if not learner._is_ensemble else ...
print("Mean std:", std.mean().item())  # Should be > 0.01

# 2. Monitor iteration results
for r in learner.results:
    print(f"Iter {r.iteration}: BALD={r.bald_score:.4f}, ELBO={r.elbo:.2f}, grad={r.grad_norm:.4f}")

# 3. Check diagnostics
learner.diagnostics.print_final_report()
```

---

## Quick Reference

### File Locations

```
Core:
- Learner:   active_learning/src/latent_active_learning.py
- Factory:   active_learning/src/factory.py
- BALD:      active_learning/src/latent_bald.py
- VI:        active_learning/src/latent_variational_inference.py
- Utils:     active_learning/src/utils.py

Ensemble:
- active_learning/src/ensemble/ensemble_bald.py

SVGD:
- active_learning/src/svgd/svgd_vi.py
- active_learning/src/svgd/svgd_optimizer.py
- active_learning/src/svgd/sliced_svgd_optimizer.py      # NEW
- active_learning/src/svgd/sliced_svgd_vi.py             # NEW
- active_learning/src/svgd/projected_svgd_optimizer.py   # NEW
- active_learning/src/svgd/projected_svgd_vi.py          # NEW
- active_learning/src/svgd/particle_bald.py
- active_learning/src/svgd/particle_user_distribution.py

Baselines:
- active_learning/src/baselines/random_strategy.py
- active_learning/src/baselines/quasi_random_strategy.py
- active_learning/src/baselines/prior_boundary_strategy.py
- active_learning/src/baselines/multi_stage_warmup_strategy.py  # NEW
- active_learning/src/baselines/gp_strategy.py
- active_learning/src/baselines/grid_strategy.py
- active_learning/src/baselines/heuristic_strategy.py
- active_learning/src/baselines/version_space_strategy.py

Config:
- active_learning/configs/latent.yaml

Tests:
- active_learning/test/test_refactored_pipeline.py
- active_learning/test/test_batched_bald.py              # NEW
- active_learning/test/profile_pipeline.py               # NEW
- active_learning/test/profile_pipeline_detailed.py      # NEW

Diagnostics:
- active_learning/test/diagnostics/run_latent_diagnosis.py

Hyperparameter Search:
- active_learning/test/param_search/svgd_search.py
- active_learning/test/param_search/projected_svgd_search.py  # NEW
```

### Key Functions

```python
# Build learner (returns LatentActiveLearner for ALL strategies)
from active_learning.src.factory import build_learner
learner = build_learner(decoder, prior, posterior, oracle, bounds, config)

# Run learning
results = learner.run(n_iterations=50, verbose=True)

# Compute metrics
from active_learning.src.metrics import compute_reachability_metrics
iou, acc, f1 = compute_reachability_metrics(...)

# Diagnostics
learner.diagnostics.print_final_report()
```

---

## References

1. **BALD:** Houlsby et al. (2011) - "Bayesian Active Learning by Disagreement"
2. **SVGD:** Liu & Wang (2016) - "Stein Variational Gradient Descent"
3. **Sliced SVGD:** Wang et al. (2019) - "Sliced Stein Variational Gradient Descent"
4. **Projected Kernel SVGD:** Chen et al. (2020) - "Projected Stein Variational Newton"
5. **Deep Ensembles:** Lakshminarayanan et al. (2017) - "Simple and Scalable Predictive Uncertainty"
6. **Level Sets:** Osher & Sethian (1988) - "Fronts Propagating with Curvature"

---

**Documentation Version:** 3.0
**Last Updated:** January 28, 2026

---

## Changelog

### Version 3.0 (January 2026)

**New Posterior Methods:**
- Added **Projected SVGD** (`projected_svgd`): Eigenvector-informed SVGD with adaptive direction selection
- Added **Sliced SVGD** (`sliced_svgd`): 1D projection-based SVGD for efficiency

**New Acquisition Strategies:**
- Added **Multi-Stage Warmup** (`multi_stage_warmup`): Adaptive boundary discovery with entropy-based stopping

**New Test Scripts:**
- `test/param_search/projected_svgd_search.py`: Optuna hyperparameter search for Projected SVGD
- `test/profile_pipeline.py`: Performance profiling with timing breakdown
- `test/profile_pipeline_detailed.py`: Detailed component-level profiling
- `test/test_batched_bald.py`: BALD batching performance tests

**Performance Improvements:**
- Vectorized `_compute_1d_svgd_batched()` for Projected SVGD
- GPU-optimized farthest-point sampling in multi-stage warmup

### Version 2.0 (January 2026)
- Initial two-axis architecture (strategy × posterior method)
- Factory pattern with auto-detection
- Cross-combination test suite
