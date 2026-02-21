# Quick Start Guide for Agents

**TL;DR:** This codebase implements Bayesian active learning in a learned latent space to efficiently discover kinematic feasibility boundaries. A single unified learner (`LatentActiveLearner`) handles all strategies and posterior methods via a factory pattern with two orthogonal configuration axes.

---

## 30-Second Overview

```
Active Learning Loop (LatentActiveLearner.step()):
  1. Strategy selects most informative test query
  2. Oracle returns feasibility (signed distance)
  3. VI updates posterior belief about user constraints
  4. Repeat until budget exhausted or converged
```

**Key Innovation:** Work in 32D latent space (not high-dim joint space) using a pre-trained decoder.

**Key Architecture:** ONE learner class, TWO orthogonal axes (strategy × posterior method), factory assembles components.

---

## Unified Architecture with Two Orthogonal Axes

There is a **single learner class** (`LatentActiveLearner`) for all configurations. The factory (`build_learner()`) injects the right strategy and posterior type based on TWO independent config axes:

### Axis 1: Test Selection Strategy (`acquisition.strategy`)
- `bald` - Gradient-optimized BALD (default)
- `random` - Uniform random selection
- `quasi_random` - Sobol sequence then BALD
- `prior_boundary` - Prior boundary sampling then BALD
- `multi_stage_warmup` - **NEW**: Adaptive multi-stage boundary discovery then BALD
- `canonical` - Pre-defined canonical queries then BALD
- `gp` - Gaussian Process with Straddle heuristic
- `grid` - Grid-based evaluation
- `heuristic` - Dense banking (requires embeddings)
- `version_space` - Greedy maximin hypothesis pruning (requires embeddings)

### Axis 2: Posterior Inference Method (`posterior.method`)
- `vi` - Single Gaussian (default, fastest)
- `full_cov_vi` - Full covariance Gaussian
- `ensemble` - K independent Gaussians (stable, K× cost)
- `svgd` - K interacting particles with RBF kernel
- `sliced_svgd` - Sliced SVGD with 1D projections (faster)
- `projected_svgd` - **NEW**: Projected SVGD with eigenvector-informed directions (recommended)

### Cross-Combination Examples

| Strategy | Posterior Method | Use Case |
|----------|------------------|----------|
| `bald` | `vi` | Default: fast, gradient-optimized BALD |
| `bald` | `ensemble` | Stable BALD with ensemble disagreement |
| `bald` | `svgd` | Flexible BALD with particle-based uncertainty |
| `bald` | `projected_svgd` | **Recommended**: BALD with eigenvector-informed particles |
| `multi_stage_warmup` | `projected_svgd` | Adaptive warmup + efficient particle inference |
| `random` | `ensemble` | Random selection with ensemble posterior updates |
| `gp` | `svgd` | GP acquisition with particle-based VI |
| `quasi_random` | `projected_svgd` | Sobol warmup + Projected SVGD refinement |

**Any strategy can be combined with any posterior method** - BALD auto-detects the posterior type.

---

## Critical Files (Read These First)

1. **`factory.py`** - Single entry point that builds the learner with correct components
2. **`latent_active_learning.py`** - Unified active learning loop (all strategies)
3. **`latent_bald.py`** - BALD acquisition + gradient-based test optimization
4. **`latent_variational_inference.py`** - Standard VI posterior update
5. **`utils.py`** - Shared utilities (`binary_entropy`, KL weight, adaptive params)
6. **`config.py`** - Configuration loading, DEVICE, bounds

---

## Quick Code Reading Path

### Path 1: Understand the Pipeline
```
1. factory.py :: build_learner()
   -- See how strategy, VI, and posterior are assembled
   |
2. latent_active_learning.py :: LatentActiveLearner.step()
   -- The main loop: select -> query -> update
   |
3. latent_bald.py :: LatentBALD.select_test()
   -- How test points are optimized via gradient ascent
   |
4. latent_variational_inference.py :: update_posterior()
   -- How the posterior is updated via ELBO maximization
```

### Path 2: Understand Ensemble
```
1. factory.py :: _build_posterior() [posterior.method='ensemble']
   -- Creates K posteriors with perturbed means
   |
2. factory.py :: _build_bald()
   -- Auto-detects list → returns EnsembleBALD
   |
3. ensemble/ensemble_bald.py :: EnsembleBALD.compute_score()
   -- BALD via K-member disagreement
   |
4. latent_active_learning.py :: step() [self._is_ensemble branch]
   -- Updates all K VI optimizers independently
```

### Path 3: Understand SVGD
```
1. factory.py :: _build_posterior() [posterior.method='svgd']
   -- Creates ParticleUserDistribution
   |
2. factory.py :: _build_bald()
   -- Auto-detects get_particles → returns ParticleBALD
   |
3. svgd/particle_bald.py :: ParticleBALD
   -- Uses all particles (no MC sampling)
   |
4. svgd/svgd_vi.py :: SVGDVariationalInference.update_posterior()
   -- Stein forces update particles
   |
5. svgd/svgd_optimizer.py :: SVGD.step()
   -- RBF kernel, attraction + repulsion
```

### Path 4: Understand Projected SVGD (NEW)
```
1. factory.py :: _build_posterior() [posterior.method='projected_svgd']
   -- Creates ParticleUserDistribution
   |
2. svgd/projected_svgd_vi.py :: ProjectedSVGDVariationalInference
   -- Wraps ProjectedSVGD optimizer
   |
3. svgd/projected_svgd_optimizer.py :: ProjectedSVGD.step()
   |-- Phase 1: Score covariance eigendecomposition
   |   -- C = grad.T @ grad / K → eigenvalues, eigenvectors
   |
   |-- Phase 2: Adaptive r selection
   |   -- Find r eigenvectors capturing variance_threshold
   |
   |-- Phase 3: Projected directions (r eigenvector slices)
   |   -- 1D SVGD with eigenvalue weighting
   |
   |-- Phase 4: Random directions (n_slices - r)
   |   -- Standard Sliced SVGD for low-variance dimensions
   |
   +-- Phase 5: Combine and normalize
```

### Path 5: Understand Multi-Stage Warmup (NEW)
```
1. factory.py :: _build_strategy() [strategy='multi_stage_warmup']
   -- Creates MultiStageWarmupStrategy with BALD delegate
   |
2. baselines/multi_stage_warmup_strategy.py :: select_test()
   |
   |-- In warmup phase:
   |   |-- Stage 1: Prior mean boundary points
   |   |-- Stage 2+: Current posterior boundary points
   |   |-- Use farthest-point sampling for diversity
   |   +-- Check entropy-based stopping criterion
   |
   +-- After warmup (or adaptive stop):
       -- Delegate to BALD for refinement
       |
3. post_query_update() callback:
   -- Track outcomes for rolling entropy computation
   -- Switch to BALD when entropy >= threshold
```

---

## Key Concepts

### BALD (Bayesian Active Learning by Disagreement)
```
Score = H(mean prediction) - mean(H(predictions))
      = Information gain about parameters
```
High BALD = high disagreement among posterior samples = informative query.

### Latent Space
```
z in R^32 -> Decoder -> (box_limits, bump_centers, bump_strengths)
                     -> Level-set function f(q; z)
                     -> Feasibility: f(q) >= 0
```

### Variational Inference
```
Standard VI: Optimize Gaussian (mu, sigma) to maximize ELBO
SVGD:        Move particles via Stein forces (attraction + repulsion)
Ensemble:    K independent VI optimizations
```

### Factory Pattern with Two Axes
```python
# The factory ALWAYS returns LatentActiveLearner
learner = build_learner(decoder, prior, posterior, oracle, bounds, config)

# Switch strategy by changing config -- no code changes needed
config['acquisition']['strategy'] = 'bald'           # or random, gp, grid, etc.
config['posterior']['method'] = 'ensemble'           # or vi, svgd

# Examples of cross-combinations:
# BALD + Ensemble
config['acquisition']['strategy'] = 'bald'
config['posterior']['method'] = 'ensemble'

# Random + SVGD
config['acquisition']['strategy'] = 'random'
config['posterior']['method'] = 'svgd'

# GP + VI (default)
config['acquisition']['strategy'] = 'gp'
config['posterior']['method'] = 'vi'
```

---

## Common Tasks

### Run Diagnostics
```bash
# Standard BALD (default: vi posterior)
python active_learning/test/diagnostics/run_latent_diagnosis.py --budget 40

# BALD with Ensemble posterior
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy bald --posterior-method ensemble --ensemble-size 5 --budget 40

# BALD with SVGD posterior
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy bald --posterior-method svgd --n-particles 20 --budget 40

# NEW: BALD with Projected SVGD (eigenvector-informed)
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy bald --posterior-method projected-svgd --n-particles 50 --budget 40

# NEW: Multi-Stage Warmup with adaptive stopping
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy multi-stage-warmup --posterior-method projected-svgd \
    --n-particles 30 --budget 60

# Random selection with Ensemble posterior
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy random --posterior-method ensemble --budget 40

# GP with SVGD posterior
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy gp --posterior-method svgd --budget 40
```

### Run Tests
```bash
# Full test suite (119+ tests including cross-combination tests)
mamba run -n active_learning python -m pytest active_learning/test/test_refactored_pipeline.py -v

# Specific test class
mamba run -n active_learning python -m pytest active_learning/test/test_refactored_pipeline.py::TestFactory -v

# Cross-combination tests
mamba run -n active_learning python -m pytest active_learning/test/test_refactored_pipeline.py::TestCrossCombinations -v
```

### Hyperparameter Search
```bash
# Optuna search for Projected SVGD parameters
python -m active_learning.test.param_search.projected_svgd_search \
    --n-trials 50 --budget 20

# Resume existing study
python -m active_learning.test.param_search.projected_svgd_search \
    --n-trials 20 --study-name my_study \
    --storage sqlite:///active_learning/optuna_studies.db

# SVGD hyperparameter search
python -m active_learning.test.param_search.svgd_search --n-trials 50 --budget 20
```

### Performance Profiling
```bash
# Profile pipeline components
python active_learning/test/profile_pipeline.py --iterations 10

# Detailed profiling with component breakdown
python active_learning/test/profile_pipeline_detailed.py --iterations 5

# Test BALD batching performance
python active_learning/test/test_batched_bald.py
```

### Configure Strategy and Posterior Method
```yaml
# configs/latent.yaml

# Axis 1: Test selection strategy
acquisition:
  strategy: bald    # Options: bald, random, quasi_random, prior_boundary,
                    # multi_stage_warmup, canonical, gp, grid, heuristic, version_space

# Axis 2: Posterior inference method
posterior:
  method: vi        # Options: vi, full_cov_vi, ensemble, svgd, sliced_svgd, projected_svgd
  n_particles: 50   # (for svgd variants only)

# Ensemble configuration (for posterior.method: ensemble)
ensemble:
  ensemble_size: 5
  init_noise_std: 0.4

# NEW: Projected SVGD configuration
projected_svgd:
  step_size: 0.1
  n_slices: 20                    # Total directions (eigenvector + random)
  variance_threshold: 0.95        # Capture 95% of score variance with eigenvectors
  max_eigenweight: 3.0            # Prevent needle updates from peaked spectrum
  eigen_smoothing: 0.5            # 0=uniform, 0.5=sqrt, 1.0=linear weighting
  kernel_type: imq                # 'rbf' or 'imq' (inverse multi-quadric)

# NEW: Multi-Stage Warmup configuration
multi_stage_warmup:
  n_stages: 5                     # Maximum warmup stages
  queries_per_stage: 5            # Queries per stage
  n_candidates: 5000              # Candidate pool for boundary search
  boundary_percentile: 0.05       # Fraction closest to p=0.5
  use_farthest_point: true        # Spatial diversity sampling
  final_phase: bald               # 'bald', 'posterior_boundary', or 'stop'
  adaptive_stopping: true         # Use entropy-based stopping
  entropy_threshold: 0.9          # Entropy to trigger switch (0.9 ≈ 35-65% ratio)
  window_size: 15                 # Rolling window for entropy
  min_warmup_queries: 10          # Minimum before checking entropy
```

### Modify Hyperparameters
```yaml
vi:
  learning_rate: 0.058     # Adam LR for VI
  max_iters: 500           # Max VI iterations per update
  patience: 10             # Early stopping

bald:
  tau: 1.0                 # Sigmoid temperature
  n_mc_samples: 100        # Samples for BALD estimate

bald_optimization:
  n_restarts: 10           # Random restarts for test selection
  lr_adam: 0.05            # Learning rate for test optimization
```

---

## Architecture at a Glance

```
Factory (factory.py)
    | builds
LatentActiveLearner (SINGLE unified learner)
    |-- strategy: LatentBALD / EnsembleBALD / ParticleBALD / Random / GP / ...
    |-- vi: LatentVI / List[LatentVI] / SVGDVI
    |-- posterior: LatentUserDistribution / List[...] / ParticleUserDistribution
    |-- prior: LatentUserDistribution
    |-- oracle: LatentOracle
    +-- diagnostics: Diagnostics
```

**Execution Flow:**
```python
for iteration in range(budget):
    test_point = strategy.select_test(bounds)
    outcome = oracle.query(test_point)
    if hasattr(strategy, 'post_query_update'):
        strategy.post_query_update(test_point, outcome, history)
    vi.update_posterior(history)
    diagnostics.log_iteration(...)
```

---

## Key Data Structures

### LatentIterationResult
```python
@dataclass
class LatentIterationResult:
    iteration: int
    test_point: Tensor        # (J,)
    outcome: float            # Signed distance
    bald_score: float
    elbo: float
    grad_norm: float
    vi_converged: bool
    vi_iterations: int
```

### Config Structure
```yaml
bald:                    # BALD settings (tau, MC samples, weighted BALD)
bald_optimization:       # Test selection optimization (restarts, LR)
vi:                      # Variational inference (LR, max iters, KL annealing)
acquisition:             # Strategy selection (axis 1)
  strategy: bald
posterior:               # Posterior method (axis 2)
  method: vi             # Options: vi, ensemble, svgd
  n_particles: 50        # For SVGD
stopping:                # Stopping criteria (budget, BALD, ELBO, uncertainty)
prior:                   # Prior initialization (joint names, anatomical limits)
latent:                  # Model paths
ensemble:                # Ensemble config (for posterior.method: ensemble)
  ensemble_size: 5
  init_noise_std: 0.4
svgd:                    # SVGD config (step size)
  step_size: 0.1
  max_iters: 100
```

---

## Adding a New Acquisition Strategy

This is the most common extension. The modular architecture makes it straightforward, and **your new strategy automatically works with all posterior methods (vi, ensemble, svgd)**:

### Step 1: Create strategy class
```python
# active_learning/src/baselines/my_strategy.py
class MyStrategy:
    def __init__(self, config=None, **kwargs):
        self.config = config or {}

    def select_test(self, bounds, **kwargs):
        """Return (test_point, score) or (test_point, score, stats)."""
        test_point = ...  # Tensor of shape (n_joints,)
        score = 0.0
        return test_point, score

    # OPTIONAL: for stateful strategies (like GP, VersionSpace)
    def post_query_update(self, test_point, outcome, history):
        pass
```

### Step 2: Register in factory
```python
# factory.py :: _build_strategy()
elif strategy_type == 'my_strategy':
    from active_learning.src.baselines.my_strategy import MyStrategy
    return MyStrategy(config=config)
```

### Step 3: Add tests (including cross-combinations)
```python
# test/test_refactored_pipeline.py
def test_my_strategy(self, mock_bounds, mock_config):
    strategy = MyStrategy(config=mock_config)
    result = strategy.select_test(mock_bounds)
    assert result[0].shape == (N_JOINTS,)

# Test with different posterior methods
def test_my_strategy_with_ensemble(self, ...):
    config['acquisition']['strategy'] = 'my_strategy'
    config['posterior']['method'] = 'ensemble'
    learner = build_learner(...)
    result = learner.step()
    assert isinstance(result, LatentIterationResult)
```

Done. The factory automatically wires it into `LatentActiveLearner`, and it works with any posterior method (vi, ensemble, svgd) without any extra code.

### Adding a New Posterior Method

Implement a new `_build_posterior()` case in `factory.py`, create corresponding VI and BALD classes if needed, and register in `_build_vi()` and `_build_bald()`.

See [ARCHITECTURE.md Section 13](ARCHITECTURE.md#13-extending-the-codebase) for full details.

---

## Debugging Quick Checks

### Posterior Collapse?
```python
std = torch.exp(learner.posterior.log_std)
print("Mean std:", std.mean().item())  # Should be > 0.01
```

### VI Not Converging?
```python
for result in learner.results:
    print(f"Iter {result.iteration}: grad={result.grad_norm:.4f}")
# grad_norm should be > 0.01
```

### BALD Scores Too Low?
```python
for result in learner.results:
    print(f"Iter {result.iteration}: BALD={result.bald_score:.4f}")
# BALD should be > 0.01 in early iterations
```

### Check Diagnostics
```python
learner.diagnostics.print_final_report()
```

---

## Comparison Table: Posterior Methods

| Feature | VI | Full Cov VI | Ensemble | SVGD | Sliced SVGD | Projected SVGD |
|---------|-----|-------------|----------|------|-------------|----------------|
| **Config** | `vi` | `full_cov_vi` | `ensemble` | `svgd` | `sliced_svgd` | `projected_svgd` |
| **Posterior** | 1 Gaussian | 1 full-cov | K Gaussians | K particles | K particles | K particles |
| **Memory** | O(D) | O(D²) | O(K×D) | O(K×D) | O(K×D) | O(K×D) |
| **Speed** | Fast | Medium | K× slower | K× slower | Faster | **Fastest SVGD** |
| **Uncertainty** | Diagonal σ | Full Σ | Disagreement | Empirical | Empirical | Eigenvector-informed |
| **Best For** | Prototyping | Correlations | Stable | Complex | High-dim | **Recommended** |

**Remember:** Any strategy (bald, random, gp, etc.) can use any posterior method!

### SVGD Variant Comparison

| Variant | Key Innovation | When to Use |
|---------|----------------|-------------|
| **SVGD** | Standard RBF kernel | Baseline, simple cases |
| **Sliced SVGD** | 1D random projections | High dimensions, faster |
| **Projected SVGD** | Eigenvector-informed directions | **Best default**, captures structure |

---

## For Complete Details

See `ARCHITECTURE.md` for:
- Full theoretical foundation
- Detailed API reference
- File dependencies
- Comprehensive extensibility guide (Section 13)
- Test suite documentation (Section 14)
- Performance optimization
- Complete usage examples

---

**Quick References:**
- Main config: `active_learning/configs/latent.yaml`
- Main runner: `active_learning/test/diagnostics/run_latent_diagnosis.py`
- Factory: `active_learning/src/factory.py`
- Tests: `active_learning/test/test_refactored_pipeline.py`
- Full docs: `active_learning/docs/ARCHITECTURE.md`
