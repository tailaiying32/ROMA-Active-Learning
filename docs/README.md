# Active Learning Documentation

Comprehensive documentation for the Bayesian active learning pipeline.

**Architecture:** A single unified learner (`LatentActiveLearner`) handles all configurations via a factory pattern with **two orthogonal axes**: test selection strategy (10+ options) × posterior inference method (6 options) = any combination works.

---

## Documentation Index

### For Quick Understanding
**[QUICK_START.md](QUICK_START.md)**
- 30-second overview
- Two-axis architecture (strategy × posterior method)
- Critical files to read first
- Code reading paths
- Common tasks and commands
- How to orchestrate experiments
- How to add new strategies (auto-compatible with all posterior methods)

### For Visual Learners
**[VISUAL_REFERENCE.md](VISUAL_REFERENCE.md)**
- Pipeline flow diagrams
- Unified learner architecture
- Component composition
- BALD score computation flow
- VI update flow (Standard, SVGD, Projected SVGD)
- Ensemble architecture
- Data flow diagrams
- Import graphs
- Strategy extension pattern
- Debug flow charts

### For Complete Understanding
**[ARCHITECTURE.md](ARCHITECTURE.md)**
- Theoretical foundation (BALD, VI, SVGD, Projected SVGD)
- Codebase architecture and directory structure
- Core components deep dive
- All 10+ acquisition strategies (each works with all 6 posterior methods)
- All 6 posterior methods (vi, full_cov_vi, ensemble, svgd, sliced_svgd, projected_svgd)
- Unified pipeline design with two orthogonal axes
- Complete API reference
- File dependencies and modularity
- **Orchestrating experiments** with cross-combinations
- **Extending the codebase** (adding strategies auto-works with all posterior methods)
- **Test suite** documentation (119+ tests including cross-combinations)
- Performance considerations
- Common pitfalls and debugging

---

## Quick Navigation

**I want to...**

### Understand the basics
-> Start with [QUICK_START.md - 30-Second Overview](QUICK_START.md#30-second-overview)

### See the code structure visually
-> See [VISUAL_REFERENCE.md - Unified Learner Architecture](VISUAL_REFERENCE.md#2-unified-learner-architecture)

### Understand how strategies are assembled
-> Read [ARCHITECTURE.md - Factory Pattern](ARCHITECTURE.md#32-design-patterns)
-> See [VISUAL_REFERENCE.md - Config Mapping](VISUAL_REFERENCE.md#8-config-to-component-mapping)

### Add a new acquisition strategy
-> Read [ARCHITECTURE.md - Adding a New Strategy](ARCHITECTURE.md#131-adding-a-new-acquisition-strategy)
-> See [VISUAL_REFERENCE.md - Strategy Extension Pattern](VISUAL_REFERENCE.md#10-strategy-extension-pattern)
-> See [QUICK_START.md - Adding a New Strategy](QUICK_START.md#adding-a-new-acquisition-strategy)

### Add a new VI method
-> Read [ARCHITECTURE.md - Adding a New VI Method](ARCHITECTURE.md#132-adding-a-new-vi-method)

### Understand BALD acquisition
-> Read [ARCHITECTURE.md - BALD Theory](ARCHITECTURE.md#22-bald-bayesian-active-learning-by-disagreement)
-> See [VISUAL_REFERENCE.md - BALD Flow](VISUAL_REFERENCE.md#4-bald-score-computation-flow)

### Understand variational inference
-> Read [ARCHITECTURE.md - VI Methods](ARCHITECTURE.md#6-variational-inference-methods)
-> See [VISUAL_REFERENCE.md - VI Update Flow](VISUAL_REFERENCE.md#5-variational-inference-update-flow)

### Orchestrate cross-combination experiments
-> See [ARCHITECTURE.md - Orchestrating Experiments](ARCHITECTURE.md#12-orchestrating-experiments--customizing-baselines)
-> See [QUICK_START.md - Configure Strategy and Posterior](QUICK_START.md#configure-strategy-and-posterior-method)

### Run experiments
-> See [QUICK_START.md - Common Tasks](QUICK_START.md#common-tasks)

### Run the test suite
-> See [ARCHITECTURE.md - Test Suite](ARCHITECTURE.md#14-test-suite)

### Debug issues
-> See [ARCHITECTURE.md - Common Pitfalls](ARCHITECTURE.md#16-common-pitfalls--debugging)
-> See [VISUAL_REFERENCE.md - Debug Flow](VISUAL_REFERENCE.md#11-typical-debug-flow)

### Understand config options
-> Read [ARCHITECTURE.md - Configuration System](ARCHITECTURE.md#8-configuration-system)

---

## Codebase Structure

```
active_learning/
|-- src/                     # Source code
|   |-- Core components      #   config, utils, distribution, checker, oracle
|   |-- Pipeline             #   learner, BALD, VI (single unified learner)
|   |-- ensemble/            #   Ensemble BALD strategy
|   |-- svgd/                #   SVGD components (particles, VI, optimizers)
|   |   |-- svgd_optimizer.py           # Standard SVGD (RBF kernel)
|   |   |-- sliced_svgd_optimizer.py    # Sliced SVGD (1D projections)
|   |   +-- projected_svgd_optimizer.py # NEW: Projected SVGD (eigenvector-informed)
|   |-- baselines/           #   Acquisition strategies
|   |   |-- random_strategy.py          # Uniform random
|   |   |-- quasi_random_strategy.py    # Sobol sequences
|   |   |-- prior_boundary_strategy.py  # Prior boundary → BALD
|   |   |-- multi_stage_warmup_strategy.py  # NEW: Adaptive multi-stage warmup
|   |   |-- gp_strategy.py              # Gaussian Process
|   |   |-- grid_strategy.py            # Grid search
|   |   |-- heuristic_strategy.py       # Dense banking
|   |   +-- version_space_strategy.py   # Maximin hypothesis pruning
|   +-- factory.py           #   Single entry point with two-axis logic: build_learner()
|-- configs/                 # Configuration files
|   +-- latent.yaml          #   Master configuration
|-- test/                    # Test suite (119+ tests) and diagnostics
|   |-- test_refactored_pipeline.py     # Main pytest suite
|   |-- diagnostics/
|   |   +-- run_latent_diagnosis.py     # Main experiment runner
|   |-- param_search/
|   |   |-- svgd_search.py              # SVGD hyperparameter search
|   |   +-- projected_svgd_search.py    # NEW: Projected SVGD search
|   |-- profile_pipeline.py             # NEW: Performance profiling
|   +-- profile_pipeline_detailed.py    # NEW: Detailed profiling
+-- docs/                    # Documentation (you are here)
    |-- README.md            #   This file (index)
    |-- QUICK_START.md       #   Quick onboarding
    |-- ARCHITECTURE.md      #   Complete reference
    |-- VISUAL_REFERENCE.md  #   Visual diagrams
    |-- GPU_OPTIMIZATION_PLAN.md        # GPU optimization roadmap
    |-- PHASE2_VECTORIZE_LEVEL_SET.md   # Level-set vectorization
    +-- PHASE3_VECTORIZE_PROJECTED_SVGD.md  # Projected SVGD vectorization
```

---

## Reading Order by Role

### For New Developers
1. [QUICK_START.md](QUICK_START.md) - Get oriented
2. [VISUAL_REFERENCE.md](VISUAL_REFERENCE.md) - See structure
3. [ARCHITECTURE.md - Core Components](ARCHITECTURE.md#4-core-components) - Understand basics
4. Run experiments using [QUICK_START.md - Common Tasks](QUICK_START.md#common-tasks)

### For ML Researchers
1. [ARCHITECTURE.md - Theoretical Foundation](ARCHITECTURE.md#2-theoretical-foundation) - Theory
2. [ARCHITECTURE.md - Acquisition Strategies](ARCHITECTURE.md#5-acquisition-strategies) - Algorithms
3. [ARCHITECTURE.md - VI Methods](ARCHITECTURE.md#6-variational-inference-methods) - Compare approaches
4. [VISUAL_REFERENCE.md - BALD/VI Flows](VISUAL_REFERENCE.md#4-bald-score-computation-flow)

### For Future AI Agents
1. [QUICK_START.md](QUICK_START.md) - Entire file (quick scan)
2. [ARCHITECTURE.md - Unified Pipeline](ARCHITECTURE.md#7-unified-pipeline) - How it all fits together
3. [ARCHITECTURE.md - Extending the Codebase](ARCHITECTURE.md#13-extending-the-codebase) - How to add strategies/VI/posteriors
4. [ARCHITECTURE.md - Test Suite](ARCHITECTURE.md#14-test-suite) - Existing test coverage
5. [VISUAL_REFERENCE.md](VISUAL_REFERENCE.md) - Reference as needed

### For Debugging
1. [ARCHITECTURE.md - Common Pitfalls](ARCHITECTURE.md#16-common-pitfalls--debugging)
2. [VISUAL_REFERENCE.md - Debug Flow](VISUAL_REFERENCE.md#11-typical-debug-flow)
3. [QUICK_START.md - Debugging Quick Checks](QUICK_START.md#debugging-quick-checks)

---

## Key Concepts

### Two-Axis Architecture
The system has **two orthogonal configuration axes**:

**Axis 1: Test Selection Strategy** (`acquisition.strategy`)
- `bald` - Gradient-optimized BALD (default)
- `random` - Uniform random selection
- `quasi_random` - Sobol sequence → BALD
- `prior_boundary` - Prior boundary sampling → BALD
- `multi_stage_warmup` - **NEW**: Adaptive multi-stage boundary discovery → BALD
- `canonical` - Pre-defined canonical queries → BALD
- `gp` - Gaussian Process + Straddle heuristic
- `grid` - Grid-based exhaustive search
- `heuristic` - Dense banking (candidate evaluation)
- `version_space` - Greedy maximin hypothesis pruning

**Axis 2: Posterior Inference Method** (`posterior.method`)
- `vi` - Single Gaussian (default, fastest)
- `full_cov_vi` - Full covariance Gaussian
- `ensemble` - K independent Gaussians
- `svgd` - K interacting particles (standard SVGD)
- `sliced_svgd` - Sliced SVGD (1D projections)
- `projected_svgd` - **NEW**: Projected SVGD (eigenvector-informed directions)

**Any strategy works with any posterior method.** The factory (`build_learner()`) assembles components and returns a single `LatentActiveLearner` instance.

### BALD (Bayesian Active Learning by Disagreement)
Selects queries that maximize information gain about model parameters by finding points where posterior samples disagree most.

**Formula:** `BALD = H(E[p]) - E[H(p)]`

BALD auto-detects the posterior type and uses the appropriate variant (LatentBALD, EnsembleBALD, or ParticleBALD).

### Latent Space Learning
Perform active learning in a learned 32D latent space instead of high-dimensional parameter space, leveraging manifold structure.

---

## Quick Comparison

| Approach | Memory | Speed | Uncertainty | Best For |
|----------|--------|-------|-------------|----------|
| Standard VI | O(D) | Fast | Parametric | Prototyping |
| Full Cov VI | O(D²) | Medium | Full covariance | Correlated posteriors |
| Ensemble | O(K×D) | K× slower | Disagreement | Stable results |
| SVGD | O(K×D) | K× slower | Empirical | Complex posteriors |
| Sliced SVGD | O(K×D) | Faster than SVGD | Empirical | High-dim posteriors |
| **Projected SVGD** | O(K×D) | Fastest SVGD | Eigenvector-informed | **Recommended for SVGD** |

---

## Getting Started

```bash
# Run standard BALD with VI posterior (default)
python active_learning/test/diagnostics/run_latent_diagnosis.py --budget 40

# Run BALD with Ensemble posterior (K=5)
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy bald --posterior-method ensemble --ensemble-size 5 --budget 40

# Run Random with SVGD posterior (cross-combination example)
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy random --posterior-method svgd --n-particles 20 --budget 40

# NEW: Run with Projected SVGD (eigenvector-informed directions)
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy bald --posterior-method projected-svgd --n-particles 50 --budget 40

# NEW: Run Multi-Stage Warmup with adaptive stopping
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy multi-stage-warmup --posterior-method projected-svgd --budget 60

# Run test suite (119+ tests including cross-combination tests)
mamba run -n active_learning python -m pytest active_learning/test/test_refactored_pipeline.py -v

# Run hyperparameter search for Projected SVGD
python -m active_learning.test.param_search.projected_svgd_search --n-trials 50 --budget 20

# Profile pipeline performance
python active_learning/test/profile_pipeline.py --iterations 10
```

**Next steps:**
1. Read [QUICK_START.md](QUICK_START.md) for detailed usage and experiments
2. Explore [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details
3. Reference [VISUAL_REFERENCE.md](VISUAL_REFERENCE.md) for diagrams

---

## Documentation Maintenance

### When to Update
- Adding new acquisition strategies (update ARCHITECTURE.md Section 5 + 13)
- Adding new VI methods (update ARCHITECTURE.md Section 6 + 13)
- Changing core APIs (update all docs)
- Adding new config options (update ARCHITECTURE.md Section 8)
- Adding tests (update ARCHITECTURE.md Section 14)

### What to Update
- **QUICK_START.md:** If basic usage or architecture changes
- **ARCHITECTURE.md:** If implementation details change
- **VISUAL_REFERENCE.md:** If architecture or data flow changes
- **This README:** If documentation structure changes

---

**Documentation Version:** 3.0
**Last Updated:** January 28, 2026

---

## What's New in Version 3.0

### New Posterior Methods
- **Projected SVGD** (`projected_svgd`): Eigenvector-informed SVGD that captures high-variance directions via score covariance eigendecomposition. Combines projected directions with random slices for efficient high-dimensional inference.
- **Sliced SVGD** (`sliced_svgd`): 1D projection-based SVGD for faster computation in high dimensions.

### New Acquisition Strategies
- **Multi-Stage Warmup** (`multi_stage_warmup`): Adaptive boundary discovery with entropy-based stopping. Iteratively queries the current posterior's decision boundary, then switches to BALD when the boundary is well-discovered.

### New Test Scripts
- `test/param_search/projected_svgd_search.py`: Optuna hyperparameter search for Projected SVGD
- `test/profile_pipeline.py`: Performance profiling with timing reports
- `test/profile_pipeline_detailed.py`: Detailed component-level profiling
- `test/test_batched_bald.py`: BALD batching performance tests

### Performance Improvements
- Vectorized Projected SVGD implementation (`_compute_1d_svgd_batched`)
- GPU-optimized farthest-point sampling in multi-stage warmup
- Batched level-set evaluation optimization plans (see GPU_OPTIMIZATION_PLAN.md)
