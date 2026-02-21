# Visual Architecture Reference

Quick visual guides to understand code structure and data flow.

---

## 1. Active Learning Pipeline Flow

```
+---------------------------------------------------------------------+
|                        INITIALIZATION                                |
+---------------------------------------------------------------------+
|                                                                      |
|  Load Config --------> Load Decoder --------> Create Prior/Posterior |
|  (latent.yaml)         (best_model.pt)        (LatentUserDistrib.)   |
|                                                                      |
|  Create Oracle ------> Get Bounds ----------> build_learner()        |
|  (Ground Truth)        (Joint Limits)         (Factory)              |
|                                                                      |
+-----------------------------+----------------------------------------+
                              |
                              v
+---------------------------------------------------------------------+
|                    ACTIVE LEARNING LOOP                               |
|                    (learner.step())                                   |
+---------------------------------------------------------------------+
|                                                                      |
|  +---------------------------------------------------+              |
|  |  1. TEST SELECTION (Strategy)                     |              |
|  |                                                    |              |
|  |  Sample from Posterior: z ~ q(z)                  |              |
|  |            |                                       |              |
|  |  Pre-decode: z -> (lower, upper, weights, ...)    |              |
|  |            |                                       |              |
|  |  For each restart:                                |              |
|  |    - Random init test point                       |              |
|  |    - Gradient ascent on BALD score                |              |
|  |    - Adam -> SGD switching                        |              |
|  |            |                                       |              |
|  |  Return: test_point (argmax BALD)                |              |
|  +---------------------------------------------------+              |
|                          |                                           |
|                          v                                           |
|  +---------------------------------------------------+              |
|  |  2. ORACLE QUERY                                  |              |
|  |                                                    |              |
|  |  Evaluate: outcome = f(test_point; z_GT)          |              |
|  |         (signed distance, + = feasible)            |              |
|  |            |                                       |              |
|  |  Store: history.add(test_point, outcome)          |              |
|  +---------------------------------------------------+              |
|                          |                                           |
|                          v                                           |
|  +---------------------------------------------------+              |
|  |  2b. STRATEGY UPDATE (optional)                   |              |
|  |                                                    |              |
|  |  if hasattr(strategy, 'post_query_update'):       |              |
|  |      strategy.post_query_update(...)              |              |
|  |  (Used by GP, VersionSpace)                       |              |
|  +---------------------------------------------------+              |
|                          |                                           |
|                          v                                           |
|  +---------------------------------------------------+              |
|  |  3. POSTERIOR UPDATE (VI)                         |              |
|  |                                                    |              |
|  |  Standard:  Adam on (mu, log_std) to max ELBO    |              |
|  |  Ensemble:  K independent VI updates              |              |
|  |  SVGD:      Stein forces update particles         |              |
|  +---------------------------------------------------+              |
|                          |                                           |
|                          v                                           |
|              Check Stopping Criteria?                                |
|              Budget / BALD / ELBO plateau / Uncertainty              |
|                      |                                               |
|                  No  |  Yes                                          |
|              +-------+--------+                                      |
|              |                |                                       |
|          Continue         STOP                                       |
|              |                                                       |
+--------------+-------------------------------------------------------+
       |
       +-----------> Next Iteration
```

---

## 2. Unified Learner Architecture

```
+---------------------------------------------------------------------+
|                UNIFIED LEARNER (LatentActiveLearner)                  |
|                                                                      |
|  All 10 strategies use this single class.                            |
|  The factory injects the right components.                           |
+---------------------------------------------------------------------+

                      +-------------------------+
                      |  build_learner()         |
                      |  (factory.py)            |
                      +------------+------------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
          v                        v                        v
  +----------------+     +------------------+     +------------------+
  | Standard       |     | Ensemble         |     | SVGD             |
  | (bald, random, |     | (ensemble_bald)  |     | (svgd)           |
  |  gp, grid ...) |     |                  |     |                  |
  +----------------+     +------------------+     +------------------+
  | posterior:      |     | posterior:       |     | posterior:       |
  |   single LUD   |     |   List[LUD] (K)  |     |   ParticleUD    |
  | vi:             |     | vi:              |     | vi:              |
  |   LatentVI      |     |   List[LVI] (K)  |     |   SVGDVI        |
  | strategy:       |     | strategy:        |     | strategy:        |
  |   varies        |     |   EnsembleBALD   |     |   ParticleBALD  |
  +--------+--------+     +--------+---------+     +--------+--------+
           |                       |                        |
           +----------+------------+------------------------+
                      |
                      v
              +-------+--------+
              | LatentActive   |
              | Learner        |
              | (same class    |
              |  for all)      |
              +----------------+
              | _is_ensemble   |
              |   = isinstance |
              |   (posterior,  |
              |    list)       |
              +----------------+

LUD = LatentUserDistribution
LVI = LatentVariationalInference
```

---

## 3. Component Composition

```
         LatentActiveLearner
                |
                |-- decoder: LevelSetDecoder (external)
                |
                |-- prior: LatentUserDistribution
                |    |-- mean: Tensor (D,)
                |    |-- log_std: Tensor (D,)
                |    +-- sample(N) -> Tensor (N,D)
                |
                |-- posterior: LatentUserDistribution | List | Particle
                |    |-- mean: Tensor (D,) [optimized]
                |    |-- log_std: Tensor (D,) [optimized]
                |    +-- sample(N) -> Tensor (N,D)
                |
                |-- oracle: LatentOracle
                |    |-- ground_truth_z: Tensor (D,)
                |    |-- query(test) -> float
                |    +-- history: TestHistory
                |         +-- results: List[TestResult]
                |
                |-- strategy: (any of 10 strategies)
                |    |-- select_test(bounds) -> (test, score[, stats])
                |    +-- post_query_update(...) [optional, for GP/VersionSpace]
                |
                |-- vi: LatentVI | List[LatentVI] | SVGDVI
                |    |-- likelihood(history) -> Tensor
                |    |-- regularizer(kl_weight) -> Tensor
                |    +-- update_posterior(...) -> VIResult
                |
                |-- diagnostics: Diagnostics
                |    |-- history: List[DiagnosticSnapshot]
                |    +-- log_iteration(...)
                |
                |-- bounds: Tensor (J, 2)
                |-- results: List[LatentIterationResult]
                +-- config: dict
```

---

## 4. BALD Score Computation Flow

```
  Input: test_point (J,)
    |
    v
+------------------------+
| Sample from Posterior  |
| zs = posterior.sample  |
|      (N, D)            |
+--------+---------------+
         |
         v
+-------------------------------+
| Decode Latent -> RBF Params   |
| (lower, upper, weights, ...)  |
| Pre-decode once for efficiency|
+--------+----------------------+
         |
         v
+---------------------------------+
| Evaluate Level Set Function     |
| logits = f(test_point; zs)      |
|        shape: (N,)              |
+--------+------------------------+
         |
         v
+---------------------------------+
| Apply Temperature Scaling       |
| logits /= tau                   |
+--------+------------------------+
         |
         v
+---------------------------------+
| Convert to Probabilities        |
| probs = sigmoid(logits)         |
|       shape: (N,)               |
+--------+------------------------+
         |
         +------------------+------------------+
         |                  |                  |
         v                  v                  v
    +---------+       +---------+        +----------+
    | p_mean  |       | H(p_i)  |        | Gate?    |
    | mean(   |       | entropy |        | (if      |
    |  probs) |       | per     |        | weighted |
    |         |       | sample  |        |  BALD)   |
    +----+----+       +----+----+        +-----+----+
         |                 |                   |
         v                 v                   |
    +---------+       +---------+              |
    | H(p_bar)|       | E[H(p)] |              |
    | entropy |       | mean of |              |
    | of mean |       | entrop. |              |
    +----+----+       +----+----+              |
         |                 |                   |
         +--------+--------+                   |
                  |                            |
                  v                            |
         +------------------+                  |
         | BALD = H(p_bar)  |                  |
         |       - E[H(p)]  |                  |
         +--------+---------+                  |
                  |                            |
                  |<---------------------------+
                  |   * gate (if enabled)
                  v
            +------------+
            | BALD Score |
            |  (scalar)  |
            +------------+
```

---

## 5. Variational Inference Update Flow

### Standard VI (ELBO Maximization)

```
+---------------------------------------------------------------------+
|            update_posterior(test_history, kl_weight)                  |
+---------------------------------------------------------------------+

  Parameters: mu, log_sigma (requires_grad=True)
  Optimizer: Adam([mu, log_sigma], lr)
    |
    v
+=====================================================+
|          FOR iter in 1..max_iters:                   |
+=====================================================+
    |
    |-> [1] SAMPLE
    |   z ~ N(mu, exp(2*log_sigma))  via reparameterization
    |
    |-> [2] LIKELIHOOD
    |   LL = Sum_{i} log p(y_i | z, x_i)
    |
    |-> [3] KL DIVERGENCE
    |   KL = KL(N(mu,sigma) || N(mu_prior, sigma_prior))
    |
    |-> [4] ELBO = LL - w*KL
    |
    |-> [5] BACKPROP: loss = -ELBO
    |
    |-> [6] GRADIENT CLIP (optional)
    |
    |-> [7] OPTIMIZER STEP
    |
    |-> [8] CLAMP: log_sigma >= log(min_std)
    |
    +-> [9] CHECK CONVERGENCE
        if ELBO improvement < tol for patience iters: BREAK
    |
    v
  Return: VIResult(converged, n_iters, final_elbo, ...)
```

### SVGD (Stein Forces)

```
+---------------------------------------------------------------------+
|               update_posterior(test_history, ...)                     |
+---------------------------------------------------------------------+

  Particles: X = [x_1, ..., x_K] shape (K, D)
    |
    v
+=====================================================+
|          FOR iter in 1..max_iters:                   |
+=====================================================+
    |
    |-> [1] DETACH & ENABLE GRAD
    |
    |-> [2] LOG LIKELIHOOD (per particle)
    |   LL[k] = Sum_i log p(y_i | x_k, x_i)
    |
    |-> [3] LOG PRIOR (per particle)
    |   LP[k] = log N(x_k; mu_prior, sigma_prior)
    |
    |-> [4] LOG JOINT = LL + LP
    |
    |-> [5] BACKWARD -> grad_log_p
    |
    |-> [6] SVGD STEP
    |       RBF kernel + attraction + repulsion
    |
    +-> [7] UPDATE: X <- X + step_size * phi
    |
    v
  Return: SVGDVIResult(...)
```

---

## 6. Ensemble Architecture

```
+---------------------------------------------------------------------+
|          LatentActiveLearner (_is_ensemble = True)                    |
+---------------------------------------------------------------------+

        +--------------------------------------------+
        |  Factory: build_learner(ensemble_bald)     |
        +--------------------------------------------+
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
   Member 1        Member 2   ...   Member K
        |               |               |
        |-- posterior_1  |-- posterior_2  |-- posterior_K
        |  (mu_1, s_1)  |  (mu_2, s_2)  |  (mu_K, s_K)
        |               |               |
        |  Init: mu_k = mu_prior + N(0, noise_std)
        |
        |-- vi_1         |-- vi_2         |-- vi_K
        |  (LatentVI)    |  (LatentVI)    |  (LatentVI)
        |               |               |
        +---------------+---------------+
                        |
                        v
              +------------------+
              | EnsembleBALD     |
              | (uses all K)     |
              +------------------+

+=====================================================+
|                  step() Flow                         |
+=====================================================+

  [1] EnsembleBALD.select_test()
      |
      |-- For each member k:
      |   - Sample: z_k ~ N(mu_k, sigma_k)
      |   - Decode: params_k = decode(z_k)
      |   - Predict: p_k = mean(sigmoid(f(test; z_k)))
      |
      +-- BALD = H(mean(p_k)) - mean(H(p_k))
             = Disagreement between members

  [2] Oracle.query(test_point)

  [3] Update ALL members INDEPENDENTLY
      |-- vi_1.update_posterior(history, kl_weight)
      |-- vi_2.update_posterior(history, kl_weight)
      +-- vi_K.update_posterior(history, kl_weight)

  [4] Log Diagnostics (best ELBO member as representative)
```

---

## 7. Data Flow: Select -> Query -> Update

```
  State[N]:
    posterior: (mu_N, sigma_N)
    history: [test_1,...,test_{N-1}]

         |
         | select_test()
         v
  Candidates: R random inits x I grad steps
         |
         | argmax BALD
         v
  test_N: (J,)
  BALD_N: scalar

         |
         | oracle.query()
         v
  outcome_N: float

         |
         | history.add()
         v
  history: [test_1,...,test_N]

         |
         | vi.update_posterior()
         v
  State[N+1]:
    posterior: (mu_{N+1}, sigma_{N+1})  [updated]

         |
         v
  LatentIterationResult:
    - iteration: N
    - test_point: test_N
    - outcome: outcome_N
    - bald_score: BALD_N
    - elbo: ELBO_{N+1}
    - grad_norm: ||grad_ELBO||
    - vi_converged: bool
```

---

## 8. Config to Component Mapping

```
acquisition:
  strategy: "bald"  ----------------> Factory routes to:
                                       ALL strategies -> LatentActiveLearner
                                       (with appropriate injected components)

bald:
  tau: 1.0  ------------------------> LatentBALD.__init__(tau=1.0)
  n_mc_samples: 100  ---------------> posterior.sample(100)
  use_weighted_bald: false  --------> LatentBALD.compute_score() gate
  weighted_bald_sigma: 0.1  --------> Gate width parameter

bald_optimization:
  n_restarts: 10  ------------------> LatentBALD.select_test() loop count
  lr_adam: 0.05  -------------------> Adam optimizer learning rate
  switch_to_sgd_at: 0.8  -----------> Iteration fraction for optimizer switch

vi:
  learning_rate: 0.058  ------------> Adam([mu, log_sigma], lr=0.058)
  max_iters: 500  ------------------> VI optimization loop limit
  patience: 10  --------------------> Early stopping patience
  kl_annealing:
    start_weight: 0.286  -----------> Initial KL weight
    end_weight: 0.286  -------------> Final KL weight
    duration: 10  ------------------> Anneal over 10 iterations

ensemble:
  ensemble_size: 5  ----------------> K = 5 members (posteriors + VIs)
  init_noise_std: 0.4  ------------->  mu_k = mu_prior + N(0, 0.4)

posterior:
  n_particles: 50  ----------------> ParticleUserDistribution(n_particles=50)
svgd:
  step_size: 0.1  -----------------> particles += 0.1 * phi
  max_iters: 100  -----------------> SVGD optimization loop

stopping:
  budget: 20  ---------------------> learner.run(n_iterations=20)
```

---

## 9. File Import Graph

```
run_latent_diagnosis.py
    |
    |-> factory.py
    |   |
    |   +-> latent_active_learning.py (UNIFIED learner)
    |       |
    |       |-> latent_bald.py
    |       |   +-> latent_feasibility_checker.py
    |       |
    |       |-> latent_variational_inference.py
    |       |   +-> latent_feasibility_checker.py
    |       |
    |       |-> diagnostics.py
    |       +-> test_history.py
    |
    |   (Lazy imports based on strategy:)
    |   |
    |   |-> ensemble/ensemble_bald.py
    |   |   +-> latent_bald.py (extends)
    |   |
    |   |-> svgd/particle_bald.py
    |   |   +-> latent_bald.py (extends)
    |   |-> svgd/svgd_vi.py
    |   |   +-> svgd/svgd_optimizer.py
    |   |-> svgd/particle_user_distribution.py
    |   |
    |   |-> baselines/random_strategy.py
    |   |-> baselines/quasi_random_strategy.py
    |   |-> baselines/gp_strategy.py (requires sklearn)
    |   |-> baselines/grid_strategy.py
    |   |-> baselines/heuristic_strategy.py
    |   |-> baselines/version_space_strategy.py
    |   +-> canonical_acquisition.py
    |
    |-> config.py (load_config, get_bounds)
    |-> latent_prior_generation.py
    |   +-> latent_user_distribution.py
    |-> latent_oracle.py
    |   |-> latent_feasibility_checker.py
    |   +-> test_history.py
    +-> metrics.py
        +-> infer_params.training.level_set_torch
```

**Dependency Layers (bottom-up):**
```
Layer 0: torch, numpy, external models
         |
Layer 1: config, test_history, utils, latent_user_distribution
         |
Layer 2: latent_feasibility_checker, latent_oracle
         |
Layer 3: latent_bald, latent_variational_inference, diagnostics
         |            |                    |
         particle_bald              svgd_optimizer
         |                               |
         ensemble_bald              svgd_vi
         |
Layer 4: latent_active_learning (SINGLE unified learner)
         |
Layer 5: factory (assembles components, lazy imports)
         |
Layer 6: run_latent_diagnosis, test_refactored_pipeline
```

---

## 10. Strategy Extension Pattern

```
+---------------------------------------------------------------------+
|  How to add a new acquisition strategy                               |
+---------------------------------------------------------------------+

Step 1: Create Strategy Class
+-------------------------------+
| baselines/my_strategy.py      |
|-------------------------------|
| class MyStrategy:             |
|   def select_test(bounds,     |
|       **kwargs):              |
|       -> (test_point, score)  |
|                               |
|   def post_query_update(...)  |  <-- optional (for stateful)
|       -> None                 |
+-------------------------------+

Step 2: Register in Factory
+-------------------------------+
| factory.py                    |
|-------------------------------|
| def _build_strategy(...):     |
|   ...                         |
|   elif strategy_type ==       |
|       'my_strategy':          |
|     from ... import MyStrat.  |
|     return MyStrategy(config) |
+-------------------------------+

Step 3: Use via Config
+-------------------------------+
| configs/latent.yaml           |
|-------------------------------|
| acquisition:                  |
|   strategy: my_strategy       |
+-------------------------------+

         |
         | build_learner()
         v

+-------------------------------+
| LatentActiveLearner           |
|-------------------------------|
| strategy = MyStrategy(...)    |
| vi = LatentVI(...)            |
| posterior = single LUD        |
|                               |
| step() calls:                 |
|   strategy.select_test(...)   |
|   oracle.query(...)           |
|   strategy.post_query_update  |
|   vi.update_posterior(...)    |
+-------------------------------+
```

---

## 11. Typical Debug Flow

```
+-------------------------------------------------------------+
|             Symptom: IoU not improving                        |
+-------------------------------------------------------------+
         |
         v
    Check BALD scores
         |
    +----+----+
    |         |
    v         v
  High     Low/Zero
    |         |
    |         v
    |    +-----------------------------+
    |    | Posterior collapsed?         |
    |    | -> Check posterior.log_std   |
    |    | -> Increase min_std         |
    |    | -> Reduce kl_weight         |
    |    +-----------------------------+
    |
    v
Check VI convergence
    |
+---+---+
|       |
v       v
Good   Poor
|       |
|       v
|   +------------------------------+
|   | Check grad_norm              |
|   |                              |
|   | grad_norm < 0.01?            |
|   | -> Increase tau (2.0+)       |
|   | -> Increase learning_rate    |
|   |                              |
|   | grad_norm > 10?              |
|   | -> Enable grad_clip (1.0)    |
|   | -> Reduce learning_rate      |
|   +------------------------------+
|
v
Check query informativeness
    |
+---+---+
|       |
v       v
Near   Far from
bound  boundary
|       |
|       v
|   +------------------------------+
|   | Queries not reaching edge?   |
|   | -> Check bounds              |
|   | -> Increase BALD restarts    |
|   | -> Check oracle.query()      |
|   +------------------------------+
|
v
Deep dive into diagnostics
    - learner.diagnostics.print_final_report()
    - Check coverage over time
    - Check posterior movement
    - Visualize landscape plots
```

---

## 12. Quick Command Reference

```bash
# Standard pipeline
python active_learning/test/diagnostics/run_latent_diagnosis.py --budget 40 --seed 42

# Ensemble (K=5)
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy ensemble_bald --ensemble-size 5 --budget 40

# SVGD
python active_learning/test/diagnostics/run_latent_diagnosis.py \
    --strategy svgd --budget 40

# Run test suite
mamba run -n active_learning python -m pytest active_learning/test/test_refactored_pipeline.py -v
```

---

**For complete details, see:**
- **ARCHITECTURE.md** - Full theoretical and implementation details
- **QUICK_START.md** - Quick onboarding guide
- Source code inline documentation
