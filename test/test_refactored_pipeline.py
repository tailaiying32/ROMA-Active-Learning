"""
Comprehensive pytest test suite for the refactored active learning pipeline.

Verifies all functionality retained after refactoring: utilities, strategies,
factory, learner, ensemble, SVGD, config, diagnostics, and stopping criteria.

~112 tests covering the full latent active learning pipeline.
"""

import sys
import os
import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
from unittest.mock import patch

# ---------------------------------------------------------------------------
# sys.path setup (matches existing test conventions)
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Force CPU device for all tests
# ---------------------------------------------------------------------------
import active_learning.src.config as _cfg
_cfg.DEVICE = 'cpu'

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from active_learning.src.utils import (
    binary_entropy,
    calculate_kl_weight,
    get_adaptive_param,
    AcquisitionStrategy,
)
from active_learning.src.config import load_config, get_bounds_from_config, create_generator
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_bald import LatentBALD
from active_learning.src.latent_variational_inference import (
    LatentVariationalInference,
    LatentVIResult,
)
from active_learning.src.latent_active_learning import (
    LatentActiveLearner,
    LatentIterationResult,
)
from active_learning.src.diagnostics import Diagnostics, DiagnosticSnapshot
from active_learning.src.factory import build_learner
from active_learning.src.baselines.random_strategy import RandomStrategy
from active_learning.src.ensemble.ensemble_bald import EnsembleBALD

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    from active_learning.src.baselines.gp_strategy import GPStrategy
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from active_learning.src.baselines.quasi_random_strategy import QuasiRandomStrategy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from active_learning.src.baselines.grid_strategy import GridStrategy
    HAS_GRID = True
except ImportError:
    HAS_GRID = False

try:
    from active_learning.src.baselines.heuristic_strategy import HeuristicStrategy
    HAS_HEURISTIC = True
except ImportError:
    HAS_HEURISTIC = False

try:
    from active_learning.src.baselines.version_space_strategy import VersionSpaceStrategy
    HAS_VERSION_SPACE = True
except ImportError:
    HAS_VERSION_SPACE = False

try:
    from active_learning.src.svgd.particle_user_distribution import ParticleUserDistribution
    from active_learning.src.svgd.particle_bald import ParticleBALD
    from active_learning.src.svgd.svgd_vi import SVGDVariationalInference
    HAS_SVGD = True
except ImportError:
    HAS_SVGD = False

try:
    from active_learning.src.canonical_acquisition import CanonicalAcquisition
    HAS_CANONICAL = True
except ImportError:
    HAS_CANONICAL = False

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
LATENT_DIM = 8
N_JOINTS = 4
N_SLOTS = 18
PARAMS_PER_SLOT = 6
N_MC_SAMPLES = 5
BUDGET = 3
DEVICE = 'cpu'
SEED = 42

# Minimal values for speed
MINIMAL_N_RESTARTS = 2
MINIMAL_N_ITERS = 5
MINIMAL_VI_ITERS = 10


# ===========================================================================
# MockDecoder
# ===========================================================================
class MockDecoder(nn.Module):
    """
    A real nn.Module that mimics LevelSetDecoder.decode_from_embedding.

    Returns 5 tensors with correct shapes:
        lower:       (B, N_JOINTS)
        upper:       (B, N_JOINTS)
        weights:     (B, N_JOINTS)
        pres_logits: (B, N_SLOTS)
        blob_params: (B, N_SLOTS, PARAMS_PER_SLOT)

    Uses simple linear projections with softplus to guarantee upper > lower.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_slots = N_SLOTS
        self.params_per_slot = PARAMS_PER_SLOT

        # Linear projections
        self.center_proj = nn.Linear(latent_dim, N_JOINTS)
        self.span_proj = nn.Linear(latent_dim, N_JOINTS)
        self.weight_proj = nn.Linear(latent_dim, N_JOINTS)
        self.pres_proj = nn.Linear(latent_dim, N_SLOTS)
        self.blob_proj = nn.Linear(latent_dim, N_SLOTS * PARAMS_PER_SLOT)

    def decode_from_embedding(self, embedding: torch.Tensor):
        """
        Decode from embedding tensor.

        Args:
            embedding: (B, latent_dim)

        Returns:
            (lower, upper, weights, pres_logits, blob_params)
        """
        center = self.center_proj(embedding)            # (B, 4)
        span = F.softplus(self.span_proj(embedding))    # (B, 4) — positive
        lower = center - span / 2                       # (B, 4)
        upper = center + span / 2                       # (B, 4)
        weights = self.weight_proj(embedding)            # (B, 4)
        pres_logits = self.pres_proj(embedding)          # (B, 18)
        blob_params = self.blob_proj(embedding).view(
            -1, N_SLOTS, PARAMS_PER_SLOT
        )                                                # (B, 18, 6)
        return lower, upper, weights, pres_logits, blob_params


# ===========================================================================
# Minimal config dict
# ===========================================================================
def _make_minimal_config(strategy='bald', budget=BUDGET, seed=SEED):
    """Build a minimal config dict for testing."""
    return {
        'bald': {
            'tau': 1.0,
            'n_mc_samples': N_MC_SAMPLES,
            'sampling_temperature': 1.0,
            'epsilon': 0.0,
            'epsilon_decay': 0.95,
            'boundary_weight': 0.0,
            'use_weighted_bald': False,
            'weighted_bald_sigma': 0.1,
            'diversity_weight': 0.0,
            'use_anatomical_bounds': True,
        },
        'bald_optimization': {
            'n_restarts': MINIMAL_N_RESTARTS,
            'n_iters_per_restart': MINIMAL_N_ITERS,
            'lr_adam': 0.05,
            'lr_sgd': 0.01,
            'switch_to_sgd_at': 0.8,
        },
        'vi': {
            'n_mc_samples': N_MC_SAMPLES,
            'noise_std': 1.0,
            'learning_rate': 0.05,
            'max_iters': MINIMAL_VI_ITERS,
            'convergence_tol': 1e-4,
            'patience': 5,
            'grad_clip': 1.0,
            'kl_annealing': {
                'enabled': True,
                'start_weight': 0.1,
                'end_weight': 0.3,
                'duration': 10,
                'schedule': 'linear',
            },
        },
        'stopping': {
            'budget': budget,
            'bald_enabled': False,
            'bald_threshold': 0.1,
            'bald_patience': 3,
            'elbo_plateau_enabled': False,
            'elbo_plateau_window': 3,
            'elbo_plateau_threshold': 0.01,
            'uncertainty_enabled': False,
            'uncertainty_threshold': 0.02,
        },
        'acquisition': {
            'strategy': strategy,
            'n_canonical': 2,
            'canonical_path': 'models/canonical_queries.npz',
            'n_quasi_random': 2,
            'grid_strategy_resolution': 3,
        },
        'prior': {
            'mean_noise_std': 0.3,
            'init_std': 1.0,
            'joint_names': [f'joint_{i}' for i in range(N_JOINTS)],
            'units': 'radians',
            'anatomical_limits': {
                f'joint_{i}': [-1.5, 1.5] for i in range(N_JOINTS)
            },
        },
        'ensemble': {
            'ensemble_size': 3,
            'init_noise_std': 0.2,
        },
        'posterior': {
            'n_particles': 10,
        },
        'svgd': {
            'step_size': 0.1,
            'max_iters': 5,
        },
        'seed': seed,
    }


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def mock_decoder():
    decoder = MockDecoder(latent_dim=LATENT_DIM)
    decoder.eval()
    return decoder


@pytest.fixture
def mock_config():
    return _make_minimal_config()


@pytest.fixture
def mock_bounds():
    """(N_JOINTS, 2) tensor with [-1.5, 1.5] per joint."""
    return torch.tensor([[-1.5, 1.5]] * N_JOINTS, dtype=torch.float32)


@pytest.fixture
def mock_prior(mock_decoder):
    return LatentUserDistribution(
        latent_dim=LATENT_DIM,
        decoder=mock_decoder,
        mean=torch.zeros(LATENT_DIM),
        log_std=torch.zeros(LATENT_DIM),
        device=DEVICE,
    )


@pytest.fixture
def mock_posterior(mock_decoder, mock_prior):
    """Clone of prior."""
    return LatentUserDistribution(
        latent_dim=LATENT_DIM,
        decoder=mock_decoder,
        mean=mock_prior.mean.clone(),
        log_std=mock_prior.log_std.clone(),
        device=DEVICE,
    )


@pytest.fixture
def ground_truth_z():
    torch.manual_seed(SEED)
    return torch.randn(LATENT_DIM) * 0.5


@pytest.fixture
def mock_oracle(mock_decoder, ground_truth_z):
    return LatentOracle(mock_decoder, ground_truth_z, n_joints=N_JOINTS)


@pytest.fixture
def mock_embeddings():
    """(20, LATENT_DIM) embeddings for heuristic / version_space."""
    torch.manual_seed(SEED)
    return torch.randn(20, LATENT_DIM) * 0.5


@pytest.fixture
def canonical_file(tmp_path):
    """Temp .npz with points(5, N_JOINTS) and scores(5,)."""
    pts = np.random.default_rng(SEED).uniform(-1.5, 1.5, (5, N_JOINTS)).astype(np.float32)
    scores = np.zeros(5, dtype=np.float32)
    path = tmp_path / "canonical_queries.npz"
    np.savez(str(path), points=pts, scores=scores)
    return str(path)


@pytest.fixture
def mock_evaluation_grid(mock_bounds):
    """Simple cartesian_prod grid for heuristic/version_space."""
    linspaces = [torch.linspace(-1.5, 1.5, 4) for _ in range(N_JOINTS)]
    grid = torch.cartesian_prod(*linspaces)
    return grid


# ===========================================================================
# 1. TestUtilityFunctions (17 tests)
# ===========================================================================
class TestUtilityFunctions:
    """Tests for active_learning/src/utils.py."""

    # --- binary_entropy ---
    def test_binary_entropy_half(self):
        """H(0.5) = ln(2)."""
        p = torch.tensor(0.5)
        h = binary_entropy(p)
        assert torch.isclose(h, torch.tensor(math.log(2)), atol=1e-5)

    def test_binary_entropy_zero(self):
        """H(0) ≈ 0."""
        h = binary_entropy(torch.tensor(0.0))
        assert h.item() < 1e-3

    def test_binary_entropy_one(self):
        """H(1) ≈ 0."""
        h = binary_entropy(torch.tensor(1.0))
        assert h.item() < 1e-3

    def test_binary_entropy_shape(self):
        """Output shape matches input."""
        p = torch.rand(5, 3)
        assert binary_entropy(p).shape == (5, 3)

    def test_binary_entropy_symmetry(self):
        """H(p) = H(1-p)."""
        p = torch.tensor([0.2, 0.3, 0.7])
        assert torch.allclose(binary_entropy(p), binary_entropy(1 - p), atol=1e-5)

    def test_binary_entropy_nonneg(self):
        """Entropy is non-negative."""
        p = torch.rand(100)
        assert (binary_entropy(p) >= -1e-6).all()

    # --- calculate_kl_weight ---
    def test_kl_weight_disabled(self):
        cfg = {'enabled': False}
        assert calculate_kl_weight(5, cfg, 1.0) == 1.0

    def test_kl_weight_step_schedule(self):
        cfg = {'enabled': True, 'start_weight': 0.1, 'end_weight': 1.0, 'duration': 10, 'schedule': 'step'}
        assert calculate_kl_weight(3, cfg, 0.5) == pytest.approx(0.1)

    def test_kl_weight_linear_schedule(self):
        cfg = {'enabled': True, 'start_weight': 0.0, 'end_weight': 1.0, 'duration': 10, 'schedule': 'linear'}
        assert calculate_kl_weight(5, cfg, 0.5) == pytest.approx(0.5)

    def test_kl_weight_cosine_schedule(self):
        cfg = {'enabled': True, 'start_weight': 1.0, 'end_weight': 0.0, 'duration': 10, 'schedule': 'cosine'}
        w = calculate_kl_weight(5, cfg, 0.5)
        assert 0.0 <= w <= 1.0

    def test_kl_weight_logistic_schedule(self):
        cfg = {'enabled': True, 'start_weight': 0.0, 'end_weight': 1.0, 'duration': 10, 'schedule': 'logistic'}
        w = calculate_kl_weight(5, cfg, 0.5)
        assert 0.0 <= w <= 1.0

    def test_kl_weight_past_duration(self):
        cfg = {'enabled': True, 'start_weight': 0.0, 'end_weight': 1.0, 'duration': 10, 'schedule': 'linear'}
        assert calculate_kl_weight(15, cfg, 0.5) == pytest.approx(1.0)

    def test_kl_weight_iteration_zero(self):
        cfg = {'enabled': True, 'start_weight': 0.0, 'end_weight': 1.0, 'duration': 10, 'schedule': 'linear'}
        assert calculate_kl_weight(0, cfg, 0.5) == pytest.approx(0.0)

    # --- get_adaptive_param ---
    def test_adaptive_param_linear_interp(self):
        schedule = {'start': 2.0, 'end': 0.0, 'duration': 10, 'schedule': 'linear'}
        assert get_adaptive_param(schedule, 5, 99.0) == pytest.approx(1.0)

    def test_adaptive_param_none_schedule(self):
        assert get_adaptive_param(None, 5, 42.0) == 42.0

    def test_adaptive_param_none_iteration(self):
        schedule = {'start': 2.0, 'end': 0.0, 'duration': 10, 'schedule': 'linear'}
        assert get_adaptive_param(schedule, None, 42.0) == 42.0

    def test_adaptive_param_past_duration(self):
        schedule = {'start': 2.0, 'end': 0.5, 'duration': 10, 'schedule': 'linear'}
        assert get_adaptive_param(schedule, 20, 99.0) == pytest.approx(0.5)


# ===========================================================================
# 2. TestMockDecoder (6 tests)
# ===========================================================================
class TestMockDecoder:
    """Validates MockDecoder produces correct shapes and properties."""

    def test_output_shapes(self, mock_decoder):
        z = torch.randn(3, LATENT_DIM)
        lower, upper, weights, pres, blob = mock_decoder.decode_from_embedding(z)
        assert lower.shape == (3, N_JOINTS)
        assert upper.shape == (3, N_JOINTS)
        assert weights.shape == (3, N_JOINTS)
        assert pres.shape == (3, N_SLOTS)
        assert blob.shape == (3, N_SLOTS, PARAMS_PER_SLOT)

    def test_upper_gt_lower(self, mock_decoder):
        z = torch.randn(10, LATENT_DIM)
        lower, upper, _, _, _ = mock_decoder.decode_from_embedding(z)
        assert (upper > lower).all()

    def test_deterministic(self, mock_decoder):
        z = torch.randn(2, LATENT_DIM)
        out1 = mock_decoder.decode_from_embedding(z)
        out2 = mock_decoder.decode_from_embedding(z)
        for a, b in zip(out1, out2):
            assert torch.allclose(a, b)

    def test_gradient_flow(self, mock_decoder):
        z = torch.randn(2, LATENT_DIM, requires_grad=True)
        lower, upper, w, p, b = mock_decoder.decode_from_embedding(z)
        loss = lower.sum() + upper.sum()
        loss.backward()
        assert z.grad is not None

    def test_latent_dim_attribute(self, mock_decoder):
        assert mock_decoder.latent_dim == LATENT_DIM

    def test_single_sample(self, mock_decoder):
        z = torch.randn(1, LATENT_DIM)
        lower, upper, w, p, b = mock_decoder.decode_from_embedding(z)
        assert lower.shape == (1, N_JOINTS)


# ===========================================================================
# 3. TestLatentUserDistribution (4 tests)
# ===========================================================================
class TestLatentUserDistribution:
    """Tests for LatentUserDistribution."""

    def test_sample_shape(self, mock_decoder):
        dist = LatentUserDistribution(LATENT_DIM, mock_decoder, device=DEVICE)
        samples = dist.sample(10)
        assert samples.shape == (10, LATENT_DIM)

    def test_temperature_effect(self, mock_decoder):
        dist = LatentUserDistribution(LATENT_DIM, mock_decoder, device=DEVICE)
        g = torch.Generator(device=DEVICE).manual_seed(0)
        s1 = dist.sample(100, temperature=0.1, generator=g)
        g = torch.Generator(device=DEVICE).manual_seed(0)
        s2 = dist.sample(100, temperature=2.0, generator=g)
        # Higher temperature → larger std
        assert s2.std() > s1.std()

    def test_generator_reproducibility(self, mock_decoder):
        dist = LatentUserDistribution(LATENT_DIM, mock_decoder, device=DEVICE)
        g1 = torch.Generator(device=DEVICE).manual_seed(42)
        g2 = torch.Generator(device=DEVICE).manual_seed(42)
        s1 = dist.sample(5, generator=g1)
        s2 = dist.sample(5, generator=g2)
        assert torch.allclose(s1, s2)

    def test_default_init(self, mock_decoder):
        dist = LatentUserDistribution(LATENT_DIM, mock_decoder, device=DEVICE)
        assert torch.allclose(dist.mean, torch.zeros(LATENT_DIM))
        assert torch.allclose(dist.log_std, torch.zeros(LATENT_DIM))


# ===========================================================================
# 4. TestLatentFeasibilityChecker (6 tests)
# ===========================================================================
class TestLatentFeasibilityChecker:
    """Tests for LatentFeasibilityChecker."""

    def test_batched_logit_values_shape(self, mock_decoder, ground_truth_z):
        zs = ground_truth_z.unsqueeze(0).expand(3, -1)
        pts = torch.randn(5, N_JOINTS)
        logits = LatentFeasibilityChecker.batched_logit_values(mock_decoder, zs, pts)
        assert logits.shape == (3, 5)

    def test_single_point_shape(self, mock_decoder, ground_truth_z):
        zs = ground_truth_z.unsqueeze(0)
        pt = torch.randn(N_JOINTS)
        logits = LatentFeasibilityChecker.batched_logit_values(mock_decoder, zs, pt)
        # Single test point → (B, 1) or (B,)
        assert logits.dim() <= 2

    def test_decode_latent_params_tuple(self, mock_decoder, ground_truth_z):
        zs = ground_truth_z.unsqueeze(0)
        result = LatentFeasibilityChecker.decode_latent_params(mock_decoder, zs)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_evaluate_from_decoded(self, mock_decoder, ground_truth_z):
        zs = ground_truth_z.unsqueeze(0)
        decoded = LatentFeasibilityChecker.decode_latent_params(mock_decoder, zs)
        pts = torch.randn(5, N_JOINTS)
        logits = LatentFeasibilityChecker.evaluate_from_decoded(pts, decoded)
        assert logits.shape[1] == 5

    def test_logit_value_scalar(self, mock_decoder, ground_truth_z):
        checker = LatentFeasibilityChecker(mock_decoder, ground_truth_z, device=DEVICE)
        pt = torch.randn(N_JOINTS)
        logit = checker.logit_value(pt)
        assert logit.dim() == 0  # scalar

    def test_is_feasible_bool(self, mock_decoder, ground_truth_z):
        checker = LatentFeasibilityChecker(mock_decoder, ground_truth_z, device=DEVICE)
        pt = torch.randn(N_JOINTS)
        result = checker.is_feasible(pt)
        assert isinstance(result, (bool, torch.Tensor))


# ===========================================================================
# 5. TestStrategyInterface (13 tests)
# ===========================================================================
class TestStrategyInterface:
    """Tests that all strategies conform to the select_test interface."""

    def test_bald_strategy(self, mock_decoder, mock_posterior, mock_prior, mock_config, mock_bounds):
        strategy = LatentBALD(mock_decoder, mock_posterior, config=mock_config, prior=mock_prior)
        result = strategy.select_test(mock_bounds)
        assert len(result) == 3  # (test_point, score, diag_stats)
        assert result[0].shape == (N_JOINTS,)
        assert isinstance(result[1], (int, float))

    def test_random_strategy(self, mock_config, mock_bounds):
        strategy = RandomStrategy(config=mock_config)
        tp, score = strategy.select_test(mock_bounds)
        assert tp.shape == (N_JOINTS,)
        assert score == 0.0

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_gp_strategy(self, mock_config, mock_bounds):
        joint_limits = {f'joint_{i}': (-1.5, 1.5) for i in range(N_JOINTS)}
        strategy = GPStrategy(joint_limits=joint_limits, n_candidates=100)
        tp, score = strategy.select_test(mock_bounds)
        assert tp.shape == (N_JOINTS,)

    @pytest.mark.skipif(not HAS_GRID, reason="GridStrategy not available")
    def test_grid_strategy(self, mock_config, mock_bounds):
        joint_limits = {f'joint_{i}': (-1.5, 1.5) for i in range(N_JOINTS)}
        strategy = GridStrategy(joint_limits=joint_limits, resolution=3)
        tp, score = strategy.select_test(mock_bounds)
        assert tp.shape == (N_JOINTS,)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_quasi_random_strategy(self, mock_decoder, mock_posterior, mock_prior, mock_config, mock_bounds):
        bald = LatentBALD(mock_decoder, mock_posterior, config=mock_config, prior=mock_prior)
        strategy = QuasiRandomStrategy(bald_strategy=bald, n_quasi_random=2, device=DEVICE)
        tp, score = strategy.select_test(mock_bounds)
        assert tp.shape == (N_JOINTS,)
        assert score == 0.0  # First N are Sobol → score=0

    @pytest.mark.skipif(not HAS_CANONICAL, reason="CanonicalAcquisition not available")
    def test_canonical_strategy(self, mock_decoder, mock_posterior, mock_prior, mock_config, mock_bounds, canonical_file):
        bald = LatentBALD(mock_decoder, mock_posterior, config=mock_config, prior=mock_prior)
        strategy = CanonicalAcquisition(bald_strategy=bald, canonical_path=canonical_file, n_canonical=2, device=DEVICE)
        tp, score = strategy.select_test(mock_bounds)
        assert tp.shape == (N_JOINTS,)

    @pytest.mark.skipif(not HAS_HEURISTIC, reason="HeuristicStrategy not available")
    def test_heuristic_strategy(self, mock_decoder, mock_embeddings, mock_evaluation_grid, mock_bounds):
        strategy = HeuristicStrategy(
            candidates=mock_embeddings,
            queries=mock_evaluation_grid,
            checker=LatentFeasibilityChecker(mock_decoder, mock_embeddings[0]),
            device=DEVICE,
        )
        tp, score = strategy.select_test(mock_bounds)
        assert tp.shape == (N_JOINTS,)

    @pytest.mark.skipif(not HAS_VERSION_SPACE, reason="VersionSpaceStrategy not available")
    def test_version_space_strategy(self, mock_decoder, mock_embeddings, mock_evaluation_grid, mock_bounds):
        strategy = VersionSpaceStrategy(
            embeddings=mock_embeddings,
            queries=mock_evaluation_grid,
            decoder=mock_decoder,
            device=DEVICE,
        )
        tp, score = strategy.select_test(mock_bounds)
        assert tp.shape == (N_JOINTS,)

    def test_ensemble_bald_strategy(self, mock_decoder, mock_prior, mock_config, mock_bounds):
        posteriors = [
            LatentUserDistribution(LATENT_DIM, mock_decoder, device=DEVICE)
            for _ in range(3)
        ]
        strategy = EnsembleBALD(mock_decoder, posteriors=posteriors, config=mock_config, prior=mock_prior)
        result = strategy.select_test(mock_bounds)
        assert len(result) == 3
        assert result[0].shape == (N_JOINTS,)

    @pytest.mark.skipif(not HAS_SVGD, reason="SVGD not available")
    def test_svgd_strategy(self, mock_decoder, mock_posterior, mock_prior, mock_config, mock_bounds):
        particle_post = ParticleUserDistribution(
            LATENT_DIM, mock_decoder, n_particles=10, device=DEVICE
        )
        strategy = ParticleBALD(mock_decoder, posterior=particle_post, config=mock_config, prior=mock_prior)
        result = strategy.select_test(mock_bounds)
        assert result[0].shape == (N_JOINTS,)

    # --- post_query_update parametrized ---
    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_gp_post_query_update(self, mock_bounds):
        joint_limits = {f'joint_{i}': (-1.5, 1.5) for i in range(N_JOINTS)}
        strategy = GPStrategy(joint_limits=joint_limits, n_candidates=100)
        tp = torch.randn(N_JOINTS)
        strategy.post_query_update(tp, 1.0, None)
        assert len(strategy.X_train) == 1

    @pytest.mark.skipif(not HAS_VERSION_SPACE, reason="VersionSpaceStrategy not available")
    def test_version_space_post_query_update(self, mock_decoder, mock_embeddings, mock_evaluation_grid):
        strategy = VersionSpaceStrategy(
            embeddings=mock_embeddings,
            queries=mock_evaluation_grid,
            decoder=mock_decoder,
            device=DEVICE,
        )
        initial_valid = strategy.valid_mask.sum().item()
        tp = mock_evaluation_grid[0]
        strategy.post_query_update(tp, 1.0, None)
        # Some candidates may have been pruned
        assert strategy.valid_mask.sum().item() <= initial_valid

    def test_random_no_post_query_update(self):
        """RandomStrategy has no post_query_update — learner uses hasattr check."""
        strategy = RandomStrategy()
        # RandomStrategy does not define post_query_update, which is fine.
        # The learner checks hasattr before calling it.
        assert not hasattr(strategy, 'post_query_update')


# ===========================================================================
# 6. TestFactory (12 tests)
# ===========================================================================
class TestFactory:
    """Tests for build_learner factory."""

    def _build(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, config, embeddings=None):
        return build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, config, embeddings=embeddings)

    def test_bald(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('bald')
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert isinstance(learner.strategy, LatentBALD)

    def test_random(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random')
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert isinstance(learner.strategy, RandomStrategy)

    def test_ensemble_bald(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('bald')
        cfg['posterior'] = {'method': 'ensemble'}
        cfg['ensemble'] = {'ensemble_size': 3, 'init_noise_std': 0.2}
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert learner._is_ensemble
        assert isinstance(learner.posterior, list)
        assert len(learner.posterior) == 3
        assert isinstance(learner.vi, list)
        assert len(learner.vi) == 3

    @pytest.mark.skipif(not HAS_SVGD, reason="SVGD not available")
    def test_svgd(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('bald')
        cfg['posterior'] = {'method': 'svgd', 'n_particles': 10}
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert isinstance(learner.posterior, ParticleUserDistribution)

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_quasi_random(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('quasi_random')
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert isinstance(learner.strategy, QuasiRandomStrategy)

    @pytest.mark.skipif(not HAS_CANONICAL, reason="CanonicalAcquisition not available")
    def test_canonical(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, canonical_file):
        cfg = _make_minimal_config('canonical')
        cfg['acquisition']['canonical_path'] = canonical_file
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert isinstance(learner.strategy, CanonicalAcquisition)

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_gp(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('gp')
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert isinstance(learner.strategy, GPStrategy)

    @pytest.mark.skipif(not HAS_GRID, reason="GridStrategy not available")
    def test_grid(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('grid')
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert isinstance(learner.strategy, GridStrategy)

    @pytest.mark.skipif(not HAS_HEURISTIC, reason="HeuristicStrategy not available")
    def test_heuristic(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, mock_embeddings, mock_evaluation_grid, monkeypatch):
        cfg = _make_minimal_config('heuristic')
        # Monkeypatch create_evaluation_grid to avoid importing infer_params in factory
        def _mock_grid(lower, upper, resolution=10, device='cpu'):
            linspaces = [torch.linspace(l.item(), u.item(), resolution) for l, u in zip(lower, upper)]
            return torch.cartesian_prod(*linspaces)
        monkeypatch.setattr(
            'active_learning.src.factory.create_evaluation_grid' if False else None,
            None,
        ) if False else None
        # Since factory imports create_evaluation_grid at call time, monkeypatch that module
        import active_learning.src.factory as factory_mod
        # The factory imports inline: from infer_params.training.level_set_torch import create_evaluation_grid
        # We need to ensure it's patched. We'll monkeypatch at the module level where it's imported.
        import infer_params.training.level_set_torch as lst_mod
        monkeypatch.setattr(lst_mod, 'create_evaluation_grid', _mock_grid)
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg, embeddings=mock_embeddings)
        assert isinstance(learner.strategy, HeuristicStrategy)

    @pytest.mark.skipif(not HAS_VERSION_SPACE, reason="VersionSpaceStrategy not available")
    def test_version_space(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, mock_embeddings, mock_evaluation_grid, monkeypatch):
        cfg = _make_minimal_config('version_space')
        def _mock_grid(lower, upper, resolution=10, device='cpu'):
            linspaces = [torch.linspace(l.item(), u.item(), resolution) for l, u in zip(lower, upper)]
            return torch.cartesian_prod(*linspaces)
        import infer_params.training.level_set_torch as lst_mod
        monkeypatch.setattr(lst_mod, 'create_evaluation_grid', _mock_grid)
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg, embeddings=mock_embeddings)
        assert isinstance(learner.strategy, VersionSpaceStrategy)

    def test_unknown_strategy_raises(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('nonexistent')
        with pytest.raises(ValueError, match="Unknown strategy"):
            self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)

    def test_heuristic_without_embeddings_raises(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('heuristic')
        with pytest.raises(ValueError, match="embeddings"):
            self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg, embeddings=None)


# ===========================================================================
# 7. TestLearnerPipeline (10 tests)
# ===========================================================================
class TestLearnerPipeline:
    """Tests for LatentActiveLearner core pipeline."""

    def _make_learner(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, config=None):
        cfg = config or _make_minimal_config('random')
        return build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)

    def test_step_returns_result(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        result = learner.step()
        assert isinstance(result, LatentIterationResult)

    def test_result_field_types(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        r = learner.step()
        assert isinstance(r.iteration, int)
        assert isinstance(r.test_point, torch.Tensor)
        assert isinstance(r.outcome, float)
        assert isinstance(r.bald_score, float)

    def test_iteration_increments(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        r0 = learner.step()
        r1 = learner.step()
        assert r0.iteration == 0
        assert r1.iteration == 1

    def test_test_point_shape(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        r = learner.step()
        assert r.test_point.shape == (N_JOINTS,)

    def test_test_point_within_bounds(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        r = learner.step()
        assert (r.test_point >= mock_bounds[:, 0] - 1e-5).all()
        assert (r.test_point <= mock_bounds[:, 1] + 1e-5).all()

    def test_run_produces_exact_count(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random', budget=100)  # high budget so stopping won't trigger
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        results = learner.run(n_iterations=3, verbose=False)
        assert len(results) == 3

    def test_get_posterior(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        learner.step()
        post = learner.get_posterior()
        assert isinstance(post, LatentUserDistribution)

    def test_get_history_count(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        learner.step()
        learner.step()
        history = learner.get_history()
        assert len(history.get_all()) == 2

    def test_epsilon_greedy_forces_random(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('bald')
        cfg['bald']['epsilon'] = 1.0  # Always random
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        r = learner.step()
        # When epsilon=1.0, all steps are random → bald_score should be computed but selection is random
        assert isinstance(r.bald_score, float)

    def test_epsilon_decay_applied(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random')
        cfg['bald']['epsilon'] = 0.5
        cfg['bald']['epsilon_decay'] = 0.5
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        initial_eps = learner.epsilon
        learner.step()
        assert learner.epsilon < initial_eps
        assert learner.epsilon == pytest.approx(initial_eps * 0.5)


# ===========================================================================
# 8. TestStoppingCriteria (5 tests)
# ===========================================================================
class TestStoppingCriteria:
    """Tests for check_stopping_criteria."""

    def _make_learner(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg):
        return build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)

    def test_budget_reached(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random', budget=2)
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        learner.step()
        learner.step()
        should_stop, reason = learner.check_stopping_criteria()
        assert should_stop
        assert "Budget" in reason

    def test_bald_threshold_patience(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random', budget=100)
        cfg['stopping']['bald_enabled'] = True
        cfg['stopping']['bald_threshold'] = 999.0  # Very high → always below
        cfg['stopping']['bald_patience'] = 2
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        learner.step()
        learner.step()
        should_stop, reason = learner.check_stopping_criteria()
        assert should_stop
        assert "BALD" in reason

    def test_elbo_plateau(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random', budget=100)
        cfg['stopping']['elbo_plateau_enabled'] = True
        cfg['stopping']['elbo_plateau_window'] = 3
        cfg['stopping']['elbo_plateau_threshold'] = 1e10  # Very large → always plateau
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        for _ in range(3):
            learner.step()
        should_stop, reason = learner.check_stopping_criteria()
        assert should_stop
        assert "ELBO" in reason

    def test_uncertainty_threshold(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random', budget=100)
        cfg['stopping']['uncertainty_enabled'] = True
        cfg['stopping']['uncertainty_threshold'] = 999.0  # Very high → posterior std always below
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        learner.step()
        should_stop, reason = learner.check_stopping_criteria()
        assert should_stop
        assert "Uncertainty" in reason

    def test_no_early_stopping(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random', budget=100)
        # All early stopping disabled (default)
        learner = self._make_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        learner.step()
        should_stop, reason = learner.check_stopping_criteria()
        assert not should_stop


# ===========================================================================
# 9. TestEnsembleSupport (8 tests)
# ===========================================================================
class TestEnsembleSupport:
    """Tests for ensemble_bald via build_learner."""

    def _make_ensemble(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('bald')
        cfg['posterior'] = {'method': 'ensemble'}
        cfg['ensemble'] = {'ensemble_size': 3, 'init_noise_std': 0.2}
        return build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)

    def test_is_ensemble_flag(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_ensemble(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        assert learner._is_ensemble

    def test_k_posteriors(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_ensemble(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        assert len(learner.posterior) == 3

    def test_k_vi_optimizers(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_ensemble(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        assert isinstance(learner.vi, list)
        assert len(learner.vi) == 3

    def test_step_completes(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_ensemble(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        r = learner.step()
        assert isinstance(r, LatentIterationResult)

    def test_get_posterior_returns_single(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_ensemble(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        learner.step()
        post = learner.get_posterior()
        assert isinstance(post, LatentUserDistribution)
        assert not isinstance(post, list)

    def test_ensemble_predict_probs_shape(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_ensemble(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        learner.step()
        pts = torch.randn(5, N_JOINTS)
        probs = learner.ensemble_predict_probs(pts)
        assert probs.shape == (5,)

    def test_ensemble_predict_probs_range(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._make_ensemble(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds)
        learner.step()
        pts = torch.randn(5, N_JOINTS)
        probs = learner.ensemble_predict_probs(pts)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_non_ensemble_raises_for_predict_probs(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random')
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        with pytest.raises(ValueError, match="ensemble"):
            learner.ensemble_predict_probs(torch.randn(5, N_JOINTS))


# ===========================================================================
# 10. TestSVGDSupport (4 tests)
# ===========================================================================
@pytest.mark.skipif(not HAS_SVGD, reason="SVGD modules not available")
class TestSVGDSupport:
    """Tests for SVGD via build_learner."""

    def test_particle_posterior(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('bald')
        cfg['posterior'] = {'method': 'svgd', 'n_particles': 10}
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert isinstance(learner.posterior, ParticleUserDistribution)

    def test_correct_particle_count(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('bald')
        cfg['posterior'] = {'method': 'svgd', 'n_particles': 10}
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert learner.posterior.n_particles == 10

    def test_step_completes(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('bald')
        cfg['posterior'] = {'method': 'svgd', 'n_particles': 10}
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        r = learner.step()
        assert isinstance(r, LatentIterationResult)

    def test_particle_sample_shape(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('bald')
        cfg['posterior'] = {'method': 'svgd', 'n_particles': 10}
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        samples = learner.posterior.sample(5)
        assert samples.shape == (5, LATENT_DIM)


# ===========================================================================
# 11. TestConfigIntegration (4 tests)
# ===========================================================================
class TestConfigIntegration:
    """Tests for config loading and bounds extraction."""

    def test_load_config_returns_dict(self):
        yaml_path = os.path.join(project_root, 'active_learning', 'configs', 'latent.yaml')
        if os.path.exists(yaml_path):
            cfg = load_config(yaml_path)
            assert isinstance(cfg, dict)
            assert 'bald' in cfg
            assert 'vi' in cfg
            assert 'stopping' in cfg
        else:
            pytest.skip("latent.yaml not found")

    def test_get_bounds_shape(self):
        yaml_path = os.path.join(project_root, 'active_learning', 'configs', 'latent.yaml')
        if os.path.exists(yaml_path):
            cfg = load_config(yaml_path)
            bounds = get_bounds_from_config(cfg, device='cpu')
            n_joints = len(cfg.get('prior', {}).get('joint_names', []))
            assert bounds.shape == (n_joints, 2)
        else:
            pytest.skip("latent.yaml not found")

    def test_degrees_to_radians(self):
        cfg = {
            'prior': {
                'joint_names': ['j0'],
                'units': 'degrees',
                'anatomical_limits': {'j0': [-90, 90]},
            },
            'bald': {'use_anatomical_bounds': True},
        }
        bounds = get_bounds_from_config(cfg, device='cpu')
        assert bounds[0, 0].item() == pytest.approx(np.deg2rad(-90), abs=1e-5)
        assert bounds[0, 1].item() == pytest.approx(np.deg2rad(90), abs=1e-5)

    def test_strategy_override(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random')
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert isinstance(learner.strategy, RandomStrategy)
        # Override to bald
        cfg2 = _make_minimal_config('bald')
        learner2 = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg2)
        assert isinstance(learner2.strategy, LatentBALD)


# ===========================================================================
# 12. TestKLAnnealingIntegration (3 tests)
# ===========================================================================
class TestKLAnnealingIntegration:
    """Tests for KL annealing integration with the learner."""

    def test_weight_changes_across_iterations(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random')
        cfg['vi']['kl_annealing'] = {
            'enabled': True,
            'start_weight': 0.0,
            'end_weight': 1.0,
            'duration': 5,
            'schedule': 'linear',
        }
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        w0 = learner._calculate_kl_weight(0)
        w3 = learner._calculate_kl_weight(3)
        assert w3 > w0

    def test_different_values_at_0_vs_5(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random')
        cfg['vi']['kl_annealing'] = {
            'enabled': True,
            'start_weight': 0.1,
            'end_weight': 0.9,
            'duration': 10,
            'schedule': 'linear',
        }
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        w0 = learner._calculate_kl_weight(0)
        w5 = learner._calculate_kl_weight(5)
        assert w0 != w5

    def test_disabled_constant_weight(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random')
        cfg['vi']['kl_annealing'] = {'enabled': False}
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        w0 = learner._calculate_kl_weight(0)
        w5 = learner._calculate_kl_weight(5)
        assert w0 == w5


# ===========================================================================
# 13. TestDiagnosticsIntegration (3 tests)
# ===========================================================================
class TestDiagnosticsIntegration:
    """Tests for diagnostics creation and population."""

    def test_created_when_oracle_has_ground_truth(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random')
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert learner.diagnostics is not None

    def test_not_created_without_ground_truth(self, mock_decoder, mock_prior, mock_posterior, mock_bounds):
        """Oracle without ground_truth_z → no diagnostics."""

        class FakeOracle:
            def __init__(self):
                from active_learning.src.test_history import TestHistory
                self.history = TestHistory(['j0', 'j1', 'j2', 'j3'])

            def query(self, test_point):
                self.history.add(test_point, 0.5, h_value=0.5)
                return 0.5

            def get_history(self):
                return self.history

        cfg = _make_minimal_config('random')
        strategy = RandomStrategy(config=cfg)
        vi = LatentVariationalInference(mock_decoder, mock_prior, mock_posterior, config=cfg)
        learner = LatentActiveLearner(
            mock_decoder, mock_prior, mock_posterior, FakeOracle(), mock_bounds,
            config=cfg, acquisition_strategy=strategy, vi=vi
        )
        assert learner.diagnostics is None

    def test_step_populates_diagnostics(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        cfg = _make_minimal_config('random')
        learner = build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)
        assert learner.diagnostics is not None
        learner.step()
        assert len(learner.diagnostics.history) >= 1


# ===========================================================================
# 14. TestStrategySpecificBehavior (17 tests)
# ===========================================================================
class TestStrategySpecificBehavior:
    """Tests for strategy-specific behavior beyond the interface."""

    # --- Random ---
    def test_random_within_bounds(self, mock_bounds):
        strategy = RandomStrategy(config=_make_minimal_config())
        tp, _ = strategy.select_test(mock_bounds)
        assert (tp >= mock_bounds[:, 0] - 1e-5).all()
        assert (tp <= mock_bounds[:, 1] + 1e-5).all()

    def test_random_different_per_call(self, mock_bounds):
        """Two calls without seed should give different points (with high probability)."""
        strategy = RandomStrategy(config={'seed': None})
        tp1, _ = strategy.select_test(mock_bounds)
        tp2, _ = strategy.select_test(mock_bounds)
        # Very unlikely to be exactly equal with float64
        assert not torch.allclose(tp1, tp2)

    # --- GP ---
    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_gp_adds_to_x_train(self):
        joint_limits = {f'joint_{i}': (-1.5, 1.5) for i in range(N_JOINTS)}
        strategy = GPStrategy(joint_limits=joint_limits, n_candidates=100)
        tp = torch.randn(N_JOINTS)
        strategy.post_query_update(tp, 1.0, None)
        assert len(strategy.X_train) == 1
        strategy.post_query_update(tp, -1.0, None)
        assert len(strategy.X_train) == 2

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_gp_select_after_update(self, mock_bounds):
        joint_limits = {f'joint_{i}': (-1.5, 1.5) for i in range(N_JOINTS)}
        strategy = GPStrategy(joint_limits=joint_limits, n_candidates=100)
        # First select (random)
        tp1, _ = strategy.select_test(mock_bounds)
        # Update with observation
        strategy.post_query_update(tp1, 1.0, None)
        # Second select (GP-informed)
        tp2, _ = strategy.select_test(mock_bounds)
        assert tp2.shape == (N_JOINTS,)

    # --- Grid ---
    @pytest.mark.skipif(not HAS_GRID, reason="GridStrategy not available")
    def test_grid_returns_grid_point(self, mock_bounds):
        joint_limits = {f'joint_{i}': (-1.5, 1.5) for i in range(N_JOINTS)}
        strategy = GridStrategy(joint_limits=joint_limits, resolution=3)
        tp, _ = strategy.select_test(mock_bounds)
        assert tp.shape == (N_JOINTS,)

    @pytest.mark.skipif(not HAS_GRID, reason="GridStrategy not available")
    def test_grid_resolution_count(self):
        joint_limits = {f'joint_{i}': (-1.5, 1.5) for i in range(N_JOINTS)}
        strategy = GridStrategy(joint_limits=joint_limits, resolution=3)
        assert len(strategy.grid) == 3 ** N_JOINTS

    # --- VersionSpace ---
    @pytest.mark.skipif(not HAS_VERSION_SPACE, reason="VersionSpaceStrategy not available")
    def test_version_space_prunes(self, mock_decoder, mock_embeddings, mock_evaluation_grid):
        strategy = VersionSpaceStrategy(
            embeddings=mock_embeddings, queries=mock_evaluation_grid,
            decoder=mock_decoder, device=DEVICE,
        )
        initial = strategy.valid_mask.sum().item()
        tp = mock_evaluation_grid[0]
        strategy.update(tp, True)
        after = strategy.valid_mask.sum().item()
        assert after <= initial

    @pytest.mark.skipif(not HAS_VERSION_SPACE, reason="VersionSpaceStrategy not available")
    def test_version_space_returns_query(self, mock_decoder, mock_embeddings, mock_evaluation_grid, mock_bounds):
        strategy = VersionSpaceStrategy(
            embeddings=mock_embeddings, queries=mock_evaluation_grid,
            decoder=mock_decoder, device=DEVICE,
        )
        tp, score = strategy.select_test(mock_bounds)
        assert tp.shape == (N_JOINTS,)

    # --- QuasiRandom ---
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_quasi_random_first_n_sobol(self, mock_decoder, mock_posterior, mock_prior, mock_config, mock_bounds):
        bald = LatentBALD(mock_decoder, mock_posterior, config=mock_config, prior=mock_prior)
        strategy = QuasiRandomStrategy(bald_strategy=bald, n_quasi_random=3, device=DEVICE)
        tp, score = strategy.select_test(mock_bounds)
        assert score == 0.0  # Sobol phase
        assert strategy.in_exploration_phase

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_quasi_random_phase_property(self, mock_decoder, mock_posterior, mock_prior, mock_config, mock_bounds):
        bald = LatentBALD(mock_decoder, mock_posterior, config=mock_config, prior=mock_prior)
        strategy = QuasiRandomStrategy(bald_strategy=bald, n_quasi_random=1, device=DEVICE)
        assert strategy.phase_name == "Quasi-Random"
        strategy.select_test(mock_bounds)  # consume 1 Sobol point
        assert strategy.phase_name == "BALD"

    # --- Heuristic ---
    @pytest.mark.skipif(not HAS_HEURISTIC, reason="HeuristicStrategy not available")
    def test_heuristic_returns_from_pool(self, mock_decoder, mock_embeddings, mock_evaluation_grid, mock_bounds):
        checker = LatentFeasibilityChecker(mock_decoder, mock_embeddings[0])
        strategy = HeuristicStrategy(
            candidates=mock_embeddings, queries=mock_evaluation_grid,
            checker=checker, device=DEVICE,
        )
        tp, _ = strategy.select_test(mock_bounds)
        # The returned point should be from the evaluation grid
        distances = (mock_evaluation_grid - tp.unsqueeze(0)).norm(dim=1)
        assert distances.min().item() < 1e-5

    # --- Canonical ---
    @pytest.mark.skipif(not HAS_CANONICAL, reason="CanonicalAcquisition not available")
    def test_canonical_first_n(self, mock_decoder, mock_posterior, mock_prior, mock_config, mock_bounds, canonical_file):
        bald = LatentBALD(mock_decoder, mock_posterior, config=mock_config, prior=mock_prior)
        strategy = CanonicalAcquisition(bald_strategy=bald, canonical_path=canonical_file, n_canonical=2, device=DEVICE)
        tp1, _ = strategy.select_test(mock_bounds)
        assert strategy.in_exploration_phase  # Still have canonical left
        tp2, _ = strategy.select_test(mock_bounds)
        assert not strategy.in_exploration_phase  # Now in BALD phase

    @pytest.mark.skipif(not HAS_CANONICAL, reason="CanonicalAcquisition not available")
    def test_canonical_missing_file_fallback(self, mock_decoder, mock_posterior, mock_prior, mock_config, mock_bounds):
        bald = LatentBALD(mock_decoder, mock_posterior, config=mock_config, prior=mock_prior)
        strategy = CanonicalAcquisition(
            bald_strategy=bald, canonical_path='/nonexistent/path.npz',
            n_canonical=5, device=DEVICE
        )
        # Should fall back to BALD immediately
        assert not strategy.in_exploration_phase
        result = strategy.select_test(mock_bounds)
        # BALD select_test returns 3 values (test_point, score, diag_stats)
        tp = result[0]
        assert tp.shape == (N_JOINTS,)

    # --- BALD ---
    def test_bald_compute_score_scalar(self, mock_decoder, mock_posterior, mock_prior, mock_config):
        strategy = LatentBALD(mock_decoder, mock_posterior, config=mock_config, prior=mock_prior)
        tp = torch.randn(N_JOINTS)
        score = strategy.compute_score(tp)
        # Score is scalar or (1,) depending on internal unsqueezing
        assert score.numel() == 1

    def test_bald_score_nonneg(self, mock_decoder, mock_posterior, mock_prior, mock_config):
        strategy = LatentBALD(mock_decoder, mock_posterior, config=mock_config, prior=mock_prior)
        tp = torch.randn(N_JOINTS)
        score = strategy.compute_score(tp)
        assert score.item() >= -1e-5  # BALD is non-negative (up to numerical noise)


# ===========================================================================
# 15. TestCrossCombinations - Any strategy x any posterior method
# ===========================================================================
class TestCrossCombinations:
    """Test mixing any acquisition strategy with any posterior method."""

    def _build(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
               strategy, posterior_method):
        cfg = _make_minimal_config(strategy)
        cfg['posterior'] = {'method': posterior_method, 'n_particles': 5}
        cfg['ensemble'] = {'ensemble_size': 3, 'init_noise_std': 0.2}
        return build_learner(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds, cfg)

    def test_random_ensemble_builds(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
                              'random', 'ensemble')
        assert learner._is_ensemble
        assert isinstance(learner.posterior, list)
        assert len(learner.posterior) == 3
        assert isinstance(learner.vi, list)

    @pytest.mark.skipif(not HAS_SVGD, reason="SVGD not available")
    def test_random_svgd_builds(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
                              'random', 'svgd')
        assert hasattr(learner.posterior, 'get_particles')

    def test_random_ensemble_step(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        """Random selection + ensemble posterior produces valid result."""
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
                              'random', 'ensemble')
        result = learner.step()
        assert isinstance(result, LatentIterationResult)
        assert learner._is_ensemble
        assert isinstance(learner.vi, list)

    @pytest.mark.skipif(not HAS_SVGD, reason="SVGD not available")
    def test_random_svgd_step(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        """Random selection + SVGD posterior produces valid result."""
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
                              'random', 'svgd')
        result = learner.step()
        assert isinstance(result, LatentIterationResult)
        assert hasattr(learner.posterior, 'get_particles')

    @pytest.mark.skipif(not HAS_GRID, reason="GridStrategy not available")
    def test_grid_ensemble_builds(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
                              'grid', 'ensemble')
        assert learner._is_ensemble
        assert isinstance(learner.vi, list)

    @pytest.mark.skipif(not HAS_GRID or not HAS_SVGD, reason="GridStrategy or SVGD not available")
    def test_grid_svgd_builds(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
                              'grid', 'svgd')
        assert hasattr(learner.posterior, 'get_particles')

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
    def test_gp_ensemble_step(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        """GP selection + ensemble posterior."""
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
                              'gp', 'ensemble')
        # GPStrategy hardcodes device=cuda; force CPU for test consistency
        if hasattr(learner.strategy, 'device'):
            learner.strategy.device = torch.device('cpu')
        result = learner.step()
        assert isinstance(result, LatentIterationResult)
        assert learner._is_ensemble

    def test_bald_ensemble_step(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        """BALD selection + ensemble posterior."""
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
                              'bald', 'ensemble')
        result = learner.step()
        assert isinstance(result, LatentIterationResult)
        assert learner._is_ensemble

    @pytest.mark.skipif(not HAS_SVGD, reason="SVGD not available")
    def test_bald_svgd_step(self, mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds):
        """BALD selection + SVGD posterior."""
        learner = self._build(mock_decoder, mock_prior, mock_posterior, mock_oracle, mock_bounds,
                              'bald', 'svgd')
        result = learner.step()
        assert isinstance(result, LatentIterationResult)
        assert hasattr(learner.posterior, 'get_particles')
