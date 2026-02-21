"""
Factory for constructing active learners based on configuration.

Single entry point: build_learner() constructs the appropriate learner
with the correct strategy, VI, and posterior(s).

Two orthogonal axes:
  - acquisition.strategy: test selection method (bald, random, gp, grid, etc.)
  - posterior.method: posterior inference method (vi, ensemble, svgd, sliced_svgd, projected_svgd)

Any strategy can be combined with any posterior method.
"""

import torch
import numpy as np
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.config import DEVICE
# print(DEVICE)

def _build_posterior(decoder, prior, posterior, config):
    """
    Build posterior based on config['posterior']['method'].

    Args:
        decoder: LevelSetDecoder model
        prior: Prior distribution
        posterior: Base posterior distribution (used for 'vi' method)
        config: Full config dict

    Returns:
        Posterior: single LatentUserDistribution, list (ensemble), or ParticleUserDistribution (svgd/sliced_svgd/projected_svgd)
    """
    method = config.get('posterior', {}).get('method', 'vi')

    if method == 'ensemble':
        ensemble_cfg = config.get('ensemble', {})
        K = ensemble_cfg.get('ensemble_size', 5)
        init_noise_std = ensemble_cfg.get('init_noise_std', config.get('prior', {}).get('mean_noise_std', 0.3))

        posteriors = []
        for _ in range(K):
            noise = torch.randn(prior.latent_dim, device=DEVICE) * init_noise_std
            p = LatentUserDistribution(
                latent_dim=prior.latent_dim,
                decoder=decoder,
                mean=(prior.mean + noise).clone(),
                log_std=prior.log_std.clone(),
                device=DEVICE
            )
            posteriors.append(p)
        return posteriors  # List[LatentUserDistribution]

    elif method in ('svgd', 'sliced_svgd', 'projected_svgd', 'projected_svn'):
        from active_learning.src.svgd.particle_user_distribution import ParticleUserDistribution
        n_particles = config.get('posterior', {}).get('n_particles', 20)
        init_particles = prior.sample(n_particles)
        return ParticleUserDistribution(
            latent_dim=prior.latent_dim,
            decoder=decoder,
            n_particles=n_particles,
            init_particles=init_particles,
            device=DEVICE
        )

    elif method == 'full_cov_vi':
        # Upgrade posterior with full covariance initialized from prior
        if getattr(prior, 'cov_cholesky', None) is not None:
            posterior.cov_cholesky = prior.cov_cholesky.clone()
        else:
            posterior.cov_cholesky = torch.diag(torch.exp(prior.log_std)).clone().to(DEVICE)
        return posterior

    else:  # 'vi' (default)
        return posterior  # Pass through the single posterior


def _build_bald(decoder, prior, posterior, config):
    """
    Build the correct BALD variant based on posterior type.

    Auto-detects:
      - list of posteriors -> EnsembleBALD
      - posterior with get_particles -> ParticleBALD
      - single posterior -> LatentBALD
    """
    if isinstance(posterior, list):
        from active_learning.src.ensemble.ensemble_bald import EnsembleBALD
        return EnsembleBALD(decoder=decoder, posteriors=posterior, config=config, prior=prior)
    elif hasattr(posterior, 'get_particles'):
        from active_learning.src.svgd.particle_bald import ParticleBALD
        return ParticleBALD(decoder=decoder, posterior=posterior, config=config, prior=prior)
    else:
        from active_learning.src.latent_bald import LatentBALD
        return LatentBALD(decoder=decoder, posterior=posterior, config=config, prior=prior)


def _build_strategy(decoder, prior, posterior, config, bounds, oracle=None, embeddings=None):
    """
    Build acquisition strategy from config.

    Args:
        decoder: LevelSetDecoder model
        prior: Prior distribution
        posterior: Posterior distribution (single, list, or particle)
        config: Full config dict
        bounds: Tensor (n_joints, 2)
        oracle: Optional oracle (needed for some baselines)
        embeddings: Optional embeddings tensor (needed for version_space, heuristic)

    Returns:
        strategy object with select_test() method
    """
    acq_config = config.get('acquisition', {})
    strategy_type = acq_config.get('strategy', 'bald')

    if strategy_type == 'bald':
        return _build_bald(decoder, prior, posterior, config)

    elif strategy_type == 'random':
        from active_learning.src.baselines.random_strategy import RandomStrategy
        return RandomStrategy(config=config)

    elif strategy_type in ('quasi_random', 'quasi-random'):
        from active_learning.src.baselines.quasi_random_strategy import QuasiRandomStrategy
        bald = _build_bald(decoder, prior, posterior, config)
        n_quasi = acq_config.get('n_quasi_random', 10)
        return QuasiRandomStrategy(bald_strategy=bald, n_quasi_random=n_quasi, device=DEVICE)

    elif strategy_type in ('prior_boundary', 'prior-boundary'):
        from active_learning.src.baselines.prior_boundary_strategy import PriorBoundaryStrategy
        bald = _build_bald(decoder, prior, posterior, config)
        n_warmup = acq_config.get('n_prior_boundary', 10)
        return PriorBoundaryStrategy(
            decoder=decoder, prior=prior, bald_strategy=bald,
            n_warmup=n_warmup, device=DEVICE
        )

    elif strategy_type in ('multi_stage_warmup', 'multi-stage-warmup'):
        from active_learning.src.baselines.multi_stage_warmup_strategy import MultiStageWarmupStrategy
        bald = _build_bald(decoder, prior, posterior, config)
        msw_config = acq_config.get('multi_stage_warmup', {})
        return MultiStageWarmupStrategy(
            decoder=decoder,
            posterior=posterior,
            bald_strategy=bald,
            n_stages=msw_config.get('n_stages', 10),
            queries_per_stage=msw_config.get('queries_per_stage', 5),
            n_candidates=msw_config.get('n_candidates', 5000),
            boundary_percentile=msw_config.get('boundary_percentile', 0.05),
            use_farthest_point=msw_config.get('use_farthest_point', True),
            final_phase=msw_config.get('final_phase', 'bald'),
            n_mc_samples=config.get('bald', {}).get('n_mc_samples', 50),
            adaptive_stopping=msw_config.get('adaptive_stopping', True),
            entropy_threshold=msw_config.get('entropy_threshold', 0.9),
            window_size=msw_config.get('window_size', 15),
            min_warmup_queries=msw_config.get('min_warmup_queries', 10),
            device=DEVICE
        )

    elif strategy_type in ('canonical', 'hybrid'):
        from active_learning.src.canonical_acquisition import CanonicalAcquisition
        bald = _build_bald(decoder, prior, posterior, config)
        return CanonicalAcquisition(
            bald_strategy=bald,
            canonical_path=acq_config.get('canonical_path', 'models/canonical_queries.npz'),
            n_canonical=acq_config.get('n_canonical', 5),
            device=DEVICE
        )

    elif strategy_type == 'gbald':
        from active_learning.src.gbald import GeometricBALD
        return GeometricBALD(decoder=decoder, posterior=posterior, config=config)

    elif strategy_type == 'kbald':
        from active_learning.src.kbald import KBaldStrategy
        return KBaldStrategy(decoder=decoder, posterior=posterior, config=config)

    elif strategy_type == 'gp':
        from active_learning.src.baselines.gp_strategy import GPStrategy
        joint_limits = _get_joint_limits_dict(config)
        return GPStrategy(joint_limits=joint_limits, n_candidates=acq_config.get('gp_n_candidates', 5000))

    elif strategy_type == 'grid':
        from active_learning.src.baselines.grid_strategy import GridStrategy
        joint_limits = _get_joint_limits_dict(config)
        resolution = acq_config.get('grid_strategy_resolution', 16)
        return GridStrategy(joint_limits=joint_limits, resolution=resolution)

    elif strategy_type == 'heuristic':
        from active_learning.src.baselines.heuristic_strategy import HeuristicStrategy
        if embeddings is None:
            raise ValueError("Heuristic strategy requires embeddings parameter")
        from infer_params.training.level_set_torch import create_evaluation_grid
        queries = create_evaluation_grid(
            bounds[:, 0], bounds[:, 1],
            resolution=acq_config.get('heuristic_resolution', 10),
            device=DEVICE
        )
        return HeuristicStrategy(
            candidates=embeddings,
            queries=queries,
            checker=decoder,
            device=DEVICE
        )

    elif strategy_type == 'version_space':
        from active_learning.src.baselines.version_space_strategy import VersionSpaceStrategy
        if embeddings is None:
            raise ValueError("Version space strategy requires embeddings parameter")
        from infer_params.training.level_set_torch import create_evaluation_grid
        queries = create_evaluation_grid(
            bounds[:, 0], bounds[:, 1],
            resolution=acq_config.get('version_space_resolution', 10),
            device=DEVICE
        )
        return VersionSpaceStrategy(
            embeddings=embeddings,
            queries=queries,
            decoder=decoder,
            device=DEVICE
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_type}. "
                         f"Available: bald, gbald, kbald, random, quasi_random, prior_boundary, "
                         f"multi_stage_warmup, canonical, gp, grid, heuristic, version_space")


def _get_joint_limits_dict(config):
    """Extract joint limits as a dict from config."""
    prior_cfg = config.get('prior', {})
    joint_names = prior_cfg.get('joint_names', [])
    anat_limits = prior_cfg.get('anatomical_limits', {})
    units = prior_cfg.get('units', 'radians')
    result = {}
    for name in joint_names:
        lo, hi = anat_limits[name]
        if units == 'degrees':
            lo, hi = np.deg2rad(lo), np.deg2rad(hi)
        result[name] = (lo, hi)
    return result


def _build_vi(decoder, prior, posterior, config):
    """
    Build VI optimizer(s) based on posterior method.

    Auto-detects:
      - svgd -> SVGDVariationalInference
      - sliced_svgd -> SlicedSVGDVariationalInference
      - ensemble (list) -> list of LatentVariationalInference
      - vi (default) -> single LatentVariationalInference
    """
    method = config.get('posterior', {}).get('method', 'vi')

    if method == 'svgd':
        from active_learning.src.svgd.svgd_vi import SVGDVariationalInference
        return SVGDVariationalInference(decoder=decoder, prior=prior, posterior=posterior, config=config)
    elif method == 'sliced_svgd':
        from active_learning.src.svgd.sliced_svgd_vi import SlicedSVGDVariationalInference
        return SlicedSVGDVariationalInference(decoder=decoder, prior=prior, posterior=posterior, config=config)
    elif method == 'projected_svgd':
        from active_learning.src.svgd.projected_svgd_vi import ProjectedSVGDVariationalInference
        return ProjectedSVGDVariationalInference(decoder=decoder, prior=prior, posterior=posterior, config=config)
    elif method == 'projected_svn':
        from active_learning.src.svgd.projected_svn_vi import ProjectedSVNVariationalInference
        return ProjectedSVNVariationalInference(decoder=decoder, prior=prior, posterior=posterior, config=config)
    elif method == 'ensemble':
        from active_learning.src.latent_variational_inference import LatentVariationalInference
        return [LatentVariationalInference(decoder=decoder, prior=prior, posterior=p, config=config)
                for p in posterior]
    else:
        from active_learning.src.latent_variational_inference import LatentVariationalInference
        return LatentVariationalInference(decoder=decoder, prior=prior, posterior=posterior, config=config)


def build_learner(decoder, prior, posterior, oracle, bounds, config, embeddings=None):
    """
    Build an active learner based on config.

    Two orthogonal config axes:
      - config['acquisition']['strategy']: test selection method
      - config['posterior']['method']: posterior inference method (vi, ensemble, svgd, sliced_svgd, projected_svgd)

    Any strategy can be combined with any posterior method.

    Args:
        decoder: LevelSetDecoder model
        prior: Prior latent distribution
        posterior: Posterior latent distribution
        oracle: Oracle for ground truth queries
        bounds: Tensor (n_joints, 2)
        config: Full config dict
        embeddings: Optional embeddings (needed for heuristic, version_space)

    Returns:
        LatentActiveLearner instance
    """
    from active_learning.src.latent_active_learning import LatentActiveLearner

    # 1. Build posterior based on method
    posterior = _build_posterior(decoder, prior, posterior, config)

    # 2. Build strategy (auto-detects BALD variant for bald/quasi_random/canonical)
    strategy = _build_strategy(decoder, prior, posterior, config, bounds, oracle, embeddings)

    # 3. Build VI (auto-detects based on posterior method)
    vi = _build_vi(decoder, prior, posterior, config)

    # 4. Construct learner
    return LatentActiveLearner(
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        bounds=bounds,
        config=config,
        acquisition_strategy=strategy,
        vi=vi
    )
