"""
Shared utilities for the active learning pipeline.

Consolidates duplicated code from multiple modules into a single source of truth.
"""

import os
import torch
import numpy as np


def binary_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
    p = torch.clamp(p, eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def calculate_kl_weight(iteration: int, annealing_config: dict, default_kl_weight: float) -> float:
    """
    Calculate KL weight for the current iteration based on annealing schedule.

    Args:
        iteration: Current iteration index
        annealing_config: Dict with keys: enabled, start_weight, end_weight, duration, schedule
        default_kl_weight: Fallback KL weight when annealing is disabled

    Returns:
        KL weight as float
    """
    if not annealing_config.get('enabled', False):
        return default_kl_weight

    start = annealing_config.get('start_weight', 0.0)
    end = annealing_config.get('end_weight', default_kl_weight)
    duration = annealing_config.get('duration', 10)
    schedule = annealing_config.get('schedule', 'linear')

    if iteration >= duration:
        return end

    if schedule == 'step':
        return start

    elif schedule == 'linear':
        progress = iteration / float(duration)
        return start + (end - start) * progress

    elif schedule == 'cosine':
        progress = iteration / float(duration)
        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return start * cosine_factor + end * (1 - cosine_factor)

    elif schedule == 'logistic':
        k = 10.0 / duration
        x0 = duration / 2.0
        sigmoid = 1 / (1 + np.exp(-k * (iteration - x0)))
        return start + (end - start) * sigmoid

    elif schedule == 'exponential':
        # Exponential: faster initial change, slower near end
        # w(t) = start * (end/start)^(t/duration)
        if start <= 0:
            return start + (end - start) * progress  # Fallback to linear
        ratio = end / start
        return start * (ratio ** progress)

    return end


def get_adaptive_param(schedule: dict, iteration: int, default_val: float) -> float:
    """
    Compute a scheduled parameter value (linear interpolation from start to end).

    Args:
        schedule: Dict with keys: start, end, duration, schedule (type)
        iteration: Current iteration
        default_val: Fallback if schedule is None

    Returns:
        Interpolated parameter value
    """
    if not schedule or iteration is None:
        return default_val
    start = schedule.get('start', default_val)
    end = schedule.get('end', default_val)
    duration = schedule.get('duration', 1)
    sched_type = schedule.get('schedule', 'linear')
    if iteration >= duration:
        return end
    progress = iteration / float(duration)
    if sched_type == 'linear':
        return start + (end - start) * progress
    elif sched_type == 'cosine':
        cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return start * cosine_factor + end * (1 - cosine_factor)
    elif sched_type == 'exponential':
        # Exponential: faster initial change, slower near end
        if start <= 0:
            return start + (end - start) * progress  # Fallback to linear
        ratio = end / start
        return start * (ratio ** progress)
    return start + (end - start) * progress  # Default to linear


def load_decoder_model(checkpoint_path: str, device: str = None):
    """
    Load LevelSetDecoder from checkpoint.

    Args:
        checkpoint_path: Path to best_model.pt
        device: torch device string (defaults to config.DEVICE)

    Returns:
        (model, embeddings, train_config) tuple
    """
    if device is None:
        from active_learning.src.config import DEVICE
        device = DEVICE

    if not os.path.exists(checkpoint_path):
        # Try finding it relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        alt_path = os.path.join(project_root, checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading decoder model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    train_config = checkpoint['config']
    model_cfg = train_config['model']

    # Get num_samples from embeddings
    embeddings = checkpoint['embeddings']
    num_samples = embeddings.shape[0]
    latent_dim = model_cfg['latent_dim']

    print(f"  Num samples: {num_samples}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Hidden dim: {model_cfg['hidden_dim']}")
    print(f"  Num blocks: {model_cfg['num_blocks']}")

    from infer_params.training.model import LevelSetDecoder

    # Create model
    model = LevelSetDecoder(
        num_samples=num_samples,
        latent_dim=latent_dim,
        hidden_dim=model_cfg['hidden_dim'],
        num_blocks=model_cfg['num_blocks'],
        num_slots=model_cfg.get('num_slots', 18),
        params_per_slot=model_cfg.get('params_per_slot', 6),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Model loaded successfully (epoch {checkpoint['epoch']})")

    return model, embeddings.to(device), train_config


class AcquisitionStrategy:
    """Protocol for all acquisition strategies."""

    def select_test(self, bounds: torch.Tensor, **kwargs) -> tuple:
        """
        Select a test point.

        Args:
            bounds: Tensor (n_joints, 2) with [lower, upper]
            **kwargs: May include verbose, test_history, iteration, diagnostics

        Returns:
            Tuple of (test_point: Tensor, score: float) or
            (test_point: Tensor, score: float, diag_stats: list)
        """
        raise NotImplementedError

    def post_query_update(self, test_point: torch.Tensor, outcome, history) -> None:
        """
        Optional hook called after oracle query, before VI update.
        Override for strategies that maintain internal state (GP, VersionSpace).
        Default: no-op.
        """
        pass
