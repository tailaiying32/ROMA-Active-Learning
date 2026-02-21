
import os
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

from active_learning.src.config import DEVICE, load_config
from infer_params.training.model import LevelSetDecoder
from infer_params.training.level_set_torch import create_evaluation_grid

def load_diagnostic_model(
    model_path: Optional[str] = None, 
    device: str = DEVICE
) -> Tuple[nn.Module, Dict[str, Any], torch.Tensor]:
    """
    Load the decoder model, config, and embeddings for diagnostics.
    
    Args:
        model_path: Path to checkpoint. Defaults to 'models/best_model.pt'.
        device: Torch device string.
        
    Returns:
        decoder: Loaded LevelSetDecoder model.
        config: Configuration dictionary.
        embeddings: Training embeddings tensor.
    """
    if model_path is None:
        # Default to models/best_model.pt relative to project root
        # This assumes script is run from project root or relative paths work
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        model_path = os.path.join(root_dir, 'models/best_model.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading diagnostic model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    train_config = checkpoint['config']
    model_cfg = train_config['model']
    embeddings = checkpoint['embeddings']
    num_samples = embeddings.shape[0]

    decoder = LevelSetDecoder(
        num_samples=num_samples,
        latent_dim=model_cfg['latent_dim'],
        hidden_dim=model_cfg.get('hidden_dim', 256),
        num_blocks=model_cfg.get('num_blocks', 3),
        num_slots=model_cfg.get('num_slots', 18),
        params_per_slot=model_cfg.get('params_per_slot', 6),
    )
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder = decoder.to(device)
    decoder.eval()
    
    # Load default active learning config to combine with training details if needed
    # But usually train_config has model info, active_learning config has prior info
    # We will return the one from checkpoint for model internal consistency
    # But checking routines might need to load 'active_learning/configs/default.yaml' themselves for priors.
    
    return decoder, train_config, embeddings.to(device)

def get_diagnostic_grid(
    resolution: int = 10,
    bounds: Optional[Tuple[float, float]] = None,
    n_dims: int = 4,
    device: str = DEVICE
) -> torch.Tensor:
    """
    Create a standard evaluation grid for diagnostics.
    
    Args:
        resolution: Grid resolution per dimension.
        bounds: Tuple of (min, max) for all dimensions. Default (-pi, pi).
        n_dims: Number of joint dimensions.
        device: Torch device.
        
    Returns:
        grid: Tensor of shape (resolution**n_dims, n_dims)
    """
    if bounds is None:
        import numpy as np
        bounds = (-np.pi, np.pi)
        
    lower = torch.full((n_dims,), bounds[0], device=device)
    upper = torch.full((n_dims,), bounds[1], device=device)
    
    return create_evaluation_grid(lower, upper, resolution=resolution, device=device)
