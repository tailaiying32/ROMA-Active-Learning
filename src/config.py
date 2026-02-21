import yaml
import torch
from pathlib import Path
import numpy as np
from typing import Dict, Any

try:
    from infer_params.config import load_default_config
except ImportError:
    # Allow import without infer_params if strictly not needed, but it checks in get_bounds
    load_default_config = None


# Global device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

def create_generator(config: dict, device: str = DEVICE) -> torch.Generator:
    """
    Create a torch.Generator seeded from config.
    
    Args:
        config: Configuration dictionary with optional 'seed' key
        device: Device for the generator
        
    Returns:
        Seeded Generator if seed is specified, None otherwise (truly random)
    """
    seed = config.get('seed', None)
    if seed is not None:
        return torch.Generator(device=device).manual_seed(seed)
    return None

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.
                     If None, defaults to 'configs/default.yaml' relative to the project root.

    Returns:
        Dictionary containing the configuration.
    """
    if config_path is None:
        # Assume this file is in src/, so project root is parent of parent
        project_root = Path(__file__).parent.parent
        config_path = project_root / "configs" / "latent.yaml"

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_bounds_from_config(config: dict, device: str = DEVICE) -> torch.Tensor:
    """
    Extract test point bounds from config (anatomical limits).

    Returns:
        bounds: Tensor of shape (n_joints, 2) with [lower, upper]
    """
    # Try active learning config first
    prior_config = config.get('prior', {})
    anatomical_limits = prior_config.get('anatomical_limits', None)

    # Check if we should enforce anatomical bounds for test selection
    bald_config = config.get('bald', {})
    use_anatomical_bounds = bald_config.get('use_anatomical_bounds', True)
    
    if not use_anatomical_bounds and anatomical_limits is not None:
        # Use wide bounds [-pi, pi] for all joints
        # This assumes joints are in radians/angular.
        # If we have N joints, we return N pairs of [-pi, pi]
        joint_names = prior_config.get('joint_names', list(anatomical_limits.keys()))
        bounds_list = [[-np.pi, np.pi] for _ in joint_names]
        return torch.tensor(bounds_list, dtype=torch.float32, device=device)

    if anatomical_limits is not None:
        # Config has anatomical limits in degrees
        units = prior_config.get('units', 'degrees')
        joint_names = prior_config.get('joint_names', list(anatomical_limits.keys()))
        # Tasks for Weighted BALD investigation:
        # - [x] Investigate `use_anatomical_limit_prior` implementation
        #     - [x] Locate parameter in `legacy.yaml`
        #     - [x] Search for parameter usage in `src/legacy/`
        #     - [x] Determine implementation status
        #     - [x] Report findings to user
        # - [/] Investigate `weighted-bald` functionality
        #     - [x] Check legacy `weighted-bald` implementation
        #     - [/] Check latent `weighted-bald` implementation
        #     - [ ] Compare behaviors and identify potential bugs
        #     - [ ] Report findings to user
        
        bounds_list = []
        for joint in joint_names:
            limits = anatomical_limits[joint]
            if units == 'degrees':
                limits = [np.deg2rad(limits[0]), np.deg2rad(limits[1])]
            bounds_list.append(limits)

        return torch.tensor(bounds_list, dtype=torch.float32, device=device)

    # Fallback to infer_params default config
    if load_default_config is None:
         # Need infer_params in path
         try:
            from infer_params.config import load_default_config as ldc
            infer_config = ldc()
         except ImportError:
            raise ImportError("Could not import load_default_config from infer_params.config. Ensure project root is in python path.")
    else:
        infer_config = load_default_config()

    base_lower = infer_config['joints']['base_lower']
    base_upper = infer_config['joints']['base_upper']

    bounds = torch.tensor(
        [[l, u] for l, u in zip(base_lower, base_upper)],
        dtype=torch.float32,
        device=device
    )

    return bounds

