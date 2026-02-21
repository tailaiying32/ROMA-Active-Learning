import torch
import numpy as np
import itertools
from typing import Dict, Tuple, Optional

class GridStrategy:
    """
    Fixed Grid Sampling Strategy.
    
    Generates a deterministic grid over the joint space and selects parameters
    that maximize the acquisition function (BALD or Entropy) on this grid.
    """
    
    def __init__(self, joint_limits: Dict[str, Tuple[float, float]], resolution: int = 5):
        """
        Args:
            joint_limits: Dictionary of {joint_name: (min, max)}
            resolution: Number of points per dimension. 
                        WARNING: Total points = resolution ^ n_joints.
                        Keep small (e.g. 3-6) for high dim.
        """
        self.joint_limits = joint_limits
        self.joint_names = list(joint_limits.keys())
        self.resolution = resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Pre-compute grid
        self.grid = self._create_grid()
        self.scoring_fn = None  # Set by factory to provide BALD scoring
        print(f"[GridStrategy] Initialized with {len(self.grid)} points (resolution={resolution})")

    def _create_grid(self) -> torch.Tensor:
        """Creates a meshgrid over the joint space."""
        ranges = []
        for name in self.joint_names:
            low, high = self.joint_limits[name]
            # Handle if limits are tensors
            if isinstance(low, torch.Tensor): low = low.item()
            if isinstance(high, torch.Tensor): high = high.item()
            
            # Linspace for this dimension
            ranges.append(torch.linspace(low, high, self.resolution))
            
        # Create meshgrid
        # indexing='ij' ensures first dim corresponds to first range, etc.
        grids = torch.meshgrid(*ranges, indexing='ij')
        
        # Stack and reshape to (N, n_joints)
        grid_points = torch.stack(grids, dim=-1).reshape(-1, len(self.joint_names))
        
        return grid_points.to(self.device)

    def select_test_point(self, acquisition_fn) -> torch.Tensor:
        """
        Selects the best point from the grid using the provided acquisition function.
        
        Args:
            acquisition_fn: Callable that takes (test_points) and returns scores (N,)
                            Example: bald.compute_score_batched(test_points)
                            
        Returns:
            Selected test point (n_joints,)
        """
        with torch.no_grad():
            # Evaluate acquisition function on the entire grid
            # If grid is huge, might need to batch this, but for baselines usually <100k
            scores = acquisition_fn(self.grid)
            
            # Find max
            max_idx = torch.argmax(scores)
            best_point = self.grid[max_idx]
            
            return best_point.clone()

    def select_test(self, bounds: torch.Tensor, **kwargs) -> tuple:
        """Adapter to conform to AcquisitionStrategy interface."""
        if self.scoring_fn is not None:
            test_point = self.select_test_point(self.scoring_fn)
        else:
            # Without scoring function, pick random grid point
            idx = torch.randint(0, len(self.grid), (1,)).item()
            test_point = self.grid[idx]
        return test_point, 0.0
