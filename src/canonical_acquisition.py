"""
Canonical acquisition strategy combining canonical exploration with BALD.

Uses pre-computed canonical exploration points for initial queries,
then switches to BALD for posterior-guided refinement.
"""

import os
import torch
import numpy as np
from typing import Tuple, Optional

from active_learning.src.config import DEVICE


class CanonicalAcquisition:
    """
    Canonical acquisition strategy: canonical exploration → BALD.
    
    For the first n_canonical queries, returns pre-computed canonical
    exploration points. After that, delegates to the BALD strategy.
    
    This provides a strong foundation of globally informative observations
    before transitioning to posterior-guided selection.
    """
    
    def __init__(
        self,
        bald_strategy,
        canonical_path: str = "models/canonical_queries.npz",
        n_canonical: int = 5,
        device: str = DEVICE
    ):
        """
        Initialize canonical acquisition.
        
        Args:
            bald_strategy: Instance of LatentBALD for posterior-guided selection
            canonical_path: Path to pre-computed canonical queries (.npz file)
            n_canonical: Number of canonical queries to use
            device: Torch device
        """
        self.bald = bald_strategy
        self.n_canonical = n_canonical
        self.device = device
        self.query_count = 0
        
        # Load pre-computed canonical points
        if os.path.exists(canonical_path):
            data = np.load(canonical_path)
            self.canonical_points = torch.tensor(data['points'], device=device, dtype=torch.float32)
            self.canonical_scores = data.get('scores', np.zeros(len(data['points'])))
            print(f"Loaded {len(self.canonical_points)} canonical exploration points from {canonical_path}")
        else:
            print(f"Warning: Canonical points file not found at {canonical_path}")
            print("Falling back to pure BALD acquisition")
            self.canonical_points = None
            self.n_canonical = 0
    
    def select_test(
        self,
        bounds: torch.Tensor,
        test_history=None,
        verbose: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, float]:
        """
        Select next test point.
        
        Args:
            bounds: (n_joints, 2) tensor of [min, max] bounds
            test_history: Optional history of test results (passed to BALD)
            verbose: Print progress (passed to BALD)
            **kwargs: Additional arguments (ignored for canonical, passed to BALD)
            
        Returns:
            test_point: (n_joints,) coordinate tensor
            score: Acquisition score (0 for canonical, BALD score otherwise)
        """
        # Phase 1: Canonical exploration
        if self.query_count < self.n_canonical and self.canonical_points is not None:
            point = self.canonical_points[self.query_count].to(self.device)
            score = float(self.canonical_scores[self.query_count]) if self.canonical_scores is not None else 0.0
            self.query_count += 1
            return point, score
        
        # Phase 2: BALD (only pass test_history, not posterior)
        self.query_count += 1
        return self.bald.select_test(bounds, test_history=test_history, verbose=verbose)
    
    def reset(self):
        """Reset query counter for a new trial."""
        self.query_count = 0
    
    @property
    def in_exploration_phase(self) -> bool:
        """True if still in canonical exploration phase."""
        return self.query_count < self.n_canonical and self.canonical_points is not None
    
    @property
    def phase_name(self) -> str:
        """Current acquisition phase name."""
        return "Canonical" if self.in_exploration_phase else "BALD"
