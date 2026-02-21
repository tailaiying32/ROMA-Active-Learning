"""
Quasi-Random acquisition strategy using Sobol sequences.

Uses a low-discrepancy Sobol sequence to select the first N queries
for better space coverage than random sampling, then switches to BALD.
"""

import torch
import numpy as np
import scipy.stats.qmc as qmc
from typing import Tuple, Optional

from active_learning.src.config import DEVICE


class QuasiRandomStrategy:
    """
    Quasi-Random acquisition strategy: Sobol Sequence → BALD.

    For the first n_quasi_random queries, returns points from a Sobol sequence
    scaled to the search bounds. This ensures better initial coverage of the
    joint space than uniform random sampling.

    After the initial phase, delegates to the BALD strategy.

    Requires scipy >= 1.7.0 for qmc.Sobol
    """

    def __init__(
        self,
        bald_strategy,
        n_quasi_random: int = 10,
        device: str = DEVICE,
        seed: int = 42
    ):
        """
        Initialize quasi-random acquisition.

        Args:
            bald_strategy: Instance of LatentBALD for posterior-guided selection
            n_quasi_random: Number of quasi-random queries to use
            device: Torch device
            seed: Random seed for scrambing (if supported)
        """
        self.bald = bald_strategy
        self.n_quasi_random = n_quasi_random
        self.device = device
        self.query_count = 0

        # We delay initialization of the sampler until select_test
        # because we need the dimensions from the bounds
        self.sampler = None
        self.sobol_points = None
        self.seed = seed

    def _initialize_sampler(self, bounds: torch.Tensor):
        """
        Initialize the Sobol sampler and generate points.

        Args:
            bounds: (n_joints, 2) tensor of [min, max]
        """
        n_dims = bounds.shape[0]

        # Initialize Sobol sampler
        # Scramble=True is generally recommended for better properties
        try:
            self.sampler = qmc.Sobol(d=n_dims, scramble=True, seed=self.seed)
        except TypeError:
            # Fallback for older scipy versions if scramble/seed not supported in init
            self.sampler = qmc.Sobol(d=n_dims)

        # Generate 2^m points where 2^m >= n_quasi_random
        # Sobol sequences are best when using powers of 2
        m = int(np.ceil(np.log2(self.n_quasi_random)))
        n_points = 2**m

        # Generate raw points in [0, 1)
        raw_points = self.sampler.random(n=n_points)

        # Scale to bounds
        lower = bounds[:, 0].cpu().numpy()
        upper = bounds[:, 1].cpu().numpy()
        scaled_points = qmc.scale(raw_points, lower, upper)

        # Convert to tensor
        self.sobol_points = torch.tensor(scaled_points, device=self.device, dtype=torch.float32)

        # If we generated more than needed, we'll just index what we need
        print(f"Initialized Sobol sampler (d={n_dims}) with {self.n_quasi_random} points")

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
            **kwargs: Additional arguments (ignored for quasi-random, passed to BALD)

        Returns:
            test_point: (n_joints,) coordinate tensor
            score: Acquisition score (0 for quasi-random, BALD score otherwise)
        """
        # Phase 1: Quasi-random exploration
        # If bald_strategy is None (standalone baseline mode), we stay in this phase indefinitely
        if self.bald is None or self.query_count < self.n_quasi_random:
            # Initialize on first call
            # For standalone mode, if we run out of generated points, we should ideally generate more.
            # However, usually n_quasi_random is set to budget.
            # If we need more points than initially generated (because we are in standalone mode
            # and n_quasi_random was maybe small?), we should expand.
            # But the simplest contract is: initialize with n_quasi_random = budget for baseline.
            
            if self.sobol_points is None:
                self._initialize_sampler(bounds)

            # If we run out of points in standalone mode, generate more
            if self.query_count >= len(self.sobol_points):
                 if self.bald is None:
                    # Generate next batch (next power of 2 automatically by Sobol engine usually)
                    # But qmc.Sobol.random() continues the sequence.
                    # We just need to scale them.
                    n_new = len(self.sobol_points) # double the size
                    raw_points = self.sampler.random(n=n_new)
                    lower = bounds[:, 0].cpu().numpy()
                    upper = bounds[:, 1].cpu().numpy()
                    scaled_points = qmc.scale(raw_points, lower, upper)
                    new_tensor = torch.tensor(scaled_points, device=self.device, dtype=torch.float32)
                    self.sobol_points = torch.cat([self.sobol_points, new_tensor], dim=0)
                    if verbose:
                        print(f"QuasiRandomStrategy: Extended sequence to {len(self.sobol_points)} points")
                 else:
                     # This shouldn't happen if logic is correct (Phase 2 catches this)
                     pass

            point = self.sobol_points[self.query_count]
            self.query_count += 1
            return point, 0.0

        # Phase 2: BALD
        self.query_count += 1
        return self.bald.select_test(bounds, test_history=test_history, verbose=verbose)

    def reset(self):
        """Reset query counter for a new trial."""
        self.query_count = 0
        # Check if we should re-scramble/re-sample on reset?
        # Typically we want the same sequence for reproducibility,
        # or we might want different sequences.
        # For now, keep the same sequence.

    @property
    def in_exploration_phase(self) -> bool:
        """True if still in quasi-random exploration phase."""
        if self.bald is None:
            return True
        return self.query_count < self.n_quasi_random

    @property
    def phase_name(self) -> str:
        """Current acquisition phase name."""
        return "Quasi-Random" if self.in_exploration_phase else "BALD"
