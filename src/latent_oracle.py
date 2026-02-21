"""
Latent Oracle for querying ground truth feasibility in latent space.
"""

import torch
from typing import List

from active_learning.src.test_history import TestHistory
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker


class LatentOracle:
    """Oracle that queries ground truth feasibility using a latent code and decoder."""

    def __init__(self, decoder, ground_truth_z: torch.Tensor, n_joints: int):
        """
        Args:
            decoder: Decoder model to map latent codes to outputs
            ground_truth_z: The true user's latent code, shape (latent_dim,)
            n_joints: Number of joints (for TestHistory)
        """
        self.decoder = decoder
        self.ground_truth_z = ground_truth_z
        self.n_joints = n_joints
        self.ground_truth_checker = LatentFeasibilityChecker(
            decoder=decoder,
            z=ground_truth_z
        )
        self.history = TestHistory(joint_names=[f"joint_{i}" for i in range(n_joints)])

    def query(self, test_point: torch.Tensor) -> float:
        """
        Query feasibility at test_point.

        Args:
            test_point: Shape (n_joints,) tensor

        Returns:
            Binary feasibility label (1.0 if feasible, 0.0 if infeasible)
        """
        logit_value = self.ground_truth_checker.logit_value(test_point).squeeze()
        h_val = float(logit_value.item())
        outcome = 1.0 if h_val >= 0 else 0.0
        self.history.add(test_point, outcome, h_value=h_val)
        return outcome

    def get_history(self) -> TestHistory:
        """Return the test history."""
        return self.history

    def reset(self):
        """Clear history."""
        self.history = TestHistory(joint_names=[f"joint_{i}" for i in range(self.n_joints)])
