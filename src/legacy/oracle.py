"""
Oracle for querying ground truth feasibility.
"""

import torch
from typing import List

from active_learning.src.test_history import TestHistory
from active_learning.src.legacy.feasibility_checker import FeasibilityChecker


class Oracle:
    """Oracle that queries ground truth feasibility and records history."""

    def __init__(self, ground_truth: FeasibilityChecker, joint_names: List[str]):
        """
        Args:
            ground_truth: The true user's feasibility model
            joint_names: Ordered list of joint names
        """
        self.ground_truth = ground_truth
        self.joint_names = joint_names
        self.history = TestHistory(joint_names)

    def query(self, test_point: torch.Tensor) -> bool:
        """
        Query feasibility at test_point.

        Args:
            test_point: Shape (n_joints,) tensor

        Returns:
            True if feasible, False otherwise
        """
        h_value = self.ground_truth.h_value(test_point)
        outcome = bool(h_value >= 0)
        
        # Ensure h_value is a scalar float for history
        if hasattr(h_value, 'item'):
            h_float = float(h_value.item())
        else:
            h_float = float(h_value)
            
        self.history.add(test_point, outcome, h_float)
        return outcome

    def get_history(self) -> TestHistory:
        """Return the test history."""
        return self.history

    def reset(self):
        """Clear history."""
        self.history = TestHistory(self.joint_names)
