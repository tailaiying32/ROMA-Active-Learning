"""
Interactive Oracle for human-in-the-loop active learning.

This oracle replaces LatentOracle for interactive sessions where a human user
provides feasibility assessments instead of a simulated ground truth.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
_this_file = Path(__file__).resolve()
_active_learning_dir = _this_file.parent.parent  # active_learning/

if str(_active_learning_dir) not in sys.path:
    sys.path.insert(0, str(_active_learning_dir))

import torch
from typing import Optional, List

from src.test_history import TestHistory


class InteractiveOracle:
    """
    Oracle for human-in-the-loop active learning.

    Unlike LatentOracle, this oracle:
    - Does NOT automatically answer queries
    - Stores a pending query for the UI to display
    - Provides methods for the UI to submit user's answer
    - Maintains query history compatible with the pipeline
    """

    def __init__(self, n_joints: int, joint_names: List[str] = None):
        """
        Initialize the interactive oracle.

        Args:
            n_joints: Number of joints (typically 4 for the arm)
            joint_names: Optional list of joint names for display
        """
        self.n_joints = n_joints
        self.joint_names = joint_names or [f"joint_{i}" for i in range(n_joints)]
        self.history = TestHistory(joint_names=self.joint_names)

        # Pending query state
        self._pending_query: Optional[torch.Tensor] = None
        self._query_count: int = 0

    def set_pending_query(self, test_point: torch.Tensor) -> int:
        """
        Set a test point as pending, waiting for user response.

        Args:
            test_point: Joint configuration tensor, shape (n_joints,), in radians

        Returns:
            query_id: Incremented query counter
        """
        self._pending_query = test_point.clone().detach()
        self._query_count += 1
        return self._query_count

    def get_pending_query(self) -> Optional[torch.Tensor]:
        """
        Return the current pending query.

        Returns:
            The pending test point tensor, or None if no query pending
        """
        return self._pending_query

    def has_pending_query(self) -> bool:
        """Check if there's a query awaiting user response."""
        return self._pending_query is not None

    def submit_response(self, outcome: float, h_value: Optional[float] = None) -> None:
        """
        Submit user's response for the pending query.

        Args:
            outcome: 1.0 for feasible/yes, 0.0 for infeasible/no
            h_value: Optional confidence/distance value (not used for human input)

        Raises:
            ValueError: If no pending query exists
        """
        if self._pending_query is None:
            raise ValueError("No pending query to respond to")

        self.history.add(
            test_point=self._pending_query,
            outcome=outcome,
            h_value=h_value
        )
        self._pending_query = None

    def get_history(self) -> TestHistory:
        """
        Return the test history.

        Compatible with the pipeline's expected interface.

        Returns:
            TestHistory object containing all past queries and outcomes
        """
        return self.history

    def reset(self) -> None:
        """Clear history and pending state for a new session."""
        self.history = TestHistory(joint_names=self.joint_names)
        self._pending_query = None
        self._query_count = 0

    def get_query_count(self) -> int:
        """Return the total number of queries made (including pending)."""
        return self._query_count

    def get_completed_count(self) -> int:
        """Return the number of completed queries (with responses)."""
        return len(self.history.get_all())

    def get_feasible_count(self) -> int:
        """Return the number of feasible responses."""
        return sum(1 for r in self.history.get_all() if r.outcome > 0.5)

    def get_infeasible_count(self) -> int:
        """Return the number of infeasible responses."""
        return sum(1 for r in self.history.get_all() if r.outcome <= 0.5)
