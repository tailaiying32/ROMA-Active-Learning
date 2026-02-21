

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TestResult:
    """Data structure for a single test result."""
    test_point: torch.Tensor  # The test configuration t ∈ ℝ^n_joints
    outcome: float            # Signed distance (positive = feasible, negative = infeasible)
    timestamp: int            # Query index (0, 1, 2, ...)
    h_value: Optional[float]  # Optional: actual h-value from oracle
    metadata: Optional[dict]  # Optional: any extra info


class TestHistory:
    '''Stores and manages test history.'''

    def __init__(self, joint_names: list[str]):
        '''
        Initialize empty history with joint names.

        Args:
            joint_names: Ordered list of joint names
        '''
        self.joint_names = joint_names
        self._results: list[TestResult] = []

    def add(
        self,
        test_point: torch.Tensor,
        outcome: float,
        h_value: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> TestResult:
        '''
        Record a new test result.

        Args:
            test_point: The test configuration tensor, shape (n_joints,)
            outcome: Signed distance outcome
            h_value: Optional actual h-value from oracle
            metadata: Optional extra information

        Returns:
            The created TestResult
        '''
        result = TestResult(
            test_point=test_point.clone().detach(),
            outcome=outcome,
            timestamp=len(self._results),
            h_value=h_value,
            metadata=metadata
        )
        self._results.append(result)
        return result

    def get_all(self) -> list[TestResult]:
        '''
        Retrieve all recorded test results.

        Returns:
            List of TestResult objects in order of addition.
        '''
        return list(self._results)