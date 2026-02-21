"""
Unit test for QuasiRandomStrategy.
"""

import sys
import os
import torch
import numpy as np
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from active_learning.src.baselines.quasi_random_strategy import QuasiRandomStrategy

class FakeBALD:
    def select_test(self, bounds, test_history=None, verbose=False):
        # Return random point for testing
        return torch.zeros(bounds.shape[0]), 0.123

def test_quasi_random_sequence():
    """Test that it generates points within bounds."""
    bounds = torch.tensor([[-1.0, 1.0], [0.0, 5.0]])
    n_points = 10
    
    strategy = QuasiRandomStrategy(
        bald_strategy=FakeBALD(),
        n_quasi_random=n_points,
        device='cpu'
    )
    
    points = []
    for _ in range(n_points):
        pt, score = strategy.select_test(bounds)
        assert score == 0.0
        points.append(pt)
        
    points = torch.stack(points)
    
    # Check bounds
    assert torch.all(points[:, 0] >= -1.0)
    assert torch.all(points[:, 0] <= 1.0)
    assert torch.all(points[:, 1] >= 0.0)
    assert torch.all(points[:, 1] <= 5.0)
    
    # Check uniqueness (Sobol shouldn't repeat quickly)
    assert len(torch.unique(points, dim=0)) == n_points

def test_sobol_determinism():
    """Test that the sequence is deterministic with same seed."""
    bounds = torch.tensor([[0.0, 1.0]])
    n_points = 5
    
    s1 = QuasiRandomStrategy(FakeBALD(), n_quasi_random=n_points, device='cpu', seed=42)
    pts1 = [s1.select_test(bounds)[0] for _ in range(n_points)]
    
    s2 = QuasiRandomStrategy(FakeBALD(), n_quasi_random=n_points, device='cpu', seed=42)
    pts2 = [s2.select_test(bounds)[0] for _ in range(n_points)]
    
    for p1, p2 in zip(pts1, pts2):
        assert torch.allclose(p1, p2)

def test_switch_to_bald():
    """Test switching to BALD after n_quasi_random."""
    bounds = torch.tensor([[0.0, 1.0]])
    n_points = 2
    
    strategy = QuasiRandomStrategy(FakeBALD(), n_quasi_random=n_points, device='cpu')
    
    # First 2 are limits
    _, score1 = strategy.select_test(bounds)
    assert score1 == 0.0
    _, score2 = strategy.select_test(bounds)
    assert score2 == 0.0
    
    # 3rd should be BALD
    _, score3 = strategy.select_test(bounds)
    assert score3 == 0.123

if __name__ == "__main__":
    # Manual run if pytest not available
    test_quasi_random_sequence()
    test_sobol_determinism()
    test_switch_to_bald()
    print("All tests passed!")
