"""
Baseline acquisition strategies for active learning comparison.

Available strategies:
- RandomStrategy: Uniform random sampling in joint space
- QuasiRandomStrategy: Sobol sequence → BALD hybrid
- GridStrategy: Fixed grid sampling with acquisition function
- GPStrategy: Gaussian Process with Straddle heuristic
- MultiStageWarmupStrategy: Iterative posterior boundary targeting → BALD
"""

from active_learning.src.baselines.random_strategy import RandomStrategy
from active_learning.src.baselines.quasi_random_strategy import QuasiRandomStrategy
from active_learning.src.baselines.grid_strategy import GridStrategy
from active_learning.src.baselines.gp_strategy import GPStrategy
from active_learning.src.baselines.multi_stage_warmup_strategy import MultiStageWarmupStrategy

__all__ = [
    'RandomStrategy',
    'QuasiRandomStrategy',
    'GridStrategy',
    'GPStrategy',
    'MultiStageWarmupStrategy',
]
