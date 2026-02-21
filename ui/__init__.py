"""
Interactive UI for ROMA Active Learning Pipeline.

This module provides a Streamlit-based UI for human-in-the-loop active learning,
allowing users to provide real-time feasibility assessments for arm poses.
"""

# Lazy imports to avoid circular dependencies
__all__ = ['InteractiveOracle', 'ArmVisualizer']

def __getattr__(name):
    if name == 'InteractiveOracle':
        from .interactive_oracle import InteractiveOracle
        return InteractiveOracle
    elif name == 'ArmVisualizer':
        from .arm_visualizer import ArmVisualizer
        return ArmVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
