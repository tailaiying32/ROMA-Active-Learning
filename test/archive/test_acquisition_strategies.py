"""
Test script to verify that different acquisition strategies work correctly.
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE
from active_learning.src.legacy.prior_generation import PriorGenerator
from active_learning.src.legacy.sample_user import UserGenerator
from active_learning.src.legacy.oracle import Oracle
from active_learning.src.legacy.active_learning_pipeline import ActiveLearner


def test_acquisition_strategy(strategy: str, n_iterations: int = 3):
    """Test a single acquisition strategy."""
    print(f"\n{'='*60}")
    print(f"Testing acquisition strategy: {strategy.upper()}")
    print(f"{'='*60}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '../configs/legacy.yaml')
    config = load_config(config_path)

    # Override acquisition strategy
    if 'acquisition' not in config:
        config['acquisition'] = {}
    config['acquisition']['strategy'] = strategy

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup prior
    prior_gen = PriorGenerator(config)
    prior = prior_gen.get_prior()

    # Generate ground truth user
    user_gen = UserGenerator(
        config,
        prior_gen.joint_names,
        prior_gen.anatomical_limits,
        prior_gen.pairs
    )
    true_limits, true_bumps, true_checker = user_gen.generate_user()

    # Initialize posterior
    posterior = prior_gen.get_prior()

    # Initialize oracle
    oracle = Oracle(true_checker, prior_gen.joint_names)

    # Initialize learner
    learner = ActiveLearner(
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        config=config
    )

    # Verify strategy is set correctly
    assert learner.acquisition_strategy == strategy, \
        f"Expected strategy {strategy}, got {learner.acquisition_strategy}"

    print(f"✓ ActiveLearner initialized with strategy: {learner.acquisition_strategy}")

    # Run a few iterations
    for i in range(n_iterations):
        result = learner.step(verbose=True)
        print(f"  Iteration {i+1}: test_point shape={result.test_point.shape}, outcome={result.outcome}")

    print(f"✓ Strategy '{strategy}' completed {n_iterations} iterations successfully")
    return True


def main():
    print("\n" + "="*60)
    print("ACQUISITION STRATEGY TEST SUITE")
    print("="*60)

    strategies = ['bald', 'random', 'quasi-random']
    results = {}

    for strategy in strategies:
        try:
            success = test_acquisition_strategy(strategy, n_iterations=3)
            results[strategy] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"✗ Strategy '{strategy}' FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results[strategy] = "FAIL"

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for strategy, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {strategy:15s} : {result}")

    all_passed = all(r == "PASS" for r in results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
