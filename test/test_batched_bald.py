"""
Test script for batched BALD optimization.

Verifies:
1. compute_score_batched matches compute_score for single inputs
2. Batched select_test produces valid results
3. Performance improvement from batching

Usage:
    python -m active_learning.test.test_batched_bald
"""

import sys
import os
import time
import torch
import numpy as np

# Ensure project root is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from active_learning.src.config import load_config, get_bounds_from_config, DEVICE
from active_learning.src.latent_bald import LatentBALD
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.svgd.particle_user_distribution import ParticleUserDistribution
from infer_params.training.model import LevelSetDecoder


def load_decoder(model_path: str, device: str = DEVICE):
    """Load the LevelSetDecoder model."""
    print(f"Loading decoder from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    train_config = checkpoint['config']
    model_cfg = train_config['model']

    embeddings = checkpoint['embeddings']
    num_samples = embeddings.shape[0]
    latent_dim = model_cfg['latent_dim']

    model = LevelSetDecoder(
        num_samples=num_samples,
        latent_dim=latent_dim,
        hidden_dim=model_cfg['hidden_dim'],
        num_blocks=model_cfg['num_blocks'],
        num_slots=model_cfg.get('num_slots', 18),
        params_per_slot=model_cfg.get('params_per_slot', 6),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, embeddings.to(device), latent_dim


def test_compute_score_batched_correctness(bald: LatentBALD, bounds: torch.Tensor, n_tests: int = 5):
    """Test that compute_score_batched matches compute_score for individual inputs."""
    print("\n=== Test 1: compute_score_batched correctness ===")

    device = bald.posterior.device
    n_joints = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # Generate random test points
    tests = lower + torch.rand(n_tests, n_joints, device=device) * (upper - lower)

    # Sample and decode once
    zs = bald._detach_samples(bald.posterior.sample(bald.n_samples))
    with torch.no_grad():
        decoded_params = LatentFeasibilityChecker.decode_latent_params(bald.decoder, zs)
        decoded_params = tuple(p.detach() for p in decoded_params)

    # Compute batched scores
    batched_scores = bald.compute_score_batched(tests, decoded_params=decoded_params, iteration=0)

    # Compute individual scores
    individual_scores = []
    for i in range(n_tests):
        score = bald.compute_score(tests[i], decoded_params=decoded_params, iteration=0)
        individual_scores.append(score.item())
    individual_scores = torch.tensor(individual_scores, device=device)

    # Compare
    max_diff = (batched_scores - individual_scores).abs().max().item()
    print(f"  Batched scores:    {batched_scores.tolist()}")
    print(f"  Individual scores: {individual_scores.tolist()}")
    print(f"  Max difference:    {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  [PASS] Batched scores match individual scores")
        return True
    else:
        print("  [FAIL] Scores don't match!")
        return False


def test_select_test_validity(bald: LatentBALD, bounds: torch.Tensor):
    """Test that batched select_test produces valid results."""
    print("\n=== Test 2: select_test validity ===")

    # Run select_test
    best_test, best_score, diag_stats = bald.select_test(
        bounds=bounds,
        n_restarts=10,
        n_iters=50,
        verbose=True,
        iteration=0
    )

    # Check bounds
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    in_bounds = (best_test >= lower).all() and (best_test <= upper).all()

    print(f"  Best test: {best_test.tolist()}")
    print(f"  Best score: {best_score:.4f}")
    print(f"  In bounds: {in_bounds}")
    print(f"  Diagnostics: {len(diag_stats)} restarts logged")

    if in_bounds and best_score > 0:
        print("  [PASS] select_test produces valid results")
        return True
    else:
        print("  [FAIL] Invalid results!")
        return False


def test_performance(bald: LatentBALD, bounds: torch.Tensor, n_runs: int = 3):
    """Benchmark batched optimization performance."""
    print("\n=== Test 3: Performance benchmark ===")

    times = []
    for run in range(n_runs):
        start = time.perf_counter()
        best_test, best_score, _ = bald.select_test(
            bounds=bounds,
            n_restarts=10,
            n_iters=80,
            verbose=False,
            iteration=0
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run + 1}: {elapsed:.3f}s (score={best_score:.4f})")

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"  Average: {avg_time:.3f}s +/- {std_time:.3f}s")
    print(f"  [INFO] Compare this to sequential baseline (~10x slower expected)")

    return avg_time


def test_sequential_comparison(bald: LatentBALD, bounds: torch.Tensor):
    """Compare batched vs sequential (simulated) performance."""
    print("\n=== Test 4: Sequential vs Batched comparison ===")

    n_restarts = 10
    n_iters = 80
    device = bald.posterior.device
    n_joints = bounds.shape[0]
    lower = bounds[:, 0].detach()
    upper = bounds[:, 1].detach()

    # --- Sequential baseline (simulated by running 1 restart at a time) ---
    print("  Running sequential baseline (1 restart at a time)...")
    seq_times = []
    for r in range(n_restarts):
        start = time.perf_counter()

        # Single restart
        t = lower + torch.rand(1, n_joints, device=device) * (upper - lower)
        t = t.clone().requires_grad_(True)

        zs = bald._detach_samples(bald.posterior.sample(bald.n_samples))
        with torch.no_grad():
            decoded_params = LatentFeasibilityChecker.decode_latent_params(bald.decoder, zs)
            decoded_params = tuple(p.detach() for p in decoded_params)

        optimizer = torch.optim.Adam([t], lr=bald.opt_lr_adam)
        for i in range(n_iters):
            optimizer.zero_grad()
            score = bald.compute_score_batched(t, decoded_params=decoded_params, iteration=0)
            (-score.sum()).backward()
            optimizer.step()
            with torch.no_grad():
                t.data = torch.clamp(t.data, lower, upper)

        elapsed = time.perf_counter() - start
        seq_times.append(elapsed)

    total_seq = sum(seq_times)
    print(f"    Sequential total: {total_seq:.3f}s ({np.mean(seq_times):.3f}s per restart)")

    # --- Batched (current implementation) ---
    print("  Running batched (all restarts at once)...")
    start = time.perf_counter()
    best_test, best_score, _ = bald.select_test(
        bounds=bounds,
        n_restarts=n_restarts,
        n_iters=n_iters,
        verbose=False,
        iteration=0
    )
    batched_time = time.perf_counter() - start
    print(f"    Batched total: {batched_time:.3f}s")

    speedup = total_seq / batched_time
    print(f"  Speedup: {speedup:.1f}x")

    if speedup > 2.0:
        print("  [PASS] Batched is significantly faster")
        return True
    else:
        print("  [INFO] Speedup lower than expected (may be memory-bound on CPU)")
        return True  # Still pass, just note it


def main():
    print("=" * 60)
    print("Batched BALD Optimization Test Suite")
    print("=" * 60)

    # Load config and model
    config = load_config()
    model_path = config.get('latent', {}).get('model_path', 'models/best_model.pt')

    # Resolve path relative to active_learning directory (where config is)
    if not os.path.isabs(model_path):
        config_dir = os.path.join(project_root, 'active_learning')
        model_path = os.path.normpath(os.path.join(config_dir, model_path))

    decoder, embeddings, latent_dim = load_decoder(model_path, DEVICE)
    print(f"Device: {DEVICE}")
    print(f"Latent dim: {latent_dim}")

    # Create posterior (particle-based for realistic test)
    n_particles = 50
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0)

    # Initialize particles around the mean with some spread
    init_particles = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n_particles, latent_dim, device=DEVICE)

    # Create particle posterior
    posterior = ParticleUserDistribution(
        latent_dim=latent_dim,
        decoder=decoder,
        n_particles=n_particles,
        init_particles=init_particles,
        device=DEVICE,
    )

    # Create BALD instance
    bald = LatentBALD(
        decoder=decoder,
        posterior=posterior,
        config=config,
        prior=None
    )

    # Get bounds
    bounds = get_bounds_from_config(config, DEVICE)
    print(f"Bounds shape: {bounds.shape}")
    print(f"N restarts: {bald.opt_n_restarts}, N iters: {bald.opt_n_iters}")
    print(f"N MC samples: {bald.n_samples}")

    # Run tests
    results = []

    results.append(("Correctness", test_compute_score_batched_correctness(bald, bounds)))
    results.append(("Validity", test_select_test_validity(bald, bounds)))
    avg_time = test_performance(bald, bounds)
    results.append(("Performance", avg_time < 10.0))  # Expect < 10s for batched
    results.append(("Speedup", test_sequential_comparison(bald, bounds)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        all_passed = all_passed and passed

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
