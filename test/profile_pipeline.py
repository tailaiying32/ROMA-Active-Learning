"""
Profiling script for the Active Learning Pipeline.

Runs a few iterations with detailed timing for each phase to identify bottlenecks.
"""

import sys
import os
import time
import argparse
import torch
import numpy as np
from contextlib import contextmanager
from collections import defaultdict

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # Change to project root for relative paths

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.factory import build_learner
from active_learning.src.utils import load_decoder_model
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.metrics import compute_reachability_metrics
from infer_params.training.level_set_torch import create_evaluation_grid


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.current_context = []

    @contextmanager
    def __call__(self, name):
        self.current_context.append(name)
        full_name = "/".join(self.current_context)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            self.timings[full_name].append(elapsed)
            self.current_context.pop()

    def report(self):
        print("\n" + "="*80)
        print("TIMING REPORT")
        print("="*80)

        # Sort by total time
        sorted_items = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )

        for name, times in sorted_items:
            total = sum(times)
            mean = np.mean(times)
            std = np.std(times) if len(times) > 1 else 0
            count = len(times)

            indent = "  " * name.count("/")
            short_name = name.split("/")[-1]

            print(f"{indent}{short_name}:")
            print(f"{indent}  Total: {total:.3f}s | Mean: {mean*1000:.1f}ms | Std: {std*1000:.1f}ms | Count: {count}")

        print("="*80)


def profile_decoder(decoder, timer, n_samples=50, n_test_points=100):
    """Profile the decoder forward pass."""
    print("\n--- Profiling Decoder ---")

    latent_dim = 32
    n_joints = 4

    # Create dummy inputs
    z = torch.randn(n_samples, latent_dim, device=DEVICE)
    test_points = torch.randn(n_test_points, n_joints, device=DEVICE)

    # Profile decode_from_embedding
    with timer("decoder/decode_from_embedding"):
        for _ in range(10):
            decoded = decoder.decode_from_embedding(z)

    # Profile evaluate_from_decoded
    with timer("decoder/evaluate_from_decoded"):
        for _ in range(10):
            logits = LatentFeasibilityChecker.evaluate_from_decoded(test_points, decoded)

    # Profile batched_logit_values (combined)
    with timer("decoder/batched_logit_values"):
        for _ in range(10):
            logits = LatentFeasibilityChecker.batched_logit_values(decoder, z, test_points)


def profile_svgd_step(learner, timer):
    """Profile a single SVGD optimization step."""
    print("\n--- Profiling SVGD Step ---")

    if not hasattr(learner.posterior, 'get_particles'):
        print("Skipping - not a particle-based posterior")
        return

    vi = learner.vi
    particles = learner.posterior.get_particles().clone()
    K, D = particles.shape

    # Create dummy test history with a few points
    history = learner.oracle.get_history()

    # Profile log_likelihood
    with timer("svgd/log_likelihood"):
        for _ in range(10):
            ll = vi.log_likelihood(history, particles)

    # Profile log_prior
    with timer("svgd/log_prior"):
        for _ in range(10):
            lp = vi.log_prior(particles)

    # Profile gradient computation
    with timer("svgd/gradient_computation"):
        for _ in range(10):
            p_in = particles.detach().requires_grad_(True)
            ll = vi.log_likelihood(history, p_in)
            lp = vi.log_prior(p_in)
            ll_grad = torch.autograd.grad(ll.sum(), p_in, retain_graph=True)[0]
            lp_grad = torch.autograd.grad(lp.sum(), p_in)[0]
            log_prob_grad = ll_grad + lp_grad

    # Profile SVGD kernel step
    with timer("svgd/kernel_step"):
        for _ in range(10):
            phi = vi.optimizer.step(particles, log_prob_grad)


def profile_bald_acquisition(learner, timer):
    """Profile BALD acquisition function."""
    print("\n--- Profiling BALD Acquisition ---")

    bald = learner.bald_calculator
    bounds = learner.bounds
    n_joints = bounds.shape[0]

    # Profile sampling
    with timer("bald/sample_posterior"):
        for _ in range(10):
            if hasattr(learner.posterior, 'get_particles'):
                zs = learner.posterior.get_particles()
            else:
                zs = learner.posterior.sample(bald.n_samples)

    # Profile decoding
    with timer("bald/decode_samples"):
        for _ in range(10):
            decoded_params = LatentFeasibilityChecker.decode_latent_params(learner.decoder, zs)

    # Profile single score computation
    test_point = bounds.mean(dim=1)
    with timer("bald/compute_score_single"):
        for _ in range(10):
            score = bald.compute_score(test_point, zs=zs)

    # Profile batched score computation
    n_restarts = 5
    test_points = bounds[:, 0] + torch.rand(n_restarts, n_joints, device=DEVICE) * (bounds[:, 1] - bounds[:, 0])
    with timer("bald/compute_score_batched"):
        for _ in range(10):
            scores = bald.compute_score_batched(test_points, decoded_params=decoded_params)

    # Profile full select_test
    with timer("bald/select_test"):
        test, score, stats = bald.select_test(bounds, verbose=False, iteration=0)


def profile_metrics(decoder, gt_z, posterior, bounds, config, timer):
    """Profile metrics computation."""
    print("\n--- Profiling Metrics ---")

    # Create evaluation grid
    eval_res = config.get('metrics', {}).get('grid_resolution', 12)
    test_grid = create_evaluation_grid(bounds[:, 0], bounds[:, 1], eval_res, DEVICE)

    # Decode ground truth
    with torch.no_grad():
        gt_lower, gt_upper, gt_weights, gt_pres_logits, gt_blob_params = decoder.decode_from_embedding(gt_z.unsqueeze(0))
        ground_truth_params = {
            'box_lower': gt_lower.squeeze(0),
            'box_upper': gt_upper.squeeze(0),
            'box_weights': gt_weights.squeeze(0),
            'presence': torch.sigmoid(gt_pres_logits).squeeze(0),
            'blob_params': gt_blob_params.squeeze(0)
        }

    # Get posterior mean
    if hasattr(posterior, 'get_particles'):
        posterior_mean = posterior.mean.unsqueeze(0)
    else:
        posterior_mean = posterior.mean.unsqueeze(0)

    # Profile full metrics computation
    with timer("metrics/compute_reachability_metrics"):
        for _ in range(5):
            iou, acc, f1, boundary_acc = compute_reachability_metrics(
                decoder=decoder,
                ground_truth_params=ground_truth_params,
                posterior_mean=posterior_mean,
                test_grid=test_grid
            )


def profile_full_iteration(learner, timer):
    """Profile a complete active learning iteration."""
    print("\n--- Profiling Full Iteration ---")

    with timer("iteration/total"):
        with timer("iteration/step"):
            result = learner.step(verbose=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-iterations", type=int, default=5, help="Number of full iterations to profile")
    parser.add_argument("--posterior-method", type=str, default=None,
                        choices=['vi', 'svgd', 'sliced_svgd', 'projected_svgd'])
    parser.add_argument("--n-particles", type=int, default=50)
    args = parser.parse_args()

    timer = Timer()

    # Load config
    config = load_config(os.path.join(os.path.dirname(__file__), '../configs/latent.yaml'))
    config['stopping']['budget'] = 100

    # Override posterior method if specified
    if args.posterior_method:
        config.setdefault('posterior', {})['method'] = args.posterior_method
    if args.n_particles:
        config.setdefault('posterior', {})['n_particles'] = args.n_particles

    posterior_method = config.get('posterior', {}).get('method', 'vi')
    n_particles = config.get('posterior', {}).get('n_particles', 50)

    print(f"Profiling Active Learning Pipeline")
    print(f"  Posterior Method: {posterior_method}")
    print(f"  N Particles: {n_particles}")
    print(f"  Device: {DEVICE}")

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\nLoading model...")
    with timer("setup/load_model"):
        decoder, embeddings, _ = load_decoder_model("models/best_model.pt", DEVICE)

    # Select ground truth user
    prior_gen = LatentPriorGenerator(config, decoder)
    gt_idx = np.random.randint(0, len(embeddings))
    gt_z = embeddings[gt_idx].clone()

    # Setup prior/posterior
    with timer("setup/create_prior"):
        prior = prior_gen.get_prior(gt_z, embeddings=embeddings)

    with timer("setup/create_posterior"):
        posterior = LatentUserDistribution(
            latent_dim=prior.latent_dim,
            decoder=decoder,
            mean=prior.mean.clone(),
            log_std=prior.log_std.clone(),
            device=DEVICE
        )

    bounds = get_bounds_from_config(config, DEVICE)
    oracle = LatentOracle(decoder, gt_z, bounds.shape[0])

    # Build learner
    with timer("setup/build_learner"):
        learner = build_learner(
            decoder=decoder,
            prior=prior,
            posterior=posterior,
            oracle=oracle,
            bounds=bounds,
            config=config,
            embeddings=embeddings
        )

    # Profile individual components
    profile_decoder(decoder, timer)

    # Run a few warmup iterations
    print("\n--- Warmup Iterations ---")
    for i in range(3):
        learner.step(verbose=True)

    profile_bald_acquisition(learner, timer)
    profile_svgd_step(learner, timer)
    profile_metrics(decoder, gt_z, learner.get_posterior(), bounds, config, timer)

    # Profile full iterations
    print(f"\n--- Profiling {args.n_iterations} Full Iterations ---")
    for i in range(args.n_iterations):
        profile_full_iteration(learner, timer)
        print(f"  Iteration {i+1}/{args.n_iterations} complete")

    # Print report
    timer.report()

    # Summary of key bottlenecks
    print("\n" + "="*80)
    print("KEY BOTTLENECKS SUMMARY")
    print("="*80)

    key_timings = [
        ("BALD select_test", "bald/select_test"),
        ("SVGD log_likelihood", "svgd/log_likelihood"),
        ("SVGD kernel_step", "svgd/kernel_step"),
        ("Decoder batched_logit_values", "decoder/batched_logit_values"),
        ("Metrics computation", "metrics/compute_reachability_metrics"),
        ("Full iteration", "iteration/total"),
    ]

    for label, key in key_timings:
        if key in timer.timings:
            times = timer.timings[key]
            mean = np.mean(times) * 1000
            print(f"  {label}: {mean:.1f}ms avg")


if __name__ == "__main__":
    main()
