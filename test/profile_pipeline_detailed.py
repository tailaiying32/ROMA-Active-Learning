"""
Detailed profiling for BALD select_test and SVGD update_posterior.
"""

import sys
import os
import time
import torch
import numpy as np
from collections import defaultdict
from contextlib import contextmanager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.factory import build_learner
from active_learning.src.utils import load_decoder_model
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker


def main():
    # Load config
    config = load_config(os.path.join(PROJECT_ROOT, "active_learning/configs/latent.yaml"))
    config['stopping']['budget'] = 100
    config.setdefault('posterior', {})['method'] = 'sliced_svgd'
    config.setdefault('posterior', {})['n_particles'] = 50

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Device: {DEVICE}")
    print(f"Posterior: sliced_svgd with 50 particles")

    # Load model
    decoder, embeddings, _ = load_decoder_model(os.path.join(PROJECT_ROOT, "models/best_model.pt"), DEVICE)

    # Setup
    prior_gen = LatentPriorGenerator(config, decoder)
    gt_z = embeddings[0].clone()
    prior = prior_gen.get_prior(gt_z, embeddings=embeddings)
    posterior = LatentUserDistribution(
        latent_dim=prior.latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )
    bounds = get_bounds_from_config(config, DEVICE)
    oracle = LatentOracle(decoder, gt_z, bounds.shape[0])

    learner = build_learner(
        decoder=decoder, prior=prior, posterior=posterior,
        oracle=oracle, bounds=bounds, config=config, embeddings=embeddings
    )

    # Warmup
    for _ in range(3):
        learner.step(verbose=False)

    print("\n" + "="*80)
    print("DETAILED BALD select_test PROFILING")
    print("="*80)

    bald = learner.bald_calculator
    n_restarts = bald.opt_n_restarts
    n_iters = bald.opt_n_iters
    n_samples = bald.n_samples

    print(f"  n_restarts: {n_restarts}")
    print(f"  n_iters: {n_iters}")
    print(f"  n_mc_samples: {n_samples}")
    print(f"  n_joints: {bounds.shape[0]}")

    # Get particles
    particles = learner.posterior.get_particles()
    K = particles.shape[0]
    print(f"  n_particles: {K}")

    # Time individual components
    times = {}

    # 1. Sample from posterior (for particle-based, this is just get_particles)
    t0 = time.perf_counter()
    for _ in range(10):
        zs = learner.posterior.get_particles()
    times['get_particles'] = (time.perf_counter() - t0) / 10 * 1000

    # 2. Decode samples
    t0 = time.perf_counter()
    for _ in range(10):
        decoded_params = LatentFeasibilityChecker.decode_latent_params(decoder, zs)
    times['decode_samples'] = (time.perf_counter() - t0) / 10 * 1000

    # 3. Initialize test points
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    t0 = time.perf_counter()
    for _ in range(10):
        t = lower + torch.rand(n_restarts, bounds.shape[0], device=DEVICE) * (upper - lower)
        t = t.clone().requires_grad_(True)
    times['init_test_points'] = (time.perf_counter() - t0) / 10 * 1000

    # 4. Compute batched score (single call)
    t0 = time.perf_counter()
    for _ in range(10):
        scores = bald.compute_score_batched(t, decoded_params=decoded_params, iteration=0)
    times['compute_score_batched'] = (time.perf_counter() - t0) / 10 * 1000

    # 5. Single optimization step (forward + backward)
    optimizer = torch.optim.Adam([t], lr=0.05)
    t0 = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad()
        scores = bald.compute_score_batched(t, decoded_params=decoded_params, iteration=0)
        (-scores.sum()).backward()
        optimizer.step()
        with torch.no_grad():
            t.data = torch.clamp(t.data, lower, upper)
    times['single_opt_step'] = (time.perf_counter() - t0) / 10 * 1000

    # 6. Full optimization loop (n_iters steps)
    t = lower + torch.rand(n_restarts, bounds.shape[0], device=DEVICE) * (upper - lower)
    t = t.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([t], lr=0.05)

    t0 = time.perf_counter()
    for i in range(n_iters):
        optimizer.zero_grad()
        scores = bald.compute_score_batched(t, decoded_params=decoded_params, iteration=0)
        (-scores.sum()).backward()
        optimizer.step()
        with torch.no_grad():
            t.data = torch.clamp(t.data, lower, upper)
    times['full_opt_loop'] = (time.perf_counter() - t0) * 1000

    # 7. Final re-decode and evaluation
    t0 = time.perf_counter()
    for _ in range(10):
        final_zs = learner.posterior.get_particles()
        final_decoded = LatentFeasibilityChecker.decode_latent_params(decoder, final_zs)
        final_scores = bald.compute_score_batched(t, decoded_params=final_decoded, iteration=0)
    times['final_eval'] = (time.perf_counter() - t0) / 10 * 1000

    # 8. Full select_test call
    t0 = time.perf_counter()
    for _ in range(3):
        best_test, best_score, stats = bald.select_test(bounds, verbose=False, iteration=0)
    times['full_select_test'] = (time.perf_counter() - t0) / 3 * 1000

    print("\nBALD select_test breakdown:")
    print(f"  get_particles:          {times['get_particles']:.2f}ms")
    print(f"  decode_samples:         {times['decode_samples']:.2f}ms")
    print(f"  init_test_points:       {times['init_test_points']:.2f}ms")
    print(f"  compute_score_batched:  {times['compute_score_batched']:.2f}ms")
    print(f"  single_opt_step:        {times['single_opt_step']:.2f}ms")
    print(f"  full_opt_loop ({n_iters} iters): {times['full_opt_loop']:.2f}ms")
    print(f"  final_eval:             {times['final_eval']:.2f}ms")
    print(f"  FULL select_test:       {times['full_select_test']:.2f}ms")

    estimated_total = (
        times['decode_samples'] +  # initial decode
        times['full_opt_loop'] +   # optimization
        times['final_eval']        # final eval
    )
    print(f"\n  Estimated total: {estimated_total:.2f}ms")
    print(f"  Overhead: {times['full_select_test'] - estimated_total:.2f}ms")

    print("\n" + "="*80)
    print("DETAILED SVGD update_posterior PROFILING")
    print("="*80)

    vi = learner.vi
    history = learner.oracle.get_history()
    particles = learner.posterior.get_particles()
    max_iters = vi.max_iters

    print(f"  max_iters: {max_iters}")
    print(f"  n_particles: {particles.shape[0]}")
    print(f"  latent_dim: {particles.shape[1]}")
    print(f"  n_data_points: {len(history.get_all())}")

    svgd_times = {}

    # 1. Log likelihood
    t0 = time.perf_counter()
    for _ in range(10):
        ll = vi.log_likelihood(history, particles, iteration=0)
    svgd_times['log_likelihood'] = (time.perf_counter() - t0) / 10 * 1000

    # 2. Log prior
    t0 = time.perf_counter()
    for _ in range(10):
        lp = vi.log_prior(particles)
    svgd_times['log_prior'] = (time.perf_counter() - t0) / 10 * 1000

    # 3. Gradient computation
    t0 = time.perf_counter()
    for _ in range(10):
        p_in = particles.detach().requires_grad_(True)
        ll = vi.log_likelihood(history, p_in, iteration=0)
        lp = vi.log_prior(p_in)
        ll_grad = torch.autograd.grad(ll.sum(), p_in, retain_graph=True)[0]
        lp_grad = torch.autograd.grad(lp.sum(), p_in)[0]
        log_prob_grad = ll_grad + lp_grad
    svgd_times['gradient_computation'] = (time.perf_counter() - t0) / 10 * 1000

    # 4. SVGD kernel step
    t0 = time.perf_counter()
    for _ in range(10):
        phi = vi.optimizer.step(particles, log_prob_grad)
    svgd_times['kernel_step'] = (time.perf_counter() - t0) / 10 * 1000

    # 5. Single inner iteration
    t0 = time.perf_counter()
    for _ in range(10):
        p_in = particles.detach().requires_grad_(True)
        ll = vi.log_likelihood(history, p_in, iteration=0)
        lp = vi.log_prior(p_in)
        ll_grad = torch.autograd.grad(ll.sum(), p_in, retain_graph=True)[0]
        lp_grad = torch.autograd.grad(lp.sum(), p_in)[0]
        log_prob_grad = ll_grad + lp_grad
        phi = vi.optimizer.step(particles, log_prob_grad)
        with torch.no_grad():
            particles_temp = particles + 0.1 * phi
    svgd_times['single_inner_iter'] = (time.perf_counter() - t0) / 10 * 1000

    # 6. Full update_posterior
    t0 = time.perf_counter()
    for _ in range(3):
        result = vi.update_posterior(history, kl_weight=0.1, diagnostics=None, iteration=10)
    svgd_times['full_update_posterior'] = (time.perf_counter() - t0) / 3 * 1000

    print("\nSVGD update_posterior breakdown:")
    print(f"  log_likelihood:         {svgd_times['log_likelihood']:.2f}ms")
    print(f"  log_prior:              {svgd_times['log_prior']:.2f}ms")
    print(f"  gradient_computation:   {svgd_times['gradient_computation']:.2f}ms")
    print(f"  kernel_step:            {svgd_times['kernel_step']:.2f}ms")
    print(f"  single_inner_iter:      {svgd_times['single_inner_iter']:.2f}ms")
    print(f"  FULL update_posterior:  {svgd_times['full_update_posterior']:.2f}ms")

    estimated_inner = svgd_times['single_inner_iter'] * max_iters
    print(f"\n  Estimated ({max_iters} iters): {estimated_inner:.2f}ms")
    print(f"  Overhead: {svgd_times['full_update_posterior'] - estimated_inner:.2f}ms")

    print("\n" + "="*80)
    print("DIAGNOSTIC BOTTLENECK: log_likelihood breakdown")
    print("="*80)

    # Profile what happens inside log_likelihood
    results = history.get_all()
    n_data = len(results)
    test_points = torch.stack([r.test_point for r in results]).to(particles.device)

    t0 = time.perf_counter()
    for _ in range(10):
        # This is what happens inside log_likelihood
        pred_logits = LatentFeasibilityChecker.batched_logit_values(decoder, particles, test_points)
    ll_breakdown_decode = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(10):
        # Rest of log_likelihood
        outcomes = torch.tensor([r.outcome for r in results], device=particles.device).unsqueeze(0)
        pred_logits = LatentFeasibilityChecker.batched_logit_values(decoder, particles, test_points)
        scaled_logits = pred_logits / 0.3
        targets_expanded = outcomes.expand_as(scaled_logits)
        neg_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            scaled_logits, targets_expanded, reduction='none')
        log_lik = -neg_bce.sum(dim=1)
    ll_breakdown_full = (time.perf_counter() - t0) / 10 * 1000

    print(f"\n  batched_logit_values (K={particles.shape[0]}, N_data={n_data}): {ll_breakdown_decode:.2f}ms")
    print(f"  full log_likelihood: {ll_breakdown_full:.2f}ms")
    print(f"  BCE overhead: {ll_breakdown_full - ll_breakdown_decode:.2f}ms")

    # How much does log_likelihood scale with data?
    print("\n  Scaling with N_data:")
    for n in [1, 5, 10, 20]:
        if n > n_data:
            break
        test_subset = test_points[:n]
        t0 = time.perf_counter()
        for _ in range(10):
            pred_logits = LatentFeasibilityChecker.batched_logit_values(decoder, particles, test_subset)
        t_n = (time.perf_counter() - t0) / 10 * 1000
        print(f"    N_data={n}: {t_n:.2f}ms")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_iter = times['full_select_test'] + svgd_times['full_update_posterior']
    print(f"\nEstimated iteration time: {total_iter:.0f}ms")
    print(f"  - BALD select_test:    {times['full_select_test']:.0f}ms ({times['full_select_test']/total_iter*100:.0f}%)")
    print(f"  - SVGD update:         {svgd_times['full_update_posterior']:.0f}ms ({svgd_times['full_update_posterior']/total_iter*100:.0f}%)")

    print("\nKey bottlenecks:")
    print(f"  1. BALD optimization loop: {times['full_opt_loop']:.0f}ms ({n_iters} iters × {times['single_opt_step']:.1f}ms)")
    print(f"  2. SVGD inner loop: {svgd_times['full_update_posterior']:.0f}ms ({max_iters} iters × {svgd_times['single_inner_iter']:.1f}ms)")
    print(f"  3. Decoder calls in log_likelihood: {ll_breakdown_decode:.1f}ms per call × {max_iters} = {ll_breakdown_decode * max_iters:.0f}ms")


if __name__ == "__main__":
    main()
