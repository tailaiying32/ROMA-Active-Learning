
import sys
import os
import torch
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.config import DEVICE, load_config, get_bounds_from_config
from active_learning.test.diagnostics.utils import load_diagnostic_model
from active_learning.src.latent_bald import LatentBALD
from active_learning.src.latent_user_distribution import LatentUserDistribution

def benchmark_bald(model_path=None, n_iterations=5):
    print(f"\n--- Benchmarking Latent BALD ---")
    
    # 1. Setup Environment
    config = load_config()
    
    # Override some config settings for consistency
    config['bald']['n_mc_samples'] = 50
    config['bald_optimization']['n_restarts'] = 5
    config['bald_optimization']['n_iters_per_restart'] = 50
    
    if model_path is None:
        model_path = r"c:\Users\Tailai\Stuff\emprise\ROMA\models\best_model.pt"
        
    print(f"Loading model from: {model_path}")
    decoder, _, embeddings = load_diagnostic_model(model_path)
    bounds = get_bounds_from_config(config, DEVICE)
    
    # Create fake posterior (centered at one embedding)
    gt_z = embeddings[0].clone()
    posterior = LatentUserDistribution(
        latent_dim=gt_z.shape[0],
        decoder=decoder,
        device=DEVICE
    )
    # Initialize near gt_z
    with torch.no_grad():
        posterior.mean.copy_(gt_z)
        posterior.log_std.fill_(np.log(0.1))
        
    bald = LatentBALD(decoder, posterior, config)
    
    print(f"Configuration:")
    print(f"  MC Samples: {bald.n_samples}")
    print(f"  Restarts: {bald.opt_n_restarts}")
    print(f"  Iters/Restart: {bald.opt_n_iters}")
    print(f"  Device: {DEVICE}")
    
    # Warmup
    print("Warming up...")
    bald.select_test(bounds, n_restarts=1, n_iters=10)
    
    # Benchmark
    times = []
    print(f"Running {n_iterations} benchmarks...")
    for i in range(n_iterations):
        start_time = time.time()
        bald.select_test(bounds) # Uses config defaults
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        duration = end_time - start_time
        times.append(duration)
        print(f"  Run {i+1}: {duration:.4f}s")
        
    avg_time = sum(times) / len(times)
    print(f"\nAverage Time: {avg_time:.4f}s")
    return avg_time

if __name__ == "__main__":
    import numpy as np
    benchmark_bald()
