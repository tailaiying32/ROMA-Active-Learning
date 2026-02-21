
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_active_learning import LatentActiveLearner, LatentIterationResult
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.latent_variational_inference import LatentVariationalInference, LatentVIResult
from active_learning.test.test_latent_active_learning import load_decoder_model, load_ground_truth_from_dataset, generate_prior_from_ground_truth, get_bounds_from_config

# --- Monkeypatch LatentVariationalInference to log gradients ---

original_update_posterior = LatentVariationalInference.update_posterior

def debug_update_posterior(self, test_history):
    print(f"\n--- Debug Update Posterior (History size: {len(test_history.get_all())}) ---")
    print(f"  tau: {self.tau}")
    
    # Check predictions vs targets for checking scale mismatch
    with torch.no_grad():
        results = test_history.get_all()
        if results:
            test_points = torch.stack([r.test_point for r in results]).to(self.posterior.device)
            outcomes = torch.tensor([r.outcome for r in results], device=self.posterior.device).unsqueeze(0)
            
            # Use MEAN of posterior to check current "center" prediction
            z_mean = self.posterior.mean.unsqueeze(0) # (1, latent_dim)
            logits = LatentFeasibilityChecker.batched_logit_values(self.decoder, z_mean, test_points)
            
            pred_probs = torch.sigmoid(logits / self.tau)
            target_probs = torch.sigmoid(outcomes)
            
            print("  Predictions at Mean vs Targets:")
            for i in range(len(results)):
                l = logits[0, i].item()
                o = outcomes[0, i].item()
                p = pred_probs[0, i].item()
                t = target_probs[0, i].item()
                print(f"    Point {i}: Logit={l:.4f}, Outcome={o:.4f} | Prob={p:.4f}, Target={t:.4f} | Diff={abs(p-t):.4f}")

    # Call original but capture gradients?
    # Actually, let's copy the loop to inject printing
    
    self.posterior.mean.requires_grad_(True)
    self.posterior.log_std.requires_grad_(True)
    params = [self.posterior.mean, self.posterior.log_std]
    optimizer = torch.optim.Adam(params, lr=self.lr)
    
    # Store initial ELBO
    with torch.no_grad():
        init_ll = self.likelihood(test_history).item()
        init_kl = self.regularizer().item()
        print(f"  Init ELBO: {init_ll - init_kl:.4f} (LL={init_ll:.4f}, KL={init_kl:.4f})")

    grad_norm_history = []
    elbo_history = []
    final_grad_norm = 0.0
    
    # Run a few steps verbose
    print("  Optimization Trace:")
    for i in range(self.max_iters):
        optimizer.zero_grad()
        
        # breakdown likelihood
        # We need to call likelihood which uses sampling
        # Let's inspect gradients of likelihood vs KL
        
        # 1. KL
        kl_div = self.regularizer()
        
        # 2. LL
        log_likelihood = self.likelihood(test_history)
        
        elbo = log_likelihood - kl_div
        loss = -elbo
        loss.backward()
        
        # Log gradients
        mean_grad_norm = self.posterior.mean.grad.norm().item()
        std_grad_norm = self.posterior.log_std.grad.norm().item()
        
        if i < 5 or i % 20 == 0:
            print(f"    Iter {i}: ELBO={elbo.item():.4f} (LL={log_likelihood.item():.4f}, KL={kl_div.item():.4f}) | G_Mean={mean_grad_norm:.4f}, G_Std={std_grad_norm:.4f}")
            
        # ... standard update ...
        total_grad_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = np.sqrt(total_grad_norm)
        grad_norm_history.append(total_grad_norm)
        final_grad_norm = total_grad_norm

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

        optimizer.step()

        with torch.no_grad():
            self._clamp_params()
            
        elbo_history.append(elbo.item())
        
    return LatentVIResult(
        converged=False, # simplfied
        n_iterations=self.max_iters,
        final_elbo=elbo_history[-1],
        final_grad_norm=final_grad_norm,
        grad_norm_history=grad_norm_history,
        elbo_history=elbo_history
    )

LatentVariationalInference.update_posterior = debug_update_posterior

# --- Main Run ---

def run_debug():
    print("Running Debug Latent AL...")
    model_path = 'models/best_model.pt'
    dataset_path = 'models/training_data.npz'
    
    config = load_config()
    # Force consistent seed if needed, but keeping random for now
    
    print(f"Config loaded. Tau = {config.get('bald', {}).get('tau', 'N/A')}")
    
    decoder, embeddings, train_config = load_decoder_model(model_path, DEVICE)
    
    # Pick a random sample
    # sample_idx, ground_truth_params = load_ground_truth_from_dataset(dataset_path)
    # Use a fixed sample for reproducibility if possible?
    # Let's pick index 0
    sample_idx = 0
    print(f"Using sample index {sample_idx}")
    
    ground_truth_z = embeddings[sample_idx].clone()
    
    # Prior
    prior = generate_prior_from_ground_truth(ground_truth_z, decoder, config, DEVICE)
    
    # Posterior
    posterior = LatentUserDistribution(
        latent_dim=prior.mean.shape[0],
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )
    
    # Oracle
    oracle = LatentOracle(decoder, ground_truth_z, n_joints=4)
    
    # Learner
    bounds = get_bounds_from_config(config, DEVICE)
    learner = LatentActiveLearner(decoder, prior, posterior, oracle, bounds, config)
    
    # Run 5 steps
    print("\nStarting Loop (5 steps)...")
    for i in range(5):
        learner.step(verbose=True)

if __name__ == '__main__':
    run_debug()
