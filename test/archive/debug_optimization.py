
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from active_learning.src.latent_active_learning import LatentActiveLearner
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from infer_params.training.model import LevelSetDecoder
from active_learning.src.config import load_config, DEVICE
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.latent_prior_generation import LatentPriorGenerator

def run_debug_optimization(sample_index=42, budget=15):
    print(f"=== Optimization Debug Run (Seed {sample_index}) ===")
    config = load_config('active_learning/configs/default.yaml')
    config['stopping']['budget'] = budget
    config['stopping']['budget'] = budget
    config['update_posterior'] = True
    
    # Fix relative paths from config if running from root
    if config['latent']['model_path'].startswith('../'):
         config['latent']['model_path'] = config['latent']['model_path'].replace('../', '')
    if config['latent']['dataset_path'].startswith('../'):
         config['latent']['dataset_path'] = config['latent']['dataset_path'].replace('../', '')

    device = DEVICE
    
    # 1. Load Model
    try:
        model_path = config['latent']['model_path']
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)
            
        print(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract config and dims
        train_config = checkpoint['config']
        model_cfg = train_config['model']
        num_samples = checkpoint['embeddings'].shape[0]
        
        decoder = LevelSetDecoder(
            num_samples=num_samples,
            latent_dim=config['latent']['latent_dim'], 
            hidden_dim=model_cfg.get('hidden_dim', 256), 
            num_blocks=model_cfg.get('num_blocks', 3),
            num_slots=model_cfg.get('num_slots', 18),
            params_per_slot=model_cfg.get('params_per_slot', 6)
        ).to(device)
        
        decoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully (epoch {checkpoint['epoch']}).")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Get GT
    try:
        # Get z_gt from loaded embeddings (as per test_latent_active_learning.py)
        embeddings = checkpoint['embeddings']
        if sample_index >= embeddings.shape[0]:
            print(f"Sample index {sample_index} out of bounds (max {embeddings.shape[0]-1})")
            return
            
        z_gt = embeddings[sample_index].clone().to(device)
        print(f"GT z norm: {z_gt.norm():.4f}")
        
    except Exception as e:
        print(f"Failed to get GT from embeddings: {e}")
        return

    # 3. Setup Components
    # Use PriorGenerator to match test scripts exactly
    from active_learning.src.config import get_bounds_from_config
    
    # 6. Generate prior (perturbed ground truth)
    print("Generating prior distribution...")
    prior_gen = LatentPriorGenerator(config, decoder)
    prior = prior_gen.get_prior(z_gt) # Returns prior LatentUserDistribution

    # 7. Initialize posterior as copy of prior
    posterior = LatentUserDistribution(
        latent_dim=config['latent']['latent_dim'],
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=device
    )
    
    # 9. Get bounds for test point optimization
    bounds = get_bounds_from_config(config, device)
    
    print(f"Prior Initialized. Mean Norm: {prior.mean.norm():.4f}")
    
    gt_checker = LatentFeasibilityChecker(decoder, z_gt)
    oracle = LatentOracle(
        decoder=decoder,
        ground_truth_z=z_gt,
        n_joints=4
    )
    
    learner = LatentActiveLearner(
        decoder=decoder,
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        bounds=bounds,
        config=config
    )
    
    # Storage for analysis
    diagnostics = {
        'iter': [],
        'kl_weight': [],
        'mean_grad_norm_avg': [], # Average over dimensions
        'mean_grad_norm_max': [],
        'param_error_norm': [],
        'z_mean_trajectories': [], # (iter, dim)
        'z_gt': z_gt.cpu().numpy(),
        'likelihood_term': [],
        'kl_term': []
    }
    
    # Detailed per-iteration VI logs
    vi_details = []

    print("Starting loop...")
    
    output_dir = 'active_learning/images/debug_optimization'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(budget):
        print(f"--- Iteration {i} ---")
        result = learner.step(verbose=False)
        vi_res = learner.last_vi_result
        
        # Calculate current KL weight
        kl_weight = learner._calculate_kl_weight(i)
        
        # Store high level metrics
        z_curr = posterior.mean.detach().cpu().numpy()
        z_err = np.linalg.norm(z_curr - z_gt.cpu().numpy())
        
        diagnostics['iter'].append(i)
        diagnostics['kl_weight'].append(kl_weight)
        diagnostics['param_error_norm'].append(z_err)
        diagnostics['z_mean_trajectories'].append(z_curr.copy())
        
        # Analyze gradients from this VI step
        # vi_res.mean_grad_history is (n_vi_iters, latent_dim)
        if vi_res.mean_grad_history is not None and len(vi_res.mean_grad_history) > 0:
            final_grads = vi_res.mean_grad_history[-1] # Gradients at convergence/end
            diagnostics['mean_grad_norm_avg'].append(np.mean(np.abs(final_grads)))
            diagnostics['mean_grad_norm_max'].append(np.max(np.abs(final_grads)))
        else:
            diagnostics['mean_grad_norm_avg'].append(0)
            diagnostics['mean_grad_norm_max'].append(0)

        vi_details.append({
            'iter': i,
            'mean_history': vi_res.mean_history,
            'mean_grad_history': vi_res.mean_grad_history,
            'elbo_history': vi_res.elbo_history
        })
        
        print(f"  KL_W={kl_weight:.4f}, z_err={z_err:.4f}, final_grad_avg={diagnostics['mean_grad_norm_avg'][-1]:.6f}")

    # === PLOTTING ===
    
    # 1. Parameter Error & KL Weight
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(diagnostics['iter'], diagnostics['param_error_norm'], 'b-', label='Latent Error (L2)', marker='o')
    ax2.plot(diagnostics['iter'], diagnostics['kl_weight'], 'r--', label='KL Weight', alpha=0.5)
    
    ax1.set_xlabel('Active Learning Iteration')
    ax1.set_ylabel('Latent Error')
    ax2.set_ylabel('KL Weight')
    plt.title('Latent Error vs KL Annealing')
    plt.savefig(os.path.join(output_dir, 'error_vs_annealing.png'))
    plt.close()
    
    # 2. Gradient Magnitudes
    plt.figure(figsize=(10, 6))
    plt.plot(diagnostics['iter'], diagnostics['mean_grad_norm_avg'], label='Avg Param Gradient')
    plt.plot(diagnostics['iter'], diagnostics['mean_grad_norm_max'], label='Max Param Gradient')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Magnitude (Log Scale)')
    plt.title('Gradient Magnitudes at Convergence per Iteration')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'gradients.png'))
    plt.close()
    
    # 3. Trajectory of Top 3 Error Dimensions
    # Find dimensions with highest error at the end
    z_final = diagnostics['z_mean_trajectories'][-1]
    z_gt_np = diagnostics['z_gt']
    errors = np.abs(z_final - z_gt_np)
    top_idxs = np.argsort(errors)[-3:][::-1]
    
    trajectories = np.array(diagnostics['z_mean_trajectories']) # (n_iters, latent_dim)
    
    plt.figure(figsize=(12, 8))
    for idx in top_idxs:
        plt.plot(diagnostics['iter'], trajectories[:, idx], label=f'Dim {idx} Est')
        plt.axhline(y=z_gt_np[idx], linestyle='--', alpha=0.5, label=f'Dim {idx} GT')
        
    plt.xlabel('Iteration')
    plt.ylabel('Latent Value')
    plt.title('Trajectories of Worst Dimensions')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'worst_dims.png'))
    plt.close()

    print(f"Analysis complete. Plots saved to {output_dir}")
    
    # Text Analysis
    print("\n=== Worst Dimensions Analysis ===")
    for idx in top_idxs:
        final_val = z_final[idx]
        gt_val = z_gt_np[idx]
        err = np.abs(final_val - gt_val)
        
        # Get Average gradient for this dimension over last 5 iters
        # flattened grads: all_mean_grads is not available directly, need to extract from vi_details
        # vi_details has 'mean_grad_history' for each AL iter.
        # We want grad from the last AL iter, averaged over ITS VI steps.
        last_vi = vi_details[-1]
        last_grads = last_vi['mean_grad_history'] # (n_vi_iters, latent_dim)
        avg_grad = np.mean(last_grads[:, idx])
        
        print(f"Dim {idx}: GT={gt_val:.4f}, Pred={final_val:.4f}, Err={err:.4f}, MeanGrad={avg_grad:.6f}")
        
        # Check if parameter moved at all from Prior (Init)
        # We need init value. z_mean_trajectories[0] is after 1st update.
        # But we can look at trajectory range.
        traj = trajectories[:, idx]
        moved = np.max(traj) - np.min(traj)
        print(f"  Moved range: {moved:.4f}")

if __name__ == "__main__":
    run_debug_optimization()
