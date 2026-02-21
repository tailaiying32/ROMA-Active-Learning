"""
Debug script to analyze BALD vs Random pipeline behavior in depth.

Tracks and visualizes:
- Posterior mean movement (gradient magnitude and direction)
- ELBO components (log-likelihood, KL divergence)
- Selected test points and their properties
- Oracle outputs (logit values)
- Posterior uncertainty evolution
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from active_learning.src.latent_user_distribution import LatentUserDistribution
from active_learning.src.latent_oracle import LatentOracle
from active_learning.src.latent_active_learning import LatentActiveLearner
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker
from active_learning.src.latent_prior_generation import LatentPriorGenerator
from active_learning.src.baselines.random_strategy import RandomStrategy
from active_learning.src.latent_bald import LatentBALD
from active_learning.src.latent_variational_inference import LatentVariationalInference
from active_learning.src.test_history import TestHistory

from infer_params.training.model import LevelSetDecoder
from infer_params.training.dataset import LevelSetDataset


def load_decoder_model(checkpoint_path: str, device: str = DEVICE):
    """Load the LevelSetDecoder model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    return model, embeddings.to(device), train_config


def run_diagnostic_trial(
    decoder, 
    embeddings, 
    train_config, 
    sample_idx: int, 
    budget: int, 
    use_random: bool, 
    config: dict,
    n_joints: int = 4
):
    """
    Run a single trial with detailed diagnostics.
    """
    latent_dim = train_config['model']['latent_dim']
    
    # Get ground truth
    ground_truth_z = embeddings[sample_idx].clone()
    
    # Create ground truth checker
    ground_truth_checker = LatentFeasibilityChecker(
        decoder=decoder,
        z=ground_truth_z,
        device=DEVICE
    )
    
    # Get true joint limits
    lower, upper = ground_truth_checker.joint_limits()
    
    # Generate prior
    prior_gen = LatentPriorGenerator(config, decoder)
    prior = prior_gen.get_prior(ground_truth_z)
    
    # Initialize posterior
    posterior = LatentUserDistribution(
        latent_dim=latent_dim,
        decoder=decoder,
        mean=prior.mean.clone(),
        log_std=prior.log_std.clone(),
        device=DEVICE
    )
    
    # Setup oracle
    oracle = LatentOracle(
        decoder=decoder,
        ground_truth_z=ground_truth_z,
        n_joints=n_joints
    )
    
    # Get bounds
    bounds = get_bounds_from_config(config, DEVICE)
    
    # Create learner
    if use_random:
        random_strategy = RandomStrategy(config=config)
        learner = LatentActiveLearner(
            decoder=decoder,
            prior=prior,
            posterior=posterior,
            oracle=oracle,
            bounds=bounds,
            config=config,
            acquisition_strategy=random_strategy
        )
    else:
        learner = LatentActiveLearner(
            decoder=decoder,
            prior=prior,
            posterior=posterior,
            oracle=oracle,
            bounds=bounds,
            config=config
        )
    
    # Diagnostic tracking
    diagnostics = {
        'iterations': [],
        'posterior_means': [],
        'posterior_stds': [],
        'test_points': [],
        'oracle_outputs': [],
        'bald_scores': [],
        'elbos': [],
        'log_likelihoods': [],
        'kl_divs': [],
        'mean_grads': [],
        'std_grads': [],
        'latent_errors': [],
        'move_magnitudes': [],  # How much posterior mean moved
        'move_directions': [],  # Direction of movement in latent space
    }
    
    # Initial state
    initial_mean = posterior.mean.detach().clone()
    diagnostics['iterations'].append(0)
    diagnostics['posterior_means'].append(initial_mean.cpu().numpy())
    diagnostics['posterior_stds'].append(torch.exp(posterior.log_std).detach().cpu().numpy())
    diagnostics['latent_errors'].append(
        float(np.linalg.norm(initial_mean.cpu().numpy() - ground_truth_z.cpu().numpy()))
    )
    diagnostics['test_points'].append(None)
    diagnostics['oracle_outputs'].append(None)
    diagnostics['bald_scores'].append(None)
    diagnostics['elbos'].append(None)
    diagnostics['log_likelihoods'].append(None)
    diagnostics['kl_divs'].append(None)
    diagnostics['mean_grads'].append(None)
    diagnostics['std_grads'].append(None)
    diagnostics['move_magnitudes'].append(0.0)
    diagnostics['move_directions'].append(None)
    
    gt_z = ground_truth_z.cpu().numpy()
    prev_mean = initial_mean.cpu().numpy()
    
    print(f"\n{'='*60}")
    print(f"Running {'RANDOM' if use_random else 'BALD'} trial (sample {sample_idx})")
    print(f"{'='*60}")
    print(f"Initial latent error: {diagnostics['latent_errors'][0]:.4f}")
    print(f"Initial posterior std: {diagnostics['posterior_stds'][0].mean():.4f}")
    print(f"{'='*60}\n")
    
    # Run iterations
    for i in range(budget):
        # Store pre-update mean
        pre_update_mean = posterior.mean.detach().clone()
        
        # Execute step
        result = learner.step(verbose=False)
        
        # Compute diagnostics
        curr_mean = posterior.mean.detach().cpu().numpy()
        curr_std = torch.exp(posterior.log_std).detach().cpu().numpy()
        
        # Movement analysis
        move = curr_mean - prev_mean
        move_mag = float(np.linalg.norm(move))
        ideal_direction = gt_z - prev_mean
        ideal_mag = np.linalg.norm(ideal_direction)
        
        if ideal_mag > 1e-8 and move_mag > 1e-8:
            # Cosine similarity with ideal direction
            cos_sim = float(np.dot(move, ideal_direction) / (move_mag * ideal_mag))
        else:
            cos_sim = 0.0
        
        prev_mean = curr_mean
        
        # Record
        diagnostics['iterations'].append(i + 1)
        diagnostics['posterior_means'].append(curr_mean)
        diagnostics['posterior_stds'].append(curr_std)
        diagnostics['test_points'].append(result.test_point.cpu().numpy())
        diagnostics['oracle_outputs'].append(result.outcome)
        diagnostics['bald_scores'].append(result.bald_score)
        diagnostics['elbos'].append(result.elbo)
        diagnostics['mean_grads'].append(result.grad_norm)
        diagnostics['std_grads'].append(None)  # Not tracked separately
        
        # Compute LL and KL separately
        history = oracle.get_history()
        ll = learner.vi.likelihood(history).item()
        kl = learner.vi.regularizer().item()
        diagnostics['log_likelihoods'].append(ll)
        diagnostics['kl_divs'].append(kl)
        
        latent_err = float(np.linalg.norm(curr_mean - gt_z))
        diagnostics['latent_errors'].append(latent_err)
        diagnostics['move_magnitudes'].append(move_mag)
        diagnostics['move_directions'].append(cos_sim)
        
        # Print per-iteration summary
        outcome_str = "FEASIBLE" if result.outcome >= 0 else "INFEASIBLE"
        print(f"[{i+1:2d}] Oracle: {result.outcome:+.3f} ({outcome_str}), "
              f"BALD: {result.bald_score:.4f}, "
              f"Err: {latent_err:.3f}, "
              f"Move: {move_mag:.4f} (cos={cos_sim:+.3f}), "
              f"LL: {ll:.2f}, KL: {kl:.2f}")
    
    # Summary
    final_err = diagnostics['latent_errors'][-1]
    initial_err = diagnostics['latent_errors'][0]
    improvement = (initial_err - final_err) / initial_err * 100
    
    print(f"\n{'='*60}")
    print(f"Trial Summary:")
    print(f"  Initial Error: {initial_err:.4f}")
    print(f"  Final Error:   {final_err:.4f}")
    print(f"  Improvement:   {improvement:.1f}%")
    print(f"  Avg. Move Mag: {np.mean(diagnostics['move_magnitudes'][1:]):.4f}")
    print(f"  Avg. Move Cos: {np.mean([d for d in diagnostics['move_directions'][1:] if d is not None]):.3f}")
    print(f"{'='*60}\n")
    
    return diagnostics


def analyze_diagnostics(bald_diag, random_diag, save_dir: str):
    """Create plots comparing BALD vs Random diagnostics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Latent Error Evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bald_diag['iterations'], bald_diag['latent_errors'], 'b-o', label='BALD', linewidth=2, markersize=4)
    ax.plot(random_diag['iterations'], random_diag['latent_errors'], 'r-o', label='Random', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Latent Error (L2)')
    ax.set_title('Latent Error Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_latent_error.png'), dpi=150)
    plt.close()
    
    # 2. Move Magnitude (how much posterior moved each iteration)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bald_diag['iterations'][1:], bald_diag['move_magnitudes'][1:], 'b-o', label='BALD', linewidth=2, markersize=4)
    ax.plot(random_diag['iterations'][1:], random_diag['move_magnitudes'][1:], 'r-o', label='Random', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Posterior Mean Move Magnitude')
    ax.set_title('How Much Posterior Moved Each Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_move_magnitude.png'), dpi=150)
    plt.close()
    
    # 3. Move Direction (cosine similarity with ideal)
    fig, ax = plt.subplots(figsize=(10, 6))
    bald_dirs = [d for d in bald_diag['move_directions'][1:] if d is not None]
    random_dirs = [d for d in random_diag['move_directions'][1:] if d is not None]
    ax.plot(range(1, len(bald_dirs)+1), bald_dirs, 'b-o', label='BALD', linewidth=2, markersize=4)
    ax.plot(range(1, len(random_dirs)+1), random_dirs, 'r-o', label='Random', linewidth=2, markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity with Ideal Direction')
    ax.set_title('Direction Quality (1=perfect, 0=orthogonal, -1=opposite)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_move_direction.png'), dpi=150)
    plt.close()
    
    # 4. ELBO Components
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ELBO
    ax = axes[0]
    bald_elbos = [e for e in bald_diag['elbos'][1:] if e is not None]
    random_elbos = [e for e in random_diag['elbos'][1:] if e is not None]
    ax.plot(range(1, len(bald_elbos)+1), bald_elbos, 'b-o', label='BALD', linewidth=2, markersize=4)
    ax.plot(range(1, len(random_elbos)+1), random_elbos, 'r-o', label='Random', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ELBO')
    ax.set_title('ELBO')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log-Likelihood
    ax = axes[1]
    bald_lls = [e for e in bald_diag['log_likelihoods'][1:] if e is not None]
    random_lls = [e for e in random_diag['log_likelihoods'][1:] if e is not None]
    ax.plot(range(1, len(bald_lls)+1), bald_lls, 'b-o', label='BALD', linewidth=2, markersize=4)
    ax.plot(range(1, len(random_lls)+1), random_lls, 'r-o', label='Random', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Log-Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # KL Divergence
    ax = axes[2]
    bald_kls = [e for e in bald_diag['kl_divs'][1:] if e is not None]
    random_kls = [e for e in random_diag['kl_divs'][1:] if e is not None]
    ax.plot(range(1, len(bald_kls)+1), bald_kls, 'b-o', label='BALD', linewidth=2, markersize=4)
    ax.plot(range(1, len(random_kls)+1), random_kls, 'r-o', label='Random', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_elbo_components.png'), dpi=150)
    plt.close()
    
    # 5. Oracle Outputs Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    bald_outs = [o for o in bald_diag['oracle_outputs'][1:] if o is not None]
    random_outs = [o for o in random_diag['oracle_outputs'][1:] if o is not None]
    
    ax = axes[0]
    ax.hist(bald_outs, bins=20, alpha=0.7, color='blue', label=f'BALD (n={len(bald_outs)})')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Feasibility Boundary')
    ax.set_xlabel('Oracle Output (Logit)')
    ax.set_ylabel('Count')
    ax.set_title('BALD: Oracle Output Distribution')
    ax.legend()
    
    ax = axes[1]
    ax.hist(random_outs, bins=20, alpha=0.7, color='red', label=f'Random (n={len(random_outs)})')
    ax.axvline(x=0, color='blue', linestyle='--', linewidth=2, label='Feasibility Boundary')
    ax.set_xlabel('Oracle Output (Logit)')
    ax.set_ylabel('Count')
    ax.set_title('Random: Oracle Output Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '5_oracle_outputs.png'), dpi=150)
    plt.close()
    
    # 6. BALD Scores
    fig, ax = plt.subplots(figsize=(10, 6))
    bald_scores = [s for s in bald_diag['bald_scores'][1:] if s is not None]
    random_scores = [s for s in random_diag['bald_scores'][1:] if s is not None]
    ax.plot(range(1, len(bald_scores)+1), bald_scores, 'b-o', label='BALD', linewidth=2, markersize=4)
    ax.plot(range(1, len(random_scores)+1), random_scores, 'r-o', label='Random (computed)', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('BALD Score')
    ax.set_title('BALD Scores of Selected Tests')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '6_bald_scores.png'), dpi=150)
    plt.close()
    
    # 7. Posterior Uncertainty Evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    bald_stds = [s.mean() for s in bald_diag['posterior_stds']]
    random_stds = [s.mean() for s in random_diag['posterior_stds']]
    ax.plot(bald_diag['iterations'], bald_stds, 'b-o', label='BALD', linewidth=2, markersize=4)
    ax.plot(random_diag['iterations'], random_stds, 'r-o', label='Random', linewidth=2, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Posterior Std')
    ax.set_title('Posterior Uncertainty Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '7_uncertainty.png'), dpi=150)
    plt.close()
    
    print(f"Saved diagnostic plots to {save_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug BALD vs Random pipeline")
    parser.add_argument("--budget", type=int, default=20, help="Query budget")
    parser.add_argument("--sample-idx", type=int, default=None, help="Dataset sample index")
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Model path")
    parser.add_argument("--dataset", type=str, default="models/training_data.npz", help="Dataset path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model
    decoder, embeddings, train_config = load_decoder_model(args.model)
    print(f"Loaded decoder with {embeddings.shape[0]} samples, latent_dim={embeddings.shape[1]}")
    
    # Load config
    config = load_config()
    config['seed'] = args.seed  # Ensure reproducible prior generation
    
    # Pick sample
    if args.sample_idx is None:
        sample_idx = np.random.randint(0, embeddings.shape[0])
    else:
        sample_idx = args.sample_idx
    print(f"Using sample index: {sample_idx}")
    
    # Run BALD trial
    bald_diag = run_diagnostic_trial(
        decoder, embeddings, train_config, 
        sample_idx, args.budget, 
        use_random=False, config=config.copy()
    )
    
    # Reset seed for fair comparison
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config['seed'] = args.seed
    
    # Run Random trial
    random_diag = run_diagnostic_trial(
        decoder, embeddings, train_config, 
        sample_idx, args.budget, 
        use_random=True, config=config.copy()
    )
    
    # Analyze and plot
    save_dir = "active_learning/images/debug_bald"
    analyze_diagnostics(bald_diag, random_diag, save_dir)
    
    # Print key comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Metric':<25} {'BALD':>12} {'Random':>12}")
    print("-"*60)
    print(f"{'Final Latent Error':<25} {bald_diag['latent_errors'][-1]:>12.4f} {random_diag['latent_errors'][-1]:>12.4f}")
    print(f"{'Avg Move Magnitude':<25} {np.mean(bald_diag['move_magnitudes'][1:]):>12.4f} {np.mean(random_diag['move_magnitudes'][1:]):>12.4f}")
    print(f"{'Avg Move Direction':<25} {np.mean([d for d in bald_diag['move_directions'][1:] if d is not None]):>12.3f} {np.mean([d for d in random_diag['move_directions'][1:] if d is not None]):>12.3f}")
    print(f"{'Final ELBO':<25} {bald_diag['elbos'][-1]:>12.2f} {random_diag['elbos'][-1]:>12.2f}")
    print(f"{'Final Uncertainty':<25} {np.mean(bald_diag['posterior_stds'][-1]):>12.4f} {np.mean(random_diag['posterior_stds'][-1]):>12.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
