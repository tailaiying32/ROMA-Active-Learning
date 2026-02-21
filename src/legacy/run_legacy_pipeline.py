import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from active_learning.src.config import load_config, DEVICE
from active_learning.src.legacy.prior_generation import PriorGenerator
from active_learning.src.legacy.sample_user import UserGenerator
from active_learning.src.legacy.user_distribution import UserDistribution
from active_learning.src.legacy.oracle import Oracle
from active_learning.src.legacy.active_learning_pipeline import ActiveLearner
from active_learning.test.visualization_utils import Visualizer, plot_comparison, legacy_to_metadata

def main():
    # 1. Load Configuration
    # We use legacy.yaml which we updated
    config_path = project_root / "active_learning" / "configs" / "legacy.yaml"
    print(f"Loading config from {config_path}")
    config = load_config(str(config_path))

    # 2. Setup Prior
    print("Initializing Prior...")
    prior_gen = PriorGenerator(config)
    prior = prior_gen.get_prior()
    
    # 3. Generate Ground Truth User
    print("Generating Ground Truth User...")
    user_gen = UserGenerator(
        config,
        prior_gen.joint_names,
        prior_gen.anatomical_limits,
        prior_gen.pairs
    )
    true_limits, true_bumps, true_checker = user_gen.generate_user()
    
    print("Ground Truth Limits:")
    for j, lim in true_limits.items():
        print(f"  {j}: {lim}")

    # 4. Initialize Posterior (starts same as Prior)
    # We create a new instance to be independent
    posterior = prior_gen.get_prior()

    # 5. Initialize Oracle
    oracle = Oracle(true_checker, prior_gen.joint_names)

    # 6. Initialize Active Learner
    learner = ActiveLearner(
        prior=prior,
        posterior=posterior,
        oracle=oracle,
        config=config
    )

    # 7. Initialize Visualizer
    output_dir = project_root / "active_learning" / "images" / "legacy"
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Create test grid for metrics
    grid_res = config.get('metrics', {}).get('grid_resolution', 12)
    from infer_params.training.level_set_torch import create_evaluation_grid
    def to_tensor(x): return torch.tensor(x, device=DEVICE, dtype=torch.float32)
    
    # Use anatomical limits for grid bounds
    grid_lowers = []
    grid_uppers = []
    for j in prior_gen.joint_names:
        l, u = prior_gen.anatomical_limits[j]
        grid_lowers.append(l)
        grid_uppers.append(u)
    
    test_grid_tensor = create_evaluation_grid(to_tensor(grid_lowers), to_tensor(grid_uppers), grid_res, DEVICE)
    test_grid_np = test_grid_tensor.cpu().numpy()

    visualizer = Visualizer(
        save_dir=str(output_dir), 
        joint_names=prior_gen.joint_names, 
        true_limits=true_limits
    )

    # 8. Run Active Learning Loop
    budget = config.get('stopping', {}).get('budget', 20)
    print(f"Starting Active Learning Loop (Budget: {budget})...")

    # Initial log (iteration 0 - state before queries? No, Visualizer logs outcomes.
    # We usually log AFTER a step. 
    # But we can log initial state if we want. Visualizer.log_iteration expects a result.
    # So we just loop.
    
    for i in range(budget):
        print(f"\n--- Iteration {i+1}/{budget} ---")
        
        # Step
        result = learner.step(verbose=True)
        
        # Log
        visualizer.log_iteration(
            iteration=i+1,
            posterior=posterior,
            result=result,
            prior=prior,
            true_checker=true_checker,
            test_grid=test_grid_np
        )
        
        # Check stopping
        should_stop, reason = learner.check_stopping_criteria()
        if should_stop:
            print(f"Stopping criteria met: {reason}")
            break

    # 9. Final Visualization
    print("\nGenerating Visualizations...")
    visualizer.save_history()
    visualizer.plot_information_gain()
    visualizer.plot_metrics()
    visualizer.plot_param_error_over_time()
    visualizer.plot_reachability_evolution()
    visualizer.plot_consolidated_metrics()
    
    # Cap to anatomical if config says so
    cap_anatomical = config.get('visualization', {}).get('cap_joint_evolution_to_anatomical', False)
    visualizer.plot_joint_evolution_and_queries(
        true_limits=true_limits, 
        cap_to_anatomical=cap_anatomical,
        anatomical_limits=prior_gen.anatomical_limits
    )
    visualizer.plot_query_distribution()
    visualizer.plot_diagnostics()
    
    # Plot reachability for all pairs
    print("Plotting Reachability Maps (this may take a moment)...")
    print("Plotting Reachability Maps (this may take a moment)...")
    visualizer.plot_reachability(true_checker=true_checker, posterior=posterior, n_samples=20)
    
    # NEW: Plot Structure Comparison (GT vs Pred)
    print("Plotting Structure Comparison...")
    try:
        # Construct metadata
        meta_gt = legacy_to_metadata(true_checker, prior_gen.joint_names)
        meta_pred = legacy_to_metadata(posterior, prior_gen.joint_names)
        
        save_path = output_dir / "final_legacy_comparison.png"
        
        plot_comparison(
            meta_gt,
            meta_pred,
            prior_gen.joint_names,
            prior_gen.anatomical_limits,
            str(save_path),
            labels=("Ground Truth (Legacy)", "Predicted (Posterior)")
        )
        print(f"Comparison plot saved to {save_path}")
    except Exception as e:
        print(f"Failed to plot legacy comparison: {e}")
    
    print(f"Run complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
