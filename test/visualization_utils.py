import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from typing import Dict, List, Any, Optional, Tuple
from active_learning.src.diagnostics import Diagnostics, DiagnosticSnapshot
from active_learning.src.metrics import compute_reachability_metrics
from active_learning.src.config import DEVICE
from infer_params.training.level_set_torch import create_evaluation_grid

class Visualizer:
    def __init__(self, save_dir: str, joint_names: List[str], true_limits: Optional[Dict] = None):
        self.save_dir = save_dir
        self.joint_names = joint_names
        self.true_limits = true_limits
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize diagnostics if true_limits provided
        self.diagnostics: Optional[Diagnostics] = None
        if true_limits is not None:
            self.diagnostics = Diagnostics(joint_names, true_limits)

        # History storage
        self.history = {
            'iteration': [],
            'elbo': [],
            'bald_score': [],
            'grad_norm': [],
            'posterior_means': {j: {'lower': [], 'upper': []} for j in joint_names},
            'posterior_stds': {j: {'lower': [], 'upper': []} for j in joint_names},
            'queries': [],
            'outcomes': [],
            # Diagnostic history
            'prior_coverage': [],
            'posterior_coverage': [],
            'query_boundary_distance': [],
            'posterior_movement': [],
            # Parameter error tracking
            'param_mae': [],
            'param_uncertainty': [],
            # Reachability metrics
            'reachability_iou': [],
            'reachability_accuracy': []
        }
        self.test_grid = None

    def save_history(self, filename='history.json'):
        import json

        # Helper to convert numpy types to python types
        def default_converter(o):
            if isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, default=default_converter, indent=4)

    def log_iteration(
        self,
        iteration: int,
        posterior: Any,
        result: Any,
        prior: Any = None,
        true_checker: Any = None,
        test_grid: Any = None
    ):
        """
        Log the state of the active learning loop at a given iteration.

        Args:
            iteration: Current iteration number
            posterior: Current posterior distribution
            result: IterationResult from ActiveLearner
            prior: Prior distribution (for diagnostic tracking)
            true_checker: Ground truth feasibility checker (for boundary distance)
        """
        self.history['iteration'].append(iteration)
        self.history['elbo'].append(result.elbo)
        self.history['bald_score'].append(result.bald_score)
        self.history['grad_norm'].append(getattr(result, 'grad_norm', 0.0) or 0.0)
        self.history['queries'].append(result.test_point.detach().cpu().numpy())
        self.history['outcomes'].append(result.outcome)

        # Extract posterior params (for legacy UserDistribution)
        if hasattr(posterior, 'params') and 'joint_limits' in posterior.params:
            for joint in self.joint_names:
                p = posterior.params['joint_limits'][joint]

                # Extract lower params
                lower_mean = p['lower_mean'].detach().cpu().item()
                lower_std = np.exp(p['lower_log_std'].detach().cpu().item())

                # Extract upper params
                upper_mean = p['upper_mean'].detach().cpu().item()
                upper_std = np.exp(p['upper_log_std'].detach().cpu().item())

                self.history['posterior_means'][joint]['lower'].append(lower_mean)
                self.history['posterior_means'][joint]['upper'].append(upper_mean)
                self.history['posterior_stds'][joint]['lower'].append(lower_std)
                self.history['posterior_stds'][joint]['upper'].append(upper_std)

            # Compute and store MAE and uncertainty if true_limits available
            if self.true_limits is not None:
                mae, unc = 0.0, 0.0
                for joint in self.joint_names:
                    l_mean = self.history['posterior_means'][joint]['lower'][-1]
                    u_mean = self.history['posterior_means'][joint]['upper'][-1]
                    l_std = self.history['posterior_stds'][joint]['lower'][-1]
                    u_std = self.history['posterior_stds'][joint]['upper'][-1]
                    l_true, u_true = self.true_limits[joint]
                    mae += (abs(l_mean - l_true) + abs(u_mean - u_true)) / 2
                    unc += (l_std + u_std) / 2
                self.history['param_mae'].append(mae / len(self.joint_names))
                self.history['param_uncertainty'].append(unc / len(self.joint_names))

        # Log diagnostics if available
        if self.diagnostics is not None and prior is not None and true_checker is not None:
            snapshot = self.diagnostics.log_iteration(
                iteration=iteration,
                prior=prior,
                posterior=posterior,
                query=result.test_point,
                true_checker=true_checker,
                grad_norm=getattr(result, 'grad_norm', 0.0) or 0.0
            )

            # Store in history for plotting
            # Compute coverage rates (fraction of joints covered)
            prior_cov = np.mean([
                snapshot.prior_coverage[j][b]
                for j in self.joint_names
                for b in ['lower', 'upper']
            ])
            post_cov = np.mean([
                snapshot.posterior_coverage[j][b]
                for j in self.joint_names
                for b in ['lower', 'upper']
            ])

            self.history['prior_coverage'].append(prior_cov)
            self.history['posterior_coverage'].append(post_cov)
            self.history['query_boundary_distance'].append(snapshot.query_distance_to_boundary)
            self.history['posterior_movement'].append(snapshot.posterior_movement)

            # Print concise diagnostic summary
            self.diagnostics.print_iteration_summary(snapshot)

        # 4. Reachability Metrics (if true_checker and test_grid available)
        is_latent = hasattr(posterior, 'decoder') and hasattr(posterior, 'mean')
        if (true_checker is not None or is_latent) and test_grid is not None:
            # We need to distinguish between legacy and latent
            # If posterior is UserDistribution, it's legacy
            # If it has .mean and .log_std (tensors) or is LatentUserDistribution, it's latent
            from active_learning.src.metrics import compute_legacy_reachability_metrics, compute_reachability_metrics
            
            # Simple check for latent vs legacy
            
            if is_latent:
                # Latent
                post_mean = posterior.mean
                if post_mean.dim() == 1:
                    post_mean = post_mean.unsqueeze(0)
                
                iou, acc, f1, _ = compute_reachability_metrics(
                    decoder=posterior.decoder,
                    ground_truth_params=getattr(self, 'ground_truth_params', None),
                    posterior_mean=post_mean,
                    test_grid=test_grid
                )

            else:
                # Legacy
                iou, acc, _ = compute_legacy_reachability_metrics(
                    true_checker=true_checker,
                    posterior=posterior,
                    test_grid=test_grid
                )

            
            self.history['reachability_iou'].append(iou)
            self.history['reachability_accuracy'].append(acc)

    def plot_information_gain(self):
        """Plot Information Gain (BALD Score) evolution."""
        if not self.history['bald_score']:
             print("No BALD score data to plot.")
             return

        fig, ax = plt.subplots(figsize=(10, 5))
        iterations = self.history['iteration']
        scores = self.history['bald_score']

        ax.plot(iterations, scores, color='tab:blue', marker='o', linestyle='-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Information Gain (BALD Score)')
        ax.set_title('Information Gain per Iteration')
        ax.grid(True, alpha=0.3)

        # Annotate max score
        if scores:
            max_score = max(scores)
            max_iter = iterations[scores.index(max_score)]
            ax.annotate(f'Max: {max_score:.4f}',
                        xy=(max_iter, max_score),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color='blue',
                        bbox=dict(boxstyle='round', facecolor='azure', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'information_gain.png'), dpi=150)
        plt.close()
        print(f"Information gain plot saved to {os.path.join(self.save_dir, 'information_gain.png')}")

    def plot_metrics(self):
        """Plot ELBO and BALD score evolution."""
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('ELBO', color=color)
        ax1.plot(self.history['iteration'], self.history['elbo'], color=color, marker='o', label='ELBO')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('BALD Score', color=color)
        ax2.plot(self.history['iteration'], self.history['bald_score'], color=color, marker='x', linestyle='--', label='BALD')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Active Learning Metrics')
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics.png'))
        plt.close()

    def plot_param_error_over_time(self):
        """Plot parameter MAE and uncertainty over iterations."""
        if not self.history['param_mae']:
            print("No parameter error data to plot.")
            return
        fig, ax1 = plt.subplots(figsize=(10, 5))
        iterations = list(self.history['iteration'])
        maes = list(self.history['param_mae'])
        uncs = list(self.history['param_uncertainty'])

        # Filter out initial 0 jump if present (often iteration 0 or 1 is logged as 0 before first update)
        if len(maes) > 1 and maes[0] == 0.0:
            iterations = iterations[1:]
            maes = maes[1:]
            uncs = uncs[1:]

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MAE (rad)', color='tab:red')
        ax1.plot(iterations, maes, color='tab:red', marker='o', label='MAE')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Uncertainty (std)', color='tab:blue')
        ax2.plot(iterations, uncs, color='tab:blue', marker='x', linestyle='--', label='Uncertainty')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title('Parameter MAE and Uncertainty Over Time')
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'param_error.png'))
        plt.close()


    def plot_joint_evolution_and_queries(self, true_limits: Dict[str, tuple], cap_to_anatomical: bool = False, anatomical_limits: Optional[Dict[str, tuple]] = None):
        """
        Combined plot: joint limit evolution and query distribution for each joint.
        
        Args:
            true_limits: Dict mapping joint_name -> (lower, upper) for ground truth
            cap_to_anatomical: If True, cap y-axis to anatomical limits (with small margin)
            anatomical_limits: Dict mapping joint_name -> (lower, upper) for anatomical bounds.
                               Required if cap_to_anatomical is True.
        """
        n_joints = len(self.joint_names)
        fig, axes = plt.subplots(n_joints, 1, figsize=(12, 4 * n_joints), sharex=True)
        if n_joints == 1:
            axes = [axes]

        iterations = self.history['iteration']
        queries = np.array(self.history['queries']) # (n_iters, n_joints)
        outcomes = np.array(self.history['outcomes']) # (n_iters,)

        for idx, joint in enumerate(self.joint_names):
            ax = axes[idx]

            # Joint limit evolution
            l_means = np.array(self.history['posterior_means'][joint]['lower'])
            l_stds = np.array(self.history['posterior_stds'][joint]['lower'])
            u_means = np.array(self.history['posterior_means'][joint]['upper'])
            u_stds = np.array(self.history['posterior_stds'][joint]['upper'])

            # Lower Limit
            ax.plot(iterations, l_means, label='Est. Lower', color='blue')
            # 95% CI (lighter)
            ax.fill_between(iterations, l_means - 1.96 * l_stds, l_means + 1.96 * l_stds, color='blue', alpha=0.1, label='Lower 95% CI')
            # 1 std (darker)
            ax.fill_between(iterations, l_means - l_stds, l_means + l_stds, color='blue', alpha=0.2, label='Lower 1 std')

            # Upper Limit
            ax.plot(iterations, u_means, label='Est. Upper', color='green')
            # 95% CI (lighter)
            ax.fill_between(iterations, u_means - 1.96 * u_stds, u_means + 1.96 * u_stds, color='green', alpha=0.1, label='Upper 95% CI')
            # 1 std (darker)
            ax.fill_between(iterations, u_means - u_stds, u_means + u_stds, color='green', alpha=0.2, label='Upper 1 std')

            true_l, true_u = true_limits[joint]
            ax.axhline(y=true_l, color='blue', linestyle='--', label='True Lower')
            ax.axhline(y=true_u, color='green', linestyle='--', label='True Upper')

            # Query distribution overlay
            joint_queries = queries[:, idx]
            
            feasible_mask = outcomes > 0.5
            infeasible_mask = outcomes <= 0.5

            # Plot Feasible (green circles)
            if np.any(feasible_mask):
                ax.scatter(
                    np.array(self.history['iteration'])[feasible_mask],
                    joint_queries[feasible_mask],
                    color='green', marker='o', label='Feasible',
                    alpha=0.9, edgecolor='k', linewidth=0.5
                )

            # Plot Infeasible (red crosses)
            if np.any(infeasible_mask):
                ax.scatter(
                    np.array(self.history['iteration'])[infeasible_mask],
                    joint_queries[infeasible_mask],
                    color='red', marker='x', label='Infeasible', alpha=0.9
                )
            
            ax.set_ylabel(f'{joint} (rad)')
            ax.set_title(f'{joint}: Limit Evolution & Query Distribution')
            
            # Cap y-axis to anatomical limits if requested
            if cap_to_anatomical and anatomical_limits is not None and joint in anatomical_limits:
                anat_lower, anat_upper = anatomical_limits[joint]
                margin = 0.05 * (anat_upper - anat_lower)  # 5% margin for aesthetics
                ax.set_ylim(anat_lower - margin, anat_upper + margin)

        axes[-1].set_xlabel('Iteration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'joint_evolution.png'))
        plt.close()

    def plot_query_distribution(self):
        """
        Plot the distribution of queries for each joint.
        """
        queries = np.array(self.history['queries']) # (n_iters, n_joints)
        outcomes = np.array(self.history['outcomes']) # (n_iters,)

        n_joints = len(self.joint_names)
        fig, axes = plt.subplots(n_joints, 1, figsize=(10, 3 * n_joints))
        if n_joints == 1:
            axes = [axes]

        for idx, joint in enumerate(self.joint_names):
            ax = axes[idx]
            joint_queries = queries[:, idx]

            # Scatter plot of queries over iterations
            # Scatter plot of queries over iterations
            norm = plt.Normalize(-1.5, 1.5)
            cmap = plt.cm.RdYlGn
            
            feasible_mask = outcomes >= 0
            infeasible_mask = outcomes < 0

            if np.any(feasible_mask):
                ax.scatter(
                    np.array(self.history['iteration'])[feasible_mask], 
                    joint_queries[feasible_mask],
                    c=outcomes[feasible_mask], 
                    cmap=cmap, norm=norm,
                    marker='o', label='Feasible'
                )
            if np.any(infeasible_mask):
                ax.scatter(
                    np.array(self.history['iteration'])[infeasible_mask], 
                    joint_queries[infeasible_mask],
                    c=outcomes[infeasible_mask], 
                    cmap=cmap, norm=norm,
                    marker='x', label='Infeasible'
                )

            ax.set_ylabel(f'{joint} (rad)')
            ax.set_title(f'Queries for {joint}')
            
            # Add colorbar only to the last plot to save space
            if idx == n_joints - 1:
               sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
               sm.set_array([])
               fig.colorbar(sm, ax=axes.ravel().tolist(), label='Signed Distance h (Red < 0 < Green)', aspect=30)
            
            ax.legend()

        axes[-1].set_xlabel('Iteration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'query_distribution.png'))
        plt.close()

    def plot_reachability(self, true_checker: Any, posterior: Any, n_samples: int = 10, resolution: int = 200):
        """
        Plot the reachability boundaries for pairs of joints.
        Compares Ground Truth vs Posterior Probability.
        """
        # Lazy import to avoid loading legacy module when not needed
        from active_learning.src.legacy.feasibility_checker import FeasibilityChecker
        
        pairs = posterior.pairs
        if not pairs:
            print("No pairs to visualize.")
            return

        # Sample from posterior
        posterior_samples = posterior.sample(n_samples, return_list=True)
        posterior_checkers = []
        for theta in posterior_samples:
            # Convert theta to cpu/numpy for FeasibilityChecker (which expects numpy/cpu if q is numpy)
            cpu_theta = {'joint_limits': {}, 'pairwise_constraints': {}}

            for j, (l, u) in theta['joint_limits'].items():
                l_val = l.detach().cpu().item() if isinstance(l, torch.Tensor) else l
                u_val = u.detach().cpu().item() if isinstance(u, torch.Tensor) else u
                cpu_theta['joint_limits'][j] = (l_val, u_val)

            for pair, constraint in theta['pairwise_constraints'].items():
                cpu_constraint = {'bumps': []}
                if 'box' in constraint:
                    cpu_constraint['box'] = constraint['box'] # Assuming box is already compatible or not present

                for bump in constraint['bumps']:
                    cpu_bump = {}
                    for k, v in bump.items():
                        if isinstance(v, torch.Tensor):
                            cpu_bump[k] = v.detach().cpu().numpy()
                        else:
                            cpu_bump[k] = v
                    cpu_constraint['bumps'].append(cpu_bump)

                cpu_theta['pairwise_constraints'][pair] = cpu_constraint

            checker = FeasibilityChecker(
                joint_limits=cpu_theta['joint_limits'],
                pairwise_constraints=cpu_theta['pairwise_constraints'],
                config=posterior.config if hasattr(posterior, 'config') else None
            )
            posterior_checkers.append(checker)

        for (j1, j2) in pairs:
            idx1 = self.joint_names.index(j1)
            idx2 = self.joint_names.index(j2)

            # Define grid based on anatomical limits or current view
            # Use anatomical limits if available, else use true limits + margin
            if posterior.anatomical_limits:
                lim1 = posterior.anatomical_limits[j1]
                lim2 = posterior.anatomical_limits[j2]
            else:
                # Fallback
                lim1 = (-np.pi, np.pi)
                lim2 = (-np.pi, np.pi)

            x = np.linspace(lim1[0], lim1[1], resolution)
            y = np.linspace(lim2[0], lim2[1], resolution)
            X, Y = np.meshgrid(x, y)

            # Create query points (N, n_joints)
            # We set other joints to 0 (or mid-range)
            grid_points = np.zeros((X.size, len(self.joint_names)))

            # Set defaults for other joints (e.g. middle of anatomical limits)
            if posterior.anatomical_limits:
                for i, name in enumerate(self.joint_names):
                    l, u = posterior.anatomical_limits[name]
                    grid_points[:, i] = (l + u) / 2.0

            grid_points[:, idx1] = X.ravel()
            grid_points[:, idx2] = Y.ravel()

            # 1. Ground Truth
            # true_checker.h_value returns min_h. If >= 0, feasible.
            h_true = true_checker.h_value(grid_points)
            Z_true = (h_true >= 0).astype(float).reshape(X.shape)

            # 2. Posterior Probability
            Z_post_sum = np.zeros_like(X.ravel())
            for checker in posterior_checkers:
                h_val = checker.h_value(grid_points)
                # Convert to numpy if tensor
                if isinstance(h_val, torch.Tensor):
                    h_val = h_val.detach().cpu().numpy()
                Z_post_sum += (h_val >= 0).astype(float)

            Z_post = (Z_post_sum / n_samples).reshape(X.shape)

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Convert to factors of pi for plotting
            X_plot = X
            Y_plot = Y

            # Ground Truth
            im1 = axes[0].contourf(X_plot, Y_plot, Z_true, levels=[-0.1, 0.5, 1.1], colors=['red', 'green'], alpha=0.5)
            axes[0].set_title(f'Ground Truth: {j1} vs {j2}')
            axes[0].set_xlabel(f'{j1} (rad)')
            axes[0].set_ylabel(f'{j2} (rad)')

            # Posterior
            # Debug info
            print(f"Plotting {j1} vs {j2}: Z_post range [{Z_post.min()}, {Z_post.max()}], NaNs: {np.isnan(Z_post).any()}")

            im2 = axes[1].contourf(X_plot, Y_plot, Z_post, levels=np.linspace(0, 1, 11), cmap='RdYlGn', extend='both')
            plt.colorbar(im2, ax=axes[1], label='Probability of Feasibility')
            axes[1].set_title(f'Posterior Probability: {j1} vs {j2}')
            axes[1].set_xlabel(f'{j1} (rad)')
            axes[1].set_ylabel(f'{j2} (rad)')

            # Overlay queries on Posterior plot
            queries = np.array(self.history['queries'])
            outcomes = np.array(self.history['outcomes'])
            if len(queries) > 0:
                q_x = queries[:, idx1]
                q_y = queries[:, idx2]

                feasible_mask = outcomes >= 0
                infeasible_mask = outcomes < 0

                if np.any(feasible_mask):
                    axes[1].scatter(q_x[feasible_mask], q_y[feasible_mask], c='white', edgecolors='green', marker='o', s=20, label='Feasible')
                if np.any(infeasible_mask):
                    axes[1].scatter(q_x[infeasible_mask], q_y[infeasible_mask], c='black', marker='x', s=20, label='Infeasible')
                axes[1].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'reachability_{j1}_{j2}.png'))
            plt.close()

    def plot_diagnostics(self):
        """
        Plot comprehensive diagnostics dashboard.

        Creates a 2x2 grid showing:
        1. Coverage over iterations
        2. Query distance to boundary
        3. Gradient norm evolution
        4. Posterior movement
        """
        if not self.history['posterior_coverage']:
            print("No diagnostic data to plot. Ensure prior and true_checker are passed to log_iteration.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        iterations = self.history['iteration']

        # 1. Coverage over time
        ax1 = axes[0, 0]
        ax1.plot(iterations, self.history['posterior_coverage'], 'b-', linewidth=2, label='Posterior Coverage')
        if self.history['prior_coverage']:
            ax1.axhline(y=self.history['prior_coverage'][0], color='gray', linestyle='--', label='Initial Prior Coverage')
        ax1.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Perfect Coverage')
        ax1.axhline(y=0.5, color='red', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Coverage Rate')
        ax1.set_title('1. Prior/Posterior Coverage\n(Fraction of true values within 2σ)')
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Add annotation for final coverage
        if self.history['posterior_coverage']:
            final_cov = self.history['posterior_coverage'][-1]
            ax1.annotate(f'Final: {final_cov:.1%}',
                        xy=(iterations[-1], final_cov),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color='blue')

        # 2. Query distance to boundary
        ax2 = axes[0, 1]
        distances = self.history['query_boundary_distance']
        colors = ['green' if d < 0.1 else 'orange' if d < 0.3 else 'red' for d in distances]
        ax2.scatter(iterations, distances, c=colors, s=50, alpha=0.7)
        ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Near boundary (<0.1)')
        ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium (0.1-0.3)')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('|h(query)| (Distance to Boundary)')
        ax2.set_title('2. Query Informativeness\n(Lower = closer to boundary = more informative)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Add trend line
        if len(distances) > 1:
            z = np.polyfit(iterations, distances, 1)
            p = np.poly1d(z)
            ax2.plot(iterations, p(iterations), 'r--', alpha=0.5, label='Trend')

        # 3. Gradient norm evolution
        ax3 = axes[1, 0]
        grad_norms = self.history['grad_norm']
        ax3.semilogy(iterations, grad_norms, 'purple', linewidth=2, marker='o', markersize=4)
        ax3.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('‖∇ELBO‖ (log scale)')
        ax3.set_title('3. Gradient Health\n(Vanishing gradients indicate flat likelihood)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # Add warning annotation if gradients are small
        mean_grad = np.mean(grad_norms) if grad_norms else 0
        if mean_grad < 0.01:
            ax3.annotate('⚠ Low gradients!',
                        xy=(iterations[-1]//2, mean_grad),
                        fontsize=12, color='red',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 4. Posterior movement
        ax4 = axes[1, 1]
        movements = self.history['posterior_movement']
        cumulative_movement = np.cumsum(movements)

        ax4.bar(iterations, movements, alpha=0.6, label='Per-iteration movement')
        ax4.plot(iterations, cumulative_movement, 'r-', linewidth=2, label='Cumulative movement')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Posterior Movement (rad)')
        ax4.set_title('4. Posterior Movement\n(How much posterior is updating)')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # Add annotation for total movement
        if cumulative_movement.size > 0:
            total = cumulative_movement[-1]
            ax4.annotate(f'Total: {total:.3f} rad',
                        xy=(iterations[-1], total),
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=10, color='red')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'diagnostics.png'), dpi=150)
        plt.close()

        print(f"Diagnostics plot saved to {os.path.join(self.save_dir, 'diagnostics.png')}")

    def print_diagnostic_report(self):
        """Print the final diagnostic report."""
        # if self.diagnostics is not None:
        #     self.diagnostics.print_final_report()
        # else:
        #     print("No diagnostics available. Initialize Visualizer with true_limits to enable diagnostics.")

    def plot_gradient_vs_tau_analysis(self, tau_values: List[float] = None):
        return
        """
        Helper to visualize how different tau values affect gradients.
        This is a static analysis plot, not dependent on run history.
        """
        if tau_values is None:
            tau_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

        h_values = np.linspace(-1, 1, 200)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Sigmoid curves for different tau
        ax1 = axes[0]
        for tau in tau_values:
            probs = 1 / (1 + np.exp(-h_values / tau))
            ax1.plot(h_values, probs, label=f'τ={tau}')

        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('h value (distance to boundary)')
        ax1.set_ylabel('P(feasible)')
        ax1.set_title('Likelihood Sharpness vs τ\n(Smaller τ = sharper transition)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gradient magnitude for different tau
        ax2 = axes[1]
        for tau in tau_values:
            probs = 1 / (1 + np.exp(-h_values / tau))
            # Gradient of log-likelihood: (y - p) * (1/tau) for BCE
            # For visualization, show |dp/dh| = p*(1-p)/tau
            grad_magnitude = probs * (1 - probs) / tau
            ax2.plot(h_values, grad_magnitude, label=f'τ={tau}')

        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('h value (distance to boundary)')
        ax2.set_ylabel('Gradient magnitude |dp/dh|')
        ax2.set_title('Gradient Magnitude vs τ\n(Small τ concentrates gradients near h=0)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'tau_analysis.png'), dpi=150)
        plt.close()

        print(f"Tau analysis plot saved to {os.path.join(self.save_dir, 'tau_analysis.png')}")

    def plot_reachability_evolution(self):
        """Plot Reachability IoU and Accuracy over iterations."""
        if not self.history.get('reachability_iou'):
             print("No reachability data to plot.")
             return

        fig, ax1 = plt.subplots(figsize=(10, 5))
        iterations = self.history['iteration']
        ious = self.history['reachability_iou']
        accs = self.history['reachability_accuracy']
        
        # Ensure lengths match (sometimes iteration 0 is logged)
        if len(iterations) > len(ious):
             plot_iters = iterations[-len(ious):]
        else:
             plot_iters = iterations

        color = 'tab:blue'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('IoU', color=color)
        ax1.plot(plot_iters, ious, color=color, marker='o', label='IoU')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.0)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(plot_iters, accs, color=color, marker='x', linestyle='--', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1.0)

        plt.title('Reachability Accuracy (IoU) Over Time')
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'reachability_metrics.png'))
        plt.close()
        print(f"Reachability plot saved to {os.path.join(self.save_dir, 'reachability_metrics.png')}")

    def plot_consolidated_metrics(self):
        """
        Plot consolidated metrics: MAE, Uncertainty, IoU, and Accuracy.
        All on a single figure for easy comparison.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = self.history['iteration']
        
        # --- Parameter MAE ---
        if self.history.get('param_mae'):
            maes = self.history['param_mae']
            plot_iters = iterations[-len(maes):] if len(iterations) > len(maes) else iterations
            ax.plot(plot_iters, maes, 'b-', marker='o', markersize=3, linewidth=1.5, label=f'MAE (final: {maes[-1]:.3f} rad)')
        
        # --- Uncertainty ---
        # Try latent uncertainty first, then legacy
        uncs = None
        unc_label = ""
        if 'latent_stds' in self.history and self.history['latent_stds']:
            uncs = [np.mean(s) for s in self.history['latent_stds']]
            unc_label = "Latent Uncertainty"
        elif self.history.get('param_uncertainty'):
            uncs = self.history['param_uncertainty']
            unc_label = "Param Uncertainty"
            
        if uncs:
            plot_iters = iterations[-len(uncs):] if len(iterations) > len(uncs) else iterations
            ax.plot(plot_iters, uncs, 'purple', marker='s', markersize=3, linewidth=1.5, label=f'{unc_label} (final: {uncs[-1]:.3f})')
        
        # --- Reachability IoU ---
        if self.history.get('reachability_iou'):
            ious = self.history['reachability_iou']
            plot_iters = iterations[-len(ious):] if len(iterations) > len(ious) else iterations
            ax.plot(plot_iters, ious, 'g-', marker='^', markersize=3, linewidth=1.5, label=f'IoU (final: {ious[-1]:.3f})')
        
        # --- Reachability Accuracy ---
        if self.history.get('reachability_accuracy'):
            accs = self.history['reachability_accuracy']
            plot_iters = iterations[-len(accs):] if len(iterations) > len(accs) else iterations
            ax.plot(plot_iters, accs, 'orange', marker='v', markersize=3, linewidth=1.5, label=f'Accuracy (final: {accs[-1]:.3f})')
        
        ax.set_xlabel('Query')
        ax.set_ylabel('Metric Value')
        ax.set_title('Consolidated Metrics Over Queries')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'consolidated_metrics.png'), dpi=150)
        plt.close()
        print(f"Consolidated metrics plot saved to {os.path.join(self.save_dir, 'consolidated_metrics.png')}")

class LatentVisualizer(Visualizer):
    """
    Visualizer for Latent Active Learning.

    Handles decoding of latent state to joint limits for direct comparison
    with the explicit joint-space version.
    """


    def __init__(self, save_dir: str, joint_names: List[str], decoder: Any, true_limits: Optional[Dict] = None, ground_truth_params: Optional[Dict] = None, resolution: int = 12, anatomical_limits: Optional[Dict[str, tuple]] = None, true_checker: Any = None):
        super().__init__(save_dir, joint_names, true_limits)
        self.decoder = decoder
        self.ground_truth_params = ground_truth_params
        self.resolution = resolution
        self.anatomical_limits = anatomical_limits  # Store anatomical limits for y-axis capping
        self.true_checker = true_checker
        
        # Additional history
        self.history['latent_means'] = []
        self.history['latent_stds'] = []
        self.history['latent_error'] = []
        self.history['reachability_iou'] = []
        self.history['reachability_accuracy'] = []
        self.history['ground_truth_z'] = None  # Store ground truth latent code
        
        # Create evaluation grid if ground truth params are available
        self.evaluation_grid = None
        if self.ground_truth_params is not None:
             def to_tensor(x, device):
                 if isinstance(x, torch.Tensor):
                     return x.detach().clone().to(device)
                 return torch.tensor(x, device=device)

             gt_lower = to_tensor(self.ground_truth_params['box_lower'], DEVICE)
             gt_upper = to_tensor(self.ground_truth_params['box_upper'], DEVICE)
             self.evaluation_grid = create_evaluation_grid(gt_lower, gt_upper, resolution, DEVICE)

    def log_initial_state(self, prior: Any, ground_truth_z: Any = None):
        """
        Log the initial prior/posterior state BEFORE any active learning steps.
        This records iteration 0 to show the true starting point.
        
        Args:
            prior: The initial prior distribution (same as initial posterior)
            ground_truth_z: Ground truth latent code for error tracking
        """
        # Use iteration 0 to indicate initial prior state (before any steps)
        self.history['iteration'].append(0)
        self.history['elbo'].append(0.0)  # No ELBO yet
        self.history['bald_score'].append(0.0)  # No BALD score yet
        self.history['grad_norm'].append(0.0)
        self.history['queries'].append(np.zeros(len(self.joint_names)))  # No query yet
        self.history['outcomes'].append(0.0)  # No outcome yet
        
        # 1. Latent Statistics
        z_mean = prior.mean.detach().cpu().numpy()
        z_std = torch.exp(prior.log_std).detach().cpu().numpy()
        self.history['latent_means'].append(z_mean)
        self.history['latent_stds'].append(z_std)
        
        if ground_truth_z is not None:
            gt = ground_truth_z.detach().cpu().numpy() if isinstance(ground_truth_z, torch.Tensor) else ground_truth_z
            err = np.linalg.norm(z_mean - gt)
            self.history['latent_error'].append(float(err))
            self.history['ground_truth_z'] = gt
        
        # 2. Decoded Joint Limits (deterministic - just use mean)
        with torch.no_grad():
            z_mean_tensor = prior.mean.unsqueeze(0)  # (1, latent_dim)
            z_std_tensor = torch.exp(prior.log_std)
            
            # Decode mean
            if hasattr(self.decoder, 'decode_from_embedding'):
                lowers, uppers, _, _, _ = self.decoder.decode_from_embedding(z_mean_tensor)
            elif hasattr(self.decoder, 'transform_head'):
                centers, halfwidths = self.decoder.transform_head(z_mean_tensor)
                lowers = centers - halfwidths
                uppers = centers + halfwidths
            else:
                raise ValueError("Decoder must have either decode_from_embedding or transform_head method")
            
            l_means = lowers.squeeze().cpu().numpy()
            u_means = uppers.squeeze().cpu().numpy()
            
            # Estimate stds by sampling (use fixed seed for reproducibility in viz)
            n_samples = 50
            z_dist = torch.distributions.Normal(prior.mean, z_std_tensor)
            # Use deterministic sampling by setting generator
            z_samples = z_dist.sample((n_samples,))
            if hasattr(self.decoder, 'decode_from_embedding'):
                lowers_samples, uppers_samples, _, _, _ = self.decoder.decode_from_embedding(z_samples)
            else:
                centers, halfwidths = self.decoder.transform_head(z_samples)
                lowers_samples = centers - halfwidths
                uppers_samples = centers + halfwidths
            l_stds = lowers_samples.std(dim=0).cpu().numpy()
            u_stds = uppers_samples.std(dim=0).cpu().numpy()
        
        # Store
        for idx in range(min(len(self.joint_names), len(l_means))):
            joint = self.joint_names[idx]
            self.history['posterior_means'][joint]['lower'].append(float(l_means[idx]))
            self.history['posterior_means'][joint]['upper'].append(float(u_means[idx]))
            self.history['posterior_stds'][joint]['lower'].append(float(l_stds[idx]))
            self.history['posterior_stds'][joint]['upper'].append(float(u_stds[idx]))
        
        # Compute MAE vs True Limits
        if self.true_limits is not None:
            mae, unc = 0.0, 0.0
            for joint in self.joint_names:
                l_mean = self.history['posterior_means'][joint]['lower'][-1]
                u_mean = self.history['posterior_means'][joint]['upper'][-1]
                l_std = self.history['posterior_stds'][joint]['lower'][-1]
                u_std = self.history['posterior_stds'][joint]['upper'][-1]
                l_true, u_true = self.true_limits[joint]
                mae += (abs(l_mean - l_true) + abs(u_mean - u_true)) / 2
                unc += (l_std + u_std) / 2
            self.history['param_mae'].append(mae / len(self.joint_names))
            self.history['param_uncertainty'].append(unc / len(self.joint_names))
        
        # Log initial reachability
        if self.evaluation_grid is not None:
             from active_learning.src.metrics import compute_reachability_metrics
             post_mean = prior.mean
             if post_mean.dim() == 1:
                 post_mean = post_mean.unsqueeze(0)
             
             iou, acc, f1, _ = compute_reachability_metrics(
                 decoder=self.decoder,
                 ground_truth_params=self.ground_truth_params,
                 posterior_mean=post_mean,
                 test_grid=self.evaluation_grid
             )
             self.history['reachability_iou'].append(iou)
             self.history['reachability_accuracy'].append(acc)
        
        print(f"  Logged initial prior state (iteration 0)")

    def log_iteration(
        self,
        iteration: int,
        posterior: Any,
        result: Any,
        ground_truth_z: Any = None,
        **kwargs
    ):
        """
        Log Latent AL iteration.
        """
        # 1. Latent Statistics
        z_mean = posterior.mean.detach().cpu().numpy()
        z_std = torch.exp(posterior.log_std).detach().cpu().numpy()
        self.history['latent_means'].append(z_mean)
        self.history['latent_stds'].append(z_std)

        if ground_truth_z is not None:
             gt = ground_truth_z.detach().cpu().numpy() if isinstance(ground_truth_z, torch.Tensor) else ground_truth_z
             err = np.linalg.norm(z_mean - gt)
             self.history['latent_error'].append(float(err))
             # Store ground truth z on first iteration
             if self.history['ground_truth_z'] is None:
                 self.history['ground_truth_z'] = gt

        # 2. Decoded Joint Limits (via Monte Carlo)
        n_samples = 50
        with torch.no_grad():
            z_dist = torch.distributions.Normal(posterior.mean, torch.exp(posterior.log_std))
            z_samples = z_dist.sample((n_samples,)) # (N, latent_dim)

            # Decode: z -> (lower, upper, weights, presence_logits, blob_params)
            # Support both old (SIRSAutoDecoder) and new (LevelSetDecoder) model interfaces
            if hasattr(self.decoder, 'decode_from_embedding'):
                # New LevelSetDecoder: directly outputs lower/upper
                lowers, uppers, _, _, _ = self.decoder.decode_from_embedding(z_samples)
            elif hasattr(self.decoder, 'transform_head'):
                # Old SIRSAutoDecoder: outputs center/halfwidths
                centers, halfwidths = self.decoder.transform_head(z_samples)
                lowers = centers - halfwidths
                uppers = centers + halfwidths
            else:
                raise ValueError("Decoder must have either decode_from_embedding or transform_head method")

            # Compute stats
            l_means = lowers.mean(dim=0).cpu().numpy()
            l_stds = lowers.std(dim=0).cpu().numpy()
            u_means = uppers.mean(dim=0).cpu().numpy()
            u_stds = uppers.std(dim=0).cpu().numpy()

        # Store
        for idx in range(min(len(self.joint_names), len(l_means))):
             joint = self.joint_names[idx]
             self.history['posterior_means'][joint]['lower'].append(float(l_means[idx]))
             self.history['posterior_means'][joint]['upper'].append(float(u_means[idx]))
             self.history['posterior_stds'][joint]['lower'].append(float(l_stds[idx]))
             self.history['posterior_stds'][joint]['upper'].append(float(u_stds[idx]))

        # Compute MAE vs True Limits
        if self.true_limits is not None:
             mae, unc = 0.0, 0.0
             for joint in self.joint_names:
                l_mean = self.history['posterior_means'][joint]['lower'][-1]
                u_mean = self.history['posterior_means'][joint]['upper'][-1]
                l_std = self.history['posterior_stds'][joint]['lower'][-1]
                u_std = self.history['posterior_stds'][joint]['upper'][-1]
                l_true, u_true = self.true_limits[joint]
                mae += (abs(l_mean - l_true) + abs(u_mean - u_true)) / 2
                unc += (l_std + u_std) / 2
             self.history['param_mae'].append(mae / len(self.joint_names))
             self.history['param_uncertainty'].append(unc / len(self.joint_names))

        # 3. Use base class for reachability and diagnostics if needed
        # We pass evaluation_grid as test_grid
        super().log_iteration(
            iteration=iteration,
            posterior=posterior,
            result=result,
            true_checker=kwargs.get('true_checker', self.true_checker),
            test_grid=kwargs.get('test_grid', self.evaluation_grid)
        )
        
    def plot_latent_evolution(self):
        """Plot evolution of all latent dimensions with ground truth."""
        means = np.array(self.history['latent_means']) # (iters, dim)
        stds = np.array(self.history['latent_stds'])   # (iters, dim)
        dim = means.shape[1]

        # Plot all dimensions
        fig, axes = plt.subplots(dim, 1, figsize=(12, 2.5*dim), sharex=True)
        if dim == 1: axes = [axes]

        iters = self.history['iteration']
        ground_truth_z = self.history['ground_truth_z']

        for i in range(dim):
            ax = axes[i]
            mu = means[:, i]
            sigma = stds[:, i]

            # Plot estimated mean with uncertainty bands
            ax.plot(iters, mu, label=f'Est. z_{i}', color='purple', linewidth=1.5)
            # 95% CI (lighter)
            ax.fill_between(iters, mu - 1.96 * sigma, mu + 1.96 * sigma, color='purple', alpha=0.1)
            # 1 std (darker)
            ax.fill_between(iters, mu - sigma, mu + sigma, color='purple', alpha=0.2)

            # Plot ground truth if available
            if ground_truth_z is not None and i < len(ground_truth_z):
                ax.axhline(y=ground_truth_z[i], color='green', linestyle='--', linewidth=1.5, label=f'True z_{i}')

            ax.set_ylabel(f'z_{i}')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Iteration')
        plt.suptitle(f'Latent Parameter Evolution (All {dim} Dimensions)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'latent_evolution.png'), dpi=100)
        plt.close()

    def plot_latent_error(self):
        """Plot latent error (L2 distance to ground truth) over iterations."""
        if not self.history['latent_error']:
            print("No latent error data to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        iterations = self.history['iteration']
        errors = self.history['latent_error']

        ax.plot(iterations, errors, 'b-', marker='o', markersize=4, linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Latent Error (L2 norm)')
        ax.set_title('Latent Code Error vs Ground Truth')
        ax.grid(True, alpha=0.3)

        # Add final error annotation
        if errors:
            final_err = errors[-1]
            ax.annotate(f'Final: {final_err:.4f}',
                       xy=(iterations[-1], final_err),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, color='blue',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'latent_error.png'), dpi=150)
        plt.close()
        print(f"Latent error plot saved to {os.path.join(self.save_dir, 'latent_error.png')}")


# ==============================================================================
# Shared Visualization Functions (Latent & Legacy Comparison)
# ==============================================================================

from matplotlib.patches import Rectangle, Ellipse
from infer_params.training.level_set_torch import evaluate_level_set_batched

def decode_to_metadata(z: torch.Tensor, decoder) -> Dict[str, Any]:
    """Decode latent vector z to organized metadata (boxes, blobs)."""
    with torch.no_grad():
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        # Decode
        if hasattr(decoder, 'decode_from_embedding'):
             lower, upper, weights, pres_logits, blob_params = decoder.decode_from_embedding(z)
        else:
             # Fallback/Dummy for models without decode_from_embedding (e.g. legacy w/o latent)
             # keeping minimal structure
             return {}
             
        pres = torch.sigmoid(pres_logits)
        
        return {
            'lower': lower.squeeze(0).cpu(),
            'upper': upper.squeeze(0).cpu(),
            'weights': weights.squeeze(0).cpu(),
            'presence': pres.squeeze(0).cpu(),
            'blob_params': blob_params.squeeze(0).cpu() 
        }

def legacy_to_metadata(obj: Any, joint_names: List[str], num_slots: int = 18) -> Dict[str, Any]:
    """
    Convert legacy object (FeasibilityChecker or UserDistribution) to metadata dict.
    Adapts legacy structure to the latent decoder's fixed slot format for visualization.
    """
    device = DEVICE
    
    # 1. Extract Box Limits
    lower = torch.zeros(len(joint_names), device=device)
    upper = torch.zeros(len(joint_names), device=device)
    
    if hasattr(obj, 'joint_limits'):
        # FeasibilityChecker (Ground Truth)
        for i, name in enumerate(joint_names):
            l, u = obj.joint_limits[name]
            lower[i] = float(l)
            upper[i] = float(u)
    elif hasattr(obj, 'params') and 'joint_limits' in obj.params:
        # UserDistribution (Posterior) - use means
        for i, name in enumerate(joint_names):
            p = obj.params['joint_limits'][name]
            lower[i] = float(p['lower_mean'].detach().cpu())
            upper[i] = float(p['upper_mean'].detach().cpu())
    
    # 2. Extract Blobs (only for FeasibilityChecker)
    # Map slots: 3 slots per pair. Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    blob_params = torch.zeros((num_slots, 6), device=device) # [cx, cy, sx, sy, amp, rot]
    presence = torch.zeros(num_slots, device=device)
    
    # Legacy bumps are in obj.pairwise_constraints
    if hasattr(obj, 'pairwise_constraints'):
        
        slots_per_pair = 3
        
        for pair_key, constraint in obj.pairwise_constraints.items():
            
            # Standard pair indices
            std_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            
            p_idx = -1
            if pair_key in std_pairs:
                p_idx = std_pairs.index(pair_key)
            elif (pair_key[1], pair_key[0]) in std_pairs:
                p_idx = std_pairs.index((pair_key[1], pair_key[0]))
            
            if p_idx >= 0:
                bumps = constraint.get('bumps', [])
                for b_i, bump in enumerate(bumps):
                    if b_i >= slots_per_pair: break # Limit to available slots
                    
                    slot_abs = p_idx * slots_per_pair + b_i
                    
                    # Store params
                    # Legacy: mu=[x,y], ls=[sx,sy], alpha, theta
                    blob_params[slot_abs, 0] = bump['mu'][0]
                    blob_params[slot_abs, 1] = bump['mu'][1]
                    blob_params[slot_abs, 2] = bump['ls'][0] # sigma (std)
                    blob_params[slot_abs, 3] = bump['ls'][1]
                    blob_params[slot_abs, 4] = bump['alpha']
                    blob_params[slot_abs, 5] = bump['theta']
                    
                    presence[slot_abs] = 1.0

    return {
        'lower': lower.cpu(),
        'upper': upper.cpu(),
        'weights': torch.ones_like(lower).cpu(), # Default weights 1
        'presence': presence.cpu(),
        'blob_params': blob_params.cpu()
    }

def plot_2d_projection(
    ax, 
    metadata: Dict, 
    pair_indices: Tuple[int, int], 
    pair_names: Tuple[str, str],
    limits: Dict[str, Tuple[float, float]],
    joint_names: List[str],
    resolution: int = 50
):
    """
    Plot 2D projection of feasibility boundary, box, and blobs for dimensions (i, j).
    """
    i, j = pair_indices
    name_i, name_j = pair_names
    
    # 1. Setup Grid
    lim_i = limits.get(name_i, (-3.14, 3.14))
    lim_j = limits.get(name_j, (-3.14, 3.14))
    
    x = np.linspace(lim_i[0], lim_i[1], resolution)
    y = np.linspace(lim_j[0], lim_j[1], resolution)
    XX, YY = np.meshgrid(x, y)
    
    # Create 4D points for evaluation
    # Center on the box center (decoded)
    box_center = (metadata['lower'] + metadata['upper']) / 2
    
    points_2d = np.stack([XX.flatten(), YY.flatten()], axis=1) # (N, 2)
    N = len(points_2d)
    
    points_4d = torch.zeros((N, len(joint_names)), device=DEVICE)
    # Fill with center values
    for k in range(len(joint_names)):
        points_4d[:, k] = box_center[k]
        
    # Overwrite plotting dimensions
    points_4d[:, i] = torch.tensor(points_2d[:, 0], device=DEVICE)
    points_4d[:, j] = torch.tensor(points_2d[:, 1], device=DEVICE)
    
    # 2. Evaluate Level Set
    def to_b(t): return t.unsqueeze(0).to(DEVICE)
    
    # Handle cases where metadata might be empty or partial?
    # Assume standard structure
    with torch.no_grad():
        f_vals = evaluate_level_set_batched(
            points=points_4d,
            lower=to_b(metadata['lower']),
            upper=to_b(metadata['upper']),
            weights=to_b(metadata['weights']),
            presence=to_b(metadata['presence']),
            blob_params=to_b(metadata['blob_params'])
        ).cpu().numpy().reshape(resolution, resolution)
        
    # 3. Plot Heatmap & Boundary
    # Use global vmin/vmax for consistent heatmap colors
    # f(x) > 0 is feasible
    contour = ax.contourf(XX, YY, f_vals, levels=20, cmap='RdBu', vmin=-3, vmax=3, alpha=0.8)
    ax.contour(XX, YY, f_vals, levels=[0], colors='green', linewidths=2.5) 
    
    # 4. Plot Box Bounds (dashed)
    l = metadata['lower'].numpy()
    u = metadata['upper'].numpy()
    
    rect = Rectangle(
        (l[i], l[j]), 
        u[i] - l[i], 
        u[j] - l[j],
        linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)
    
    # 5. Plot Blobs (Orange Ellipses)
    # Map (i, j) to pair index for fixed 18-slot model
    # Standard pair order: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Assuming 4 joints. If joint count differs, this mapping needs generalized logic.
    pair_map = {(0,1): 0, (0,2): 1, (0,3): 2, (1,2): 3, (1,3): 4, (2,3): 5}
    if (i, j) in pair_map:
        idx = pair_map[(i, j)]
        slots_per_pair = 3 
        start_slot = idx * slots_per_pair
        
        blobs = metadata['blob_params'][start_slot : start_slot+slots_per_pair]
        pres = metadata['presence'][start_slot : start_slot+slots_per_pair]
        
        for k in range(slots_per_pair):
            if pres[k] > 0.5:
                # Blob params: center(2), sigma(2), amp(1), rot(1)
                b_center = blobs[k, 0:2].numpy()
                b_sigma = blobs[k, 2:4].abs().numpy() 
                
                width = 4 * np.sqrt(b_sigma[0])
                height = 4 * np.sqrt(b_sigma[1])
                b_rot_rad = blobs[k, 5].item()
                b_rot_deg = np.degrees(b_rot_rad)
                
                ell = Ellipse(
                    xy=b_center,
                    width=width,
                    height=height,
                    angle=b_rot_deg,
                    edgecolor='orange', facecolor='none', linewidth=2
                )
                ax.add_patch(ell)
    
    ax.set_xlim(lim_i)
    ax.set_ylim(lim_j)
    ax.set_xlabel(name_i, fontsize=8)
    ax.set_ylabel(name_j, fontsize=8)

def plot_latent_comparison(
    z_gt: torch.Tensor, 
    z_pred: torch.Tensor, 
    decoder, 
    joint_names: List[str], 
    anatomical_limits: Dict,
    save_path: str,
    labels: Tuple[str, str] = ("Ground Truth", "Prediction")
):
    """
    Generate comparison plot (GT vs Pred) for all joint pairs.
    Wrapper that decodes latent tensors first.
    """
    meta_gt = decode_to_metadata(z_gt, decoder)
    meta_pred = decode_to_metadata(z_pred, decoder)
    
    plot_comparison(meta_gt, meta_pred, joint_names, anatomical_limits, save_path, labels)

def plot_comparison(
    meta_gt: Dict,
    meta_pred: Dict,
    joint_names: List[str],
    limits: Dict,
    save_path: str,
    labels: Tuple[str, str] = ("Ground Truth", "Prediction")
):
    """Generic comparison plotter using metadata dicts."""
    
    # Determine pairs (assume 4 joints -> 6 pairs)
    # If more joints, generate combinations
    import itertools
    n_joints = len(joint_names)
    pairs = list(itertools.combinations(range(n_joints), 2))
    n_pairs = len(pairs)
    
    fig, axes = plt.subplots(2, n_pairs, figsize=(4 * n_pairs, 8))
    
    # Row 1: GT
    for k, (i, j) in enumerate(pairs):
        ax = axes[0, k] if n_pairs > 1 else axes[0]
        plot_2d_projection(
            ax, meta_gt, (i, j), (joint_names[i], joint_names[j]), 
            limits, joint_names
        )
        if k == 0: ax.set_title(labels[0], fontsize=14, loc='left')
            
    # Row 2: Pred
    for k, (i, j) in enumerate(pairs):
        ax = axes[1, k] if n_pairs > 1 else axes[1]
        plot_2d_projection(
            ax, meta_pred, (i, j), (joint_names[i], joint_names[j]), 
            limits, joint_names
        )
        if k == 0: ax.set_title(labels[1], fontsize=14, loc='left')
            
    plt.suptitle(f"Structure Comparison: {labels[0]} vs {labels[1]}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison structure to {save_path}")


