import os
import shutil
import subprocess
import re
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Configuration
N_RUNS = 1
# Assuming we run from root directory
TEST_SCRIPT = os.path.join("test", "test_active_learning.py")
BASE_IMAGE_DIR = "images"

# Verify we are in the right directory
if not os.path.exists(TEST_SCRIPT):
    print(f"Error: Could not find {TEST_SCRIPT}. Please run this script from the 'test_selection' directory.")
    sys.exit(1)

# Ensure base image directory exists
if not os.path.exists(BASE_IMAGE_DIR):
    os.makedirs(BASE_IMAGE_DIR)

param_maes = []
param_stds = []
history_files = []

print(f"Starting batch run of {N_RUNS} iterations...")

for i in range(N_RUNS):
    run_id = i + 1
    print(f"Running iteration {run_id}/{N_RUNS}...")

    # Create run-specific directory inside images/
    run_dir = os.path.join(BASE_IMAGE_DIR, f"run_{run_id}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    # Run the test script
    # We run it as a subprocess. The script is expected to save images to 'images/'
    result = subprocess.run(["python", TEST_SCRIPT], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error in run {run_id}: Script failed with return code {result.returncode}")
        print(result.stderr)
        continue

    # Move generated images and history from images/ to images/run_{i}/
    # The test script saves 'metrics.png', 'joint_evolution.png', 'param_error.png' and 'history.json' to 'images/'
    source_files = ["metrics.png", "joint_evolution.png", "param_error.png", "history.json"]
    for file_name in source_files:
        src_path = os.path.join(BASE_IMAGE_DIR, file_name)
        dst_path = os.path.join(run_dir, file_name)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            if file_name == "history.json":
                history_files.append(dst_path)
        else:
            print(f"Warning: {file_name} not found in {BASE_IMAGE_DIR} for run {run_id}")

    # Save stdout for later inspection
    with open(os.path.join(run_dir, "output.txt"), "w") as f:
        f.write(result.stdout)

    # Parse errors from output
    mae_match = re.search(r"Parameter Mean Absolute Error: ([0-9.]+)", result.stdout)
    std_match = re.search(r"Average Parameter Std Dev: ([0-9.]+)", result.stdout)

    if mae_match and std_match:
        mae = float(mae_match.group(1))
        std = float(std_match.group(1))
        param_maes.append(mae)
        param_stds.append(std)
        print(f"  Run {run_id} - MAE: {mae:.4f}, Std Dev: {std:.4f}")
    else:
        print(f"Warning: Could not parse errors for run {run_id}")

# Compute averages
print("\n" + "="*30)
print("BATCH RUN RESULTS")
print("="*30)
if param_maes and param_stds:
    avg_mae = sum(param_maes) / len(param_maes)
    avg_std = sum(param_stds) / len(param_stds)
    print(f"Total successful runs: {len(param_maes)}/{N_RUNS}")
    print(f"Average Parameter MAE:      {avg_mae:.4f}")
    print(f"Average Parameter Std Dev:  {avg_std:.4f}")
else:
    print("No successful runs parsed.")

# Aggregate and Plot History
if history_files:
    print("\nAggregating results and generating plots...")

    all_elbos = []
    all_balds = []
    all_maes = []
    all_uncs = []

    for h_file in history_files:
        try:
            with open(h_file, 'r') as f:
                data = json.load(f)
                all_elbos.append(data['elbo'])
                all_balds.append(data['bald_score'])
                all_maes.append(data.get('param_mae', []))
                all_uncs.append(data.get('param_uncertainty', []))
        except Exception as e:
            print(f"Error reading {h_file}: {e}")

    if not all_elbos:
        print("No history data found.")
        sys.exit(0)

    # Convert to numpy arrays for easier stats
    # We'll assume they are same length for now, or truncate to min length
    min_len = min(len(x) for x in all_elbos)

    elbo_matrix = np.array([x[:min_len] for x in all_elbos])
    bald_matrix = np.array([x[:min_len] for x in all_balds])

    iterations = np.arange(min_len)

    # Calculate stats
    elbo_mean = np.mean(elbo_matrix, axis=0)
    elbo_std = np.std(elbo_matrix, axis=0)

    bald_mean = np.mean(bald_matrix, axis=0)
    bald_std = np.std(bald_matrix, axis=0)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ELBO
    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('ELBO', color=color)
    ax1.plot(iterations, elbo_mean, color=color, label='Mean ELBO')
    ax1.fill_between(iterations, elbo_mean - elbo_std, elbo_mean + elbo_std, color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)

    # BALD
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('BALD Score', color=color)
    ax2.plot(iterations, bald_mean, color=color, linestyle='--', label='Mean BALD')
    ax2.fill_between(iterations, bald_mean - bald_std, bald_mean + bald_std, color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Aggregated Active Learning Metrics ({len(history_files)} runs)')
    fig.tight_layout()

    agg_plot_path = os.path.join(BASE_IMAGE_DIR, 'aggregated_metrics.png')
    plt.savefig(agg_plot_path)
    plt.close()
    print(f"Aggregated plot saved to {agg_plot_path}")

    # Plot aggregated MAE and uncertainty
    all_maes = [x for x in all_maes if x]
    all_uncs = [x for x in all_uncs if x]
    if all_maes and all_uncs:
        min_len_mae = min(len(x) for x in all_maes)
        mae_matrix = np.array([x[:min_len_mae] for x in all_maes])
        unc_matrix = np.array([x[:min_len_mae] for x in all_uncs])
        mae_mean, mae_std = np.mean(mae_matrix, axis=0), np.std(mae_matrix, axis=0)
        unc_mean, unc_std = np.mean(unc_matrix, axis=0), np.std(unc_matrix, axis=0)
        iters = np.arange(min_len_mae)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MAE (rad)', color='tab:red')
        ax1.plot(iters, mae_mean, color='tab:red', label='Mean MAE')
        ax1.fill_between(iters, mae_mean - mae_std, mae_mean + mae_std, color='tab:red', alpha=0.2)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Uncertainty (std)', color='tab:blue')
        ax2.plot(iters, unc_mean, color='tab:blue', linestyle='--', label='Mean Uncertainty')
        ax2.fill_between(iters, unc_mean - unc_std, unc_mean + unc_std, color='tab:blue', alpha=0.2)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title(f'Aggregated MAE & Uncertainty ({len(all_maes)} runs)')
        fig.tight_layout()
        agg_mae_path = os.path.join(BASE_IMAGE_DIR, 'aggregated_param_error.png')
        plt.savefig(agg_mae_path)
        plt.close()
        print(f"Aggregated MAE plot saved to {agg_mae_path}")
