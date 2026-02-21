#!/usr/bin/env python3
"""
Generate predictions from a trained model checkpoint.
Supports two modes:
1. Evaluate on original data (for comparison with ground truth)
2. Generate grid predictions (for dense visualization)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
import shutil
from pathlib import Path
from omegaconf import OmegaConf

from models import get_model
from dataset import load_compiled_data


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Trying with weights_only=False...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config = checkpoint['config']
    
    # Create model
    model_config = {k: v for k, v in config.model.items() if k != '_target_'}
    model_class_name = config.model._target_.split('.')[-1]
    model_name_map = {
        'DualEncoderModel': 'dual_encoder',
        'CrossAttentionModel': 'cross_attention',
        'PhysicsInformedModel': 'physics_informed',
        'HierarchicalModel': 'hierarchical',
        'VariationalModel': 'variational'
    }
    model_name = model_name_map.get(model_class_name, model_class_name.lower())
    
    model = get_model(model_name, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, config


def get_all_model_data(compiled_results_dir: str):
    """Load data from all available models."""
    
    # Find all complete model datasets
    model_files = []
    for f in os.listdir(compiled_results_dir):
        if f.endswith('_muscle_params.json'):
            model_name = f.replace('_muscle_params.json', '')
            results_file = f"{model_name}_results.txt"
            
            muscle_path = os.path.join(compiled_results_dir, f)
            results_path = os.path.join(compiled_results_dir, results_file)
            
            if os.path.exists(results_path):
                model_files.append((model_name, muscle_path, results_path))
    
    if not model_files:
        raise FileNotFoundError("No complete model datasets found")
    
    print(f"Found {len(model_files)} complete model datasets:")
    for model_name, _, _ in model_files:
        print(f"  - {model_name}")
    
    return model_files


def prepare_muscle_params(compiled_results_dir: str, model_name: str = None):
    """Load muscle parameters for a specific model or use first available."""
    
    if model_name:
        # Load specific model
        muscle_file = f"{model_name}_muscle_params.json"
        muscle_path = os.path.join(compiled_results_dir, muscle_file)
        
        if not os.path.exists(muscle_path):
            raise FileNotFoundError(f"Muscle parameters not found for model: {model_name}")
        
        print(f"Using muscle parameters from: {model_name}")
        
        with open(muscle_path, 'r') as f:
            muscle_data = json.load(f)
        
        muscle_names = sorted(muscle_data.keys())
        muscle_vector = np.array([muscle_data[name] for name in muscle_names])
        
        print(f"Loaded {len(muscle_names)} muscles")
        print(f"Non-default muscles: {sum(1 for x in muscle_vector if x != 1.0)}")
        
        return muscle_vector, muscle_names, model_name
    else:
        # Process all models
        model_data = get_all_model_data(compiled_results_dir)
        return model_data


def mode_original_data(model, config, compiled_results_dir: str, model_name: str, 
                      output_path: str, device: torch.device):
    """Mode 1: Evaluate on original data for comparison."""
    
    print(f"\n=== MODE 1: Original Data Evaluation ===")
    
    if model_name:
        # Single model mode
        muscle_vector, muscle_names, used_model_name = prepare_muscle_params(compiled_results_dir, model_name)
        model_data = [(used_model_name, muscle_vector, muscle_names)]
    else:
        # All models mode
        all_model_files = prepare_muscle_params(compiled_results_dir, None)
        model_data = []
        
        # Load muscle parameters for all models
        for model_name_iter, muscle_path, results_path in all_model_files:
            with open(muscle_path, 'r') as f:
                muscle_dict = json.load(f)
            muscle_names = sorted(muscle_dict.keys())
            muscle_vector = np.array([muscle_dict[name] for name in muscle_names])
            model_data.append((model_name_iter, muscle_vector, muscle_names))
    
    all_results = []
    combined_data = []
    
    for used_model_name, muscle_vector, muscle_names in model_data:
        print(f"\nProcessing model: {used_model_name}")
        
        # Load original results file
        results_file = os.path.join(compiled_results_dir, f"{used_model_name}_results.txt")
        if not os.path.exists(results_file):
            print(f"Warning: Results file not found for {used_model_name}, skipping")
            continue
        
        # Read original data
        results_df = pd.read_csv(results_file, sep=' ', comment='#', 
                                names=['x', 'y', 'z', 'distance'])
        
        print(f"  Loaded {len(results_df)} data points")
        print(f"  Distance range: {results_df['distance'].min():.6f} to {results_df['distance'].max():.6f}")
        print(f"  Non-default muscles: {sum(1 for x in muscle_vector if x != 1.0)}")
    
        # Store for combined processing
        combined_data.append({
            'model_name': used_model_name,
            'muscle_vector': muscle_vector,
            'results_df': results_df
        })
    
    print(f"\nGenerating predictions for {len(combined_data)} models...")
    
    # Process all models together
    all_muscle_params = []
    all_target_poses = []
    all_original_distances = []
    all_model_names = []
    
    for data in combined_data:
        model_name_iter = data['model_name']
        muscle_vector = data['muscle_vector']
        results_df = data['results_df']
        
        n_samples = len(results_df)
        muscle_params = np.tile(muscle_vector, (n_samples, 1))
        target_poses = results_df[['x', 'y', 'z']].values
        
        all_muscle_params.append(muscle_params)
        all_target_poses.append(target_poses)
        all_original_distances.append(results_df['distance'].values)
        all_model_names.extend([model_name_iter] * n_samples)
    
    # Combine all data
    all_muscle_params = np.vstack(all_muscle_params)
    all_target_poses = np.vstack(all_target_poses)
    all_original_distances = np.concatenate(all_original_distances)
    
    total_samples = len(all_muscle_params)
    print(f"Total samples across all models: {total_samples}")
    
    # Keep original poses for output (BEFORE any scaling)
    all_target_poses_original = all_target_poses.copy()
    
    # Handle data scaling based on training config
    scalers = None
    scale_back_predictions = False
    
    if hasattr(config.data, 'disable_scaling') and config.data.disable_scaling:
        print("Using unscaled data (matching training)")
    else:
        print("Data was scaled during training - loading scalers from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
            scale_back_predictions = True
            print("Scalers loaded successfully from checkpoint")
            
            # Apply input scaling (same as training)
            muscle_scaler = scalers['muscle'] 
            pose_scaler = scalers['pose']
            
            if muscle_scaler is not None:
                muscle_std = np.std(all_muscle_params, axis=0)
                if not np.all(muscle_std < 1e-10):
                    all_muscle_params = muscle_scaler.transform(all_muscle_params)
            
            if pose_scaler is not None:
                all_target_poses = pose_scaler.transform(all_target_poses)
        else:
            print("No scalers found in checkpoint - using unscaled data")
    
    # Convert to tensors
    muscle_tensor = torch.FloatTensor(all_muscle_params).to(device)
    pose_tensor = torch.FloatTensor(all_target_poses).to(device)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(muscle_tensor), batch_size):
            batch_muscles = muscle_tensor[i:i+batch_size]
            batch_poses = pose_tensor[i:i+batch_size]
            
            batch_preds = model(batch_muscles, batch_poses)
            predictions.extend(batch_preds.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Scale back predictions if needed
    if scale_back_predictions and scalers is not None:
        distance_scaler = scalers['distance']
        if distance_scaler is not None and hasattr(distance_scaler, 'mean_') and hasattr(distance_scaler, 'scale_'):
            scale = distance_scaler.scale_[0] if hasattr(distance_scaler.scale_, '__len__') else distance_scaler.scale_
            mean = distance_scaler.mean_[0] if hasattr(distance_scaler.mean_, '__len__') else distance_scaler.mean_
            predictions = predictions * scale + mean
            print(f"Predictions unscaled to original range: {predictions.min():.6f} to {predictions.max():.6f}")
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction range: {predictions.min():.6f} to {predictions.max():.6f}")
    
    # Calculate metrics for all models combined
    mse = np.mean((predictions - all_original_distances) ** 2)
    mae = np.mean(np.abs(predictions - all_original_distances))
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((all_original_distances - predictions) ** 2)
    ss_tot = np.sum((all_original_distances - np.mean(all_original_distances)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\nCombined Evaluation Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²: {r2:.4f}")
    
    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)
    
    # Save results per model in separate files (same structure as compiled_results)
    per_model_metrics = {}
    start_idx = 0
    all_output_files = []
    
    for data in combined_data:
        model_name_iter = data['model_name']
        n_samples = len(data['results_df'])
        end_idx = start_idx + n_samples
        
        model_predictions = predictions[start_idx:end_idx]
        model_gt = data['results_df']['distance'].values
        model_poses = all_target_poses_original[start_idx:end_idx]
        
        # Calculate per-model metrics
        model_mse = np.mean((model_predictions - model_gt) ** 2)
        model_mae = np.mean(np.abs(model_predictions - model_gt))
        model_rmse = np.sqrt(model_mse)
        
        model_ss_res = np.sum((model_gt - model_predictions) ** 2)
        model_ss_tot = np.sum((model_gt - np.mean(model_gt)) ** 2)
        model_r2 = 1 - (model_ss_res / model_ss_tot)
        
        per_model_metrics[model_name_iter] = {
            'mse': model_mse, 'mae': model_mae, 'rmse': model_rmse, 'r2': model_r2
        }
        
        print(f"\nMetrics for {model_name_iter}:")
        print(f"  MSE: {model_mse:.6f}")
        print(f"  MAE: {model_mae:.6f}")
        print(f"  RMSE: {model_rmse:.6f}")
        print(f"  R²: {model_r2:.4f}")
        
        # Save individual model results file (same structure as compiled_results)
        model_output_file = os.path.join(output_path, f"{model_name_iter}_results.txt")
        with open(model_output_file, 'w') as f:
            f.write(f"# {model_name_iter} prediction results\n")
            f.write(f"# Format: x y z distance\n")
            f.write(f"# Total data points: {n_samples}\n")
            f.write(f"#\n")
            
            for i in range(n_samples):
                x, y, z = model_poses[i]
                pred = model_predictions[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {pred:.6f}\n")
        
        print(f"  Results saved to: {model_output_file}")
        all_output_files.append(model_output_file)
        start_idx = end_idx
    
    # Also save combined summary
    summary_file = os.path.join(output_path, "combined_summary.txt")
    model_names_str = ", ".join(set(all_model_names))
    with open(summary_file, 'w') as f:
        f.write(f"# Combined Summary for models: {model_names_str}\n")
        f.write(f"# Model checkpoint: {checkpoint_path}\n")
        f.write(f"# Combined Metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}\n")
        f.write(f"# Total samples: {total_samples}\n")
        f.write(f"#\n")
        f.write(f"# Per-model metrics:\n")
        for model_name_iter, metrics in per_model_metrics.items():
            f.write(f"# {model_name_iter}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}\n")
        f.write(f"#\n")
        f.write(f"# Individual result files:\n")
        for output_file in all_output_files:
            f.write(f"# {os.path.basename(output_file)}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Individual results saved for {len(combined_data)} models in: {output_path}")
    
    return per_model_metrics, all_output_files


def mode_grid_sampling(model, config, compiled_results_dir: str, model_name: str, 
                      output_path: str, device: torch.device, grid_size: int = 20, checkpoint_path: str = None):
    """Mode 2: Generate predictions on a grid for visualization."""
    
    print(f"\n=== MODE 2: Grid Sampling ===")
    
    # Handle data scaling based on training config (same as original mode)
    scalers = None
    scale_back_predictions = False
    
    if hasattr(config.data, 'disable_scaling') and config.data.disable_scaling:
        print("Using unscaled data (matching training)")
    else:
        print("Data was scaled during training - loading scalers from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'scalers' in checkpoint:
            scalers = checkpoint['scalers']
            scale_back_predictions = True
            print("Scalers loaded successfully from checkpoint")
        else:
            print("No scalers found in checkpoint - using unscaled data")
    
    if model_name:
        # Single model mode
        muscle_vector, muscle_names, used_model_name = prepare_muscle_params(compiled_results_dir, model_name)
        model_data = [(used_model_name, muscle_vector, muscle_names)]
    else:
        # All models mode
        all_model_files = prepare_muscle_params(compiled_results_dir, None)
        model_data = []
        
        # Load muscle parameters for all models
        for model_name_iter, muscle_path, results_path in all_model_files:
            with open(muscle_path, 'r') as f:
                muscle_dict = json.load(f)
            muscle_names = sorted(muscle_dict.keys())
            muscle_vector = np.array([muscle_dict[name] for name in muscle_names])
            model_data.append((model_name_iter, muscle_vector, muscle_names))
    
    # Initialize results list
    all_results = []
    
    # Create grid
    x_range = np.linspace(-0.65, 0.65, grid_size)
    y_range = np.linspace(0.6, 1.9, grid_size) 
    z_range = np.linspace(-0.5, 0.8, grid_size)
    
    print(f"Creating {grid_size}³ = {grid_size**3} grid points")
    print(f"X range: {x_range[0]:.3f} to {x_range[-1]:.3f}")
    print(f"Y range: {y_range[0]:.3f} to {y_range[-1]:.3f}")
    print(f"Z range: {z_range[0]:.3f} to {z_range[-1]:.3f}")
    
    # Generate all grid combinations
    grid_points = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                grid_points.append([x, y, z])
    
    grid_points = np.array(grid_points)
    n_points = len(grid_points)
    
    # Process each model
    for used_model_name, muscle_vector, muscle_names in model_data:
        print(f"\nProcessing model: {used_model_name}")
        print(f"  Non-default muscles: {sum(1 for x in muscle_vector if x != 1.0)}")
        
        # Prepare muscle parameters (same for all grid points)
        muscle_params = np.tile(muscle_vector, (n_points, 1))
        
        # Handle scaling (exactly same as original mode)
        if hasattr(config.data, 'disable_scaling') and config.data.disable_scaling:
            muscle_params_scaled = muscle_params
            target_poses_scaled = grid_points
        else:
            # Apply same scaling as original mode
            muscle_params_scaled = muscle_params.copy()
            target_poses_scaled = grid_points.copy()
            
            if scale_back_predictions and scalers is not None:
                muscle_scaler = scalers['muscle'] 
                pose_scaler = scalers['pose']
                
                if muscle_scaler is not None:
                    muscle_std = np.std(muscle_params_scaled, axis=0)
                    if not np.all(muscle_std < 1e-10):
                        muscle_params_scaled = muscle_scaler.transform(muscle_params_scaled)
                
                if pose_scaler is not None:
                    target_poses_scaled = pose_scaler.transform(target_poses_scaled)
        
        # Convert to tensors
        muscle_tensor = torch.FloatTensor(muscle_params_scaled).to(device)
        pose_tensor = torch.FloatTensor(target_poses_scaled).to(device)
        
        # Generate predictions
        print("Generating grid predictions...")
        predictions = []
        batch_size = 1000  # Larger batch for grid generation
        
        with torch.no_grad():
            for i in range(0, len(muscle_tensor), batch_size):
                batch_muscles = muscle_tensor[i:i+batch_size]
                batch_poses = pose_tensor[i:i+batch_size]
                
                batch_preds = model(batch_muscles, batch_poses)
                predictions.extend(batch_preds.cpu().numpy())
                
                if (i // batch_size) % 10 == 0:
                    print(f"  Processed {i + len(batch_muscles)}/{n_points} points...")
        
        predictions = np.array(predictions)
        
        # Scale back predictions if needed (same as original mode)
        if scale_back_predictions and scalers is not None:
            distance_scaler = scalers['distance']
            if distance_scaler is not None and hasattr(distance_scaler, 'mean_') and hasattr(distance_scaler, 'scale_'):
                scale = distance_scaler.scale_[0] if hasattr(distance_scaler.scale_, '__len__') else distance_scaler.scale_
                mean = distance_scaler.mean_[0] if hasattr(distance_scaler.mean_, '__len__') else distance_scaler.mean_
                predictions = predictions * scale + mean
        
        print(f"Generated {len(predictions)} predictions")
        print(f"Prediction range: {predictions.min():.6f} to {predictions.max():.6f}")
        print(f"Mean prediction: {predictions.mean():.6f}")
        
        all_results.append((used_model_name, grid_points, predictions))
    
    # Create output directory if needed
    os.makedirs(output_path, exist_ok=True)
    
    # Save results per model in separate files
    all_output_files = []
    for model_name_iter, grid_points_iter, predictions_iter in all_results:
        model_output_file = os.path.join(output_path, f"{model_name_iter}_grid_predictions.txt")
        with open(model_output_file, 'w') as f:
            f.write(f"# Grid predictions for {model_name_iter}\n")
            f.write(f"# Model checkpoint: {checkpoint_path}\n")
            f.write(f"# Grid size: {grid_size}³ = {n_points} points\n")
            f.write(f"# X range: [{x_range[0]:.3f}, {x_range[-1]:.3f}]\n")
            f.write(f"# Y range: [{y_range[0]:.3f}, {y_range[-1]:.3f}]\n")
            f.write(f"# Z range: [{z_range[0]:.3f}, {z_range[-1]:.3f}]\n")
            f.write(f"# Format: x y z distance_pred\n")
            f.write(f"#\n")
            
            for point, pred in zip(grid_points_iter, predictions_iter):
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {pred:.6f}\n")
        
        print(f"Grid results for {model_name_iter} saved to: {model_output_file}")
        all_output_files.append(model_output_file)
    
    # Also save combined summary for grid mode
    summary_file = os.path.join(output_path, "grid_summary.txt")
    model_names_str = ", ".join([result[0] for result in all_results])
    with open(summary_file, 'w') as f:
        f.write(f"# Grid Summary for models: {model_names_str}\n")
        f.write(f"# Model checkpoint: {checkpoint_path}\n")
        f.write(f"# Grid size: {grid_size}³ = {n_points} points\n")
        f.write(f"# X range: [{x_range[0]:.3f}, {x_range[-1]:.3f}]\n")
        f.write(f"# Y range: [{y_range[0]:.3f}, {y_range[-1]:.3f}]\n")
        f.write(f"# Z range: [{z_range[0]:.3f}, {z_range[-1]:.3f}]\n")
        f.write(f"#\n")
        f.write(f"# Individual grid files:\n")
        for output_file in all_output_files:
            f.write(f"# {os.path.basename(output_file)}\n")
        f.write(f"#\n")
        f.write(f"# Per-model statistics:\n")
        for model_name_iter, grid_points_iter, predictions_iter in all_results:
            f.write(f"# {model_name_iter}: min={predictions_iter.min():.6f}, max={predictions_iter.max():.6f}, mean={predictions_iter.mean():.6f}\n")
    
    print(f"\nGrid summary saved to: {summary_file}")
    print(f"Individual grid results saved for {len(all_results)} models in: {output_path}")
    
    return all_results, all_output_files


def main():
    parser = argparse.ArgumentParser(description='Generate predictions from trained model')
    parser.add_argument('checkpoint_path', type=str, help='Path to model checkpoint (.pth file)')
    parser.add_argument('output_path', type=str, help='Output file path')
    parser.add_argument('--mode', type=str, choices=['original', 'grid'], default='original',
                       help='Generation mode: original data or grid sampling')
    parser.add_argument('--compiled_results_dir', type=str, default='../compiled_results',
                       help='Path to compiled results directory')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Specific model name to use (e.g., single_arm_c6_3). Uses all available models if not specified.')
    parser.add_argument('--grid_size', type=int, default=10,
                       help='Grid size for grid mode (creates grid_size³ points)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    global checkpoint_path  # For use in mode functions
    checkpoint_path = args.checkpoint_path
    model, config = load_model_from_checkpoint(args.checkpoint_path, device)
    
    # Generate predictions
    if args.mode == 'original':
        result = mode_original_data(model, config, args.compiled_results_dir, 
                                  args.model_name, args.output_path, device)
    else:  # grid
        result = mode_grid_sampling(model, config, args.compiled_results_dir, 
                                   args.model_name, args.output_path, device, args.grid_size, args.checkpoint_path)
    
    print("\n=== Generation Complete ===")


if __name__ == "__main__":
    main()