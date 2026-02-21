#!/usr/bin/env python3
"""
Dataset loading and preprocessing for SCONE muscle parameter to reach distance prediction.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SCONEReachDataset(Dataset):
    """Dataset for SCONE reach distance prediction from muscle parameters and target pose."""
    
    def __init__(
        self,
        muscle_params: np.ndarray,
        target_poses: np.ndarray, 
        distances: np.ndarray,
        muscle_names: List[str],
        muscle_scaler: Optional[StandardScaler] = None,
        pose_scaler: Optional[StandardScaler] = None,
        distance_scaler: Optional[StandardScaler] = None,
        fit_scalers: bool = False
    ):
        """
        Args:
            muscle_params: (N, 40) muscle parameter values
            target_poses: (N, 3) target pose coordinates
            distances: (N,) reach distances
            muscle_names: List of muscle names for reference
            muscle_scaler: Scaler for muscle parameters
            pose_scaler: Scaler for target poses  
            distance_scaler: Scaler for distances
            fit_scalers: Whether to fit the scalers on this data
        """
        self.muscle_names = muscle_names
        
        # Initialize scalers if needed
        if muscle_scaler is None:
            self.muscle_scaler = StandardScaler()
        else:
            self.muscle_scaler = muscle_scaler
            
        if pose_scaler is None:
            self.pose_scaler = StandardScaler()
        else:
            self.pose_scaler = pose_scaler
            
        if distance_scaler is None:
            self.distance_scaler = StandardScaler()
        else:
            self.distance_scaler = distance_scaler
        
        # Fit scalers if requested
        if fit_scalers:
            # Check if muscle parameters have any variation
            muscle_std = np.std(muscle_params, axis=0)
            if np.all(muscle_std < 1e-10):  # All identical
                print("WARNING: All muscle parameters are identical - skipping muscle scaling")
                # Don't scale muscles, just copy
                muscle_params = muscle_params.copy()
            else:
                muscle_params = self.muscle_scaler.fit_transform(muscle_params)
            
            target_poses = self.pose_scaler.fit_transform(target_poses)
            
            # Don't center distances - just scale them
            distances_2d = distances.reshape(-1, 1)
            distances_mean = np.mean(distances_2d)
            distances_std = np.std(distances_2d)
            print(f"Distance stats: mean={distances_mean:.6f}, std={distances_std:.6f}")
            
            # Manual scaling to keep distances positive
            if distances_std > 1e-10:
                distances = (distances - distances_mean) / distances_std
                # Store the scaling parameters in the scaler for later use
                self.distance_scaler.mean_ = np.array([distances_mean])
                self.distance_scaler.scale_ = np.array([distances_std])
                print(f"After scaling - Distance range: {distances.min():.6f} to {distances.max():.6f}")
            else:
                distances = distances.copy()
                self.distance_scaler.mean_ = np.array([0.0])
                self.distance_scaler.scale_ = np.array([1.0])
        else:
            # Check if we should skip muscle scaling
            muscle_std = np.std(muscle_params, axis=0)
            if np.all(muscle_std < 1e-10):
                muscle_params = muscle_params.copy()
            else:
                muscle_params = self.muscle_scaler.transform(muscle_params)
                
            target_poses = self.pose_scaler.transform(target_poses)
            
            # Apply the same scaling as training
            distances_2d = distances.reshape(-1, 1)
            if hasattr(self.distance_scaler, 'mean_') and hasattr(self.distance_scaler, 'scale_'):
                distances = (distances - self.distance_scaler.mean_) / self.distance_scaler.scale_
            else:
                distances = distances.copy()
        
        # Convert to tensors
        self.muscle_params = torch.FloatTensor(muscle_params)
        self.target_poses = torch.FloatTensor(target_poses)
        self.distances = torch.FloatTensor(distances)
        
        assert len(self.muscle_params) == len(self.target_poses) == len(self.distances)
        
    def __len__(self):
        return len(self.muscle_params)
    
    def __getitem__(self, idx):
        return {
            'muscle_params': self.muscle_params[idx],
            'target_pose': self.target_poses[idx], 
            'distance': self.distances[idx]
        }


class SimpleDataset(Dataset):
    """Simple dataset without scaling for debugging."""
    
    def __init__(self, muscle_params: np.ndarray, target_poses: np.ndarray, distances: np.ndarray):
        self.muscle_params = torch.FloatTensor(muscle_params)
        self.target_poses = torch.FloatTensor(target_poses)
        self.distances = torch.FloatTensor(distances)
        
        print(f"SimpleDataset created:")
        print(f"  Muscle params shape: {self.muscle_params.shape}")
        print(f"  Target poses shape: {self.target_poses.shape}")  
        print(f"  Distances shape: {self.distances.shape}")
        
        if len(self.muscle_params) > 0:
            print(f"  Sample muscle params: {self.muscle_params[0][:10]}")
            print(f"  Sample target pose: {self.target_poses[0]}")
            print(f"  Distance range: {self.distances.min():.6f} to {self.distances.max():.6f}")
        else:
            print(f"  Empty dataset (for validation/testing skip)")
        
    def __len__(self):
        return len(self.muscle_params)
    
    def __getitem__(self, idx):
        return {
            'muscle_params': self.muscle_params[idx],
            'target_pose': self.target_poses[idx],
            'distance': self.distances[idx]
        }


def load_compiled_data(compiled_results_dir: str, max_models: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load data from compiled_results directory.
    
    Args:
        compiled_results_dir: Path to compiled results
        max_models: If provided, only load data from first N models (for debugging)
    
    Returns:
        muscle_params: (N, num_muscles) array
        target_poses: (N, 3) array  
        distances: (N,) array
        muscle_names: List of muscle names
    """
    
    all_muscle_params = []
    all_target_poses = []
    all_distances = []
    muscle_names = None
    
    # Find all model files
    model_files = []
    for f in os.listdir(compiled_results_dir):
        if f.endswith('_results.txt'):
            model_name = f.replace('_results.txt', '')
            muscle_file = f"{model_name}_muscle_params.json"
            
            results_path = os.path.join(compiled_results_dir, f)
            muscle_path = os.path.join(compiled_results_dir, muscle_file)
            
            if os.path.exists(results_path) and os.path.exists(muscle_path):
                model_files.append((model_name, results_path, muscle_path))
    
    print(f"Found {len(model_files)} complete model datasets")
    
    # Limit number of models if requested
    if max_models is not None:
        model_files = model_files[:max_models]
        print(f"Using first {len(model_files)} models only")
    
    for model_name, results_path, muscle_path in model_files:
        print(f"Loading {model_name}...")
        
        # Load muscle parameters
        with open(muscle_path, 'r') as f:
            muscle_data = json.load(f)
        
        if muscle_names is None:
            muscle_names = sorted(muscle_data.keys())
            print(f"Found {len(muscle_names)} muscles: {muscle_names[:5]}...")
        
        # Create muscle parameter vector
        muscle_vector = np.array([muscle_data[name] for name in muscle_names])
        
        # Load results data
        results_df = pd.read_csv(results_path, sep=' ', comment='#', 
                                names=['x', 'y', 'z', 'distance'])
        
        print(f"  {len(results_df)} data points")
        
        # Each result point gets the same muscle parameters
        for _, row in results_df.iterrows():
            all_muscle_params.append(muscle_vector.copy())
            all_target_poses.append([row['x'], row['y'], row['z']])
            all_distances.append(row['distance'])
    
    # Convert to numpy arrays
    muscle_params = np.array(all_muscle_params)
    target_poses = np.array(all_target_poses)  
    distances = np.array(all_distances)
    
    print(f"Loaded dataset: {len(muscle_params)} total samples")
    print(f"Muscle params shape: {muscle_params.shape}")
    print(f"Target poses shape: {target_poses.shape}")
    print(f"Distances shape: {distances.shape}")
    print(f"Distance range: {distances.min():.6f} to {distances.max():.6f}")
    
    return muscle_params, target_poses, distances, muscle_names


def create_dataloaders(
    compiled_results_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 32,
    random_seed: int = 42,
    num_workers: int = 4,
    max_models: int = None,
    disable_scaling: bool = False,
    skip_validation: bool = False,
    skip_testing: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train/val/test dataloaders from compiled results.
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    
    # Load data
    muscle_params, target_poses, distances, muscle_names = load_compiled_data(compiled_results_dir, max_models)
    
    # Handle data splitting based on skip flags
    indices = np.arange(len(muscle_params))
    
    if skip_validation and skip_testing:
        # Use all data for training (overfitting mode)
        print("OVERFITTING MODE: Using 100% of data for training")
        train_idx = indices
        val_idx = np.array([])  # Empty
        test_idx = np.array([])  # Empty
    else:
        # Normal split
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        train_idx, temp_idx = train_test_split(
            indices, test_size=(1-train_ratio), random_state=random_seed, stratify=None
        )
        
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1-val_test_ratio), random_state=random_seed
        )
    
    print(f"Dataset split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    
    if disable_scaling:
        print("WARNING: Data scaling disabled - using raw values")
        
    # Create datasets
    if disable_scaling:
        # Simple dataset without scaling
        train_dataset = SimpleDataset(muscle_params[train_idx], target_poses[train_idx], distances[train_idx])
        
        # Create empty datasets if validation/testing are skipped
        if len(val_idx) > 0:
            val_dataset = SimpleDataset(muscle_params[val_idx], target_poses[val_idx], distances[val_idx])
        else:
            val_dataset = SimpleDataset(np.empty((0, len(muscle_names))), np.empty((0, 3)), np.empty((0,)))
            
        if len(test_idx) > 0:
            test_dataset = SimpleDataset(muscle_params[test_idx], target_poses[test_idx], distances[test_idx])
        else:
            test_dataset = SimpleDataset(np.empty((0, len(muscle_names))), np.empty((0, 3)), np.empty((0,)))
        
        # Create dummy scalers for metadata
        muscle_scaler = None
        pose_scaler = None
        distance_scaler = None
    else:
        train_dataset = SCONEReachDataset(
            muscle_params[train_idx],
            target_poses[train_idx], 
            distances[train_idx],
            muscle_names,
            fit_scalers=True  # Fit scalers on training data
        )
        
        # Create empty/dummy datasets if validation/testing are skipped
        if len(val_idx) > 0:
            val_dataset = SCONEReachDataset(
                muscle_params[val_idx],
                target_poses[val_idx],
                distances[val_idx], 
                muscle_names,
                muscle_scaler=train_dataset.muscle_scaler,
                pose_scaler=train_dataset.pose_scaler,
                distance_scaler=train_dataset.distance_scaler,
                fit_scalers=False
            )
        else:
            val_dataset = SimpleDataset(np.empty((0, len(muscle_names))), np.empty((0, 3)), np.empty((0,)))
            
        if len(test_idx) > 0:
            test_dataset = SCONEReachDataset(
                muscle_params[test_idx],
                target_poses[test_idx],
                distances[test_idx],
                muscle_names, 
                muscle_scaler=train_dataset.muscle_scaler,
                pose_scaler=train_dataset.pose_scaler,
                distance_scaler=train_dataset.distance_scaler,
                fit_scalers=False
            )
        else:
            test_dataset = SimpleDataset(np.empty((0, len(muscle_names))), np.empty((0, 3)), np.empty((0,)))
        
        # Get scalers for metadata
        muscle_scaler = train_dataset.muscle_scaler
        pose_scaler = train_dataset.pose_scaler 
        distance_scaler = train_dataset.distance_scaler
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Metadata
    metadata = {
        'muscle_names': muscle_names,
        'num_muscles': len(muscle_names),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset), 
        'test_size': len(test_dataset),
        'scalers': {
            'muscle': muscle_scaler,
            'pose': pose_scaler,
            'distance': distance_scaler
        }
    }
    
    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    # Test data loading
    compiled_dir = "../compiled_results"
    train_loader, val_loader, test_loader, metadata = create_dataloaders(compiled_dir)
    
    print(f"\nMetadata: {metadata}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"Muscle params: {batch['muscle_params'].shape}")
    print(f"Target poses: {batch['target_pose'].shape}")
    print(f"Distances: {batch['distance'].shape}")