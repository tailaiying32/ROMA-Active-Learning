#!/usr/bin/env python3
"""
Training script for SCONE reach prediction models.
"""

import os
import time
import hydra
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path
from typing import Dict, Any

from dataset import create_dataloaders
from models import get_model


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, monitor: str = "val_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def get_device(device_config: str) -> torch.device:
    """Get the appropriate device."""
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
    
    return device


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float):
    """Get optimizer."""
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name: str, **kwargs):
    """Get learning rate scheduler."""
    if scheduler_name == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('epochs', 100))
    elif scheduler_name == "step":
        return StepLR(optimizer, step_size=kwargs.get('step_size', 30), gamma=0.1)
    elif scheduler_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics."""
    mse = nn.MSELoss()(predictions, targets).item()
    mae = nn.L1Loss()(predictions, targets).item()
    
    with torch.no_grad():
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # R-squared
        ss_res = np.sum((targets_np - predictions_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # RMSE
        rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def train_epoch(model, train_loader, optimizer, device, loss_type="mse", grad_clip=None, debug_mode=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Debug prints for first batch of first few epochs
        if debug_mode and batch_idx == 0:
            print(f"DEBUG - Batch {batch_idx}:")
            print(f"  Muscle params shape: {batch['muscle_params'].shape}")
            print(f"  Target pose shape: {batch['target_pose'].shape}")  
            print(f"  Distance shape: {batch['distance'].shape}")
            print(f"  Sample muscle params (first 5, first 10 muscles): {batch['muscle_params'][:5, :10]}")  
            print(f"  Sample target poses (first 5): {batch['target_pose'][:5]}")
            print(f"  Sample distances (first 5): {batch['distance'][:5]}")
            print(f"  Distance range in batch: {batch['distance'].min():.6f} to {batch['distance'].max():.6f}")
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model.forward_batch(batch)
        loss = model.compute_loss(predictions, batch['distance'], loss_type)
        
        # Debug predictions for first batch
        if debug_mode and batch_idx == 0:
            print(f"  Sample predictions (first 5): {predictions[:5]}")
            print(f"  Sample targets (first 5): {batch['distance'][:5]}")
            print(f"  Sample diffs (first 5): {torch.abs(predictions[:5] - batch['distance'][:5])}")
            print(f"  Prediction range in batch: {predictions.min():.6f} to {predictions.max():.6f}")
            print(f"  Loss: {loss.item():.6f}")
            print()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        all_predictions.append(predictions.detach())
        all_targets.append(batch['distance'].detach())
    
    # Compute epoch metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics


def validate_epoch(model, val_loader, device, loss_type="mse"):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            predictions = model.forward_batch(batch)
            loss = model.compute_loss(predictions, batch['distance'], loss_type)
            
            # Accumulate metrics
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(batch['distance'])
    
    # Compute epoch metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    # Initialize wandb
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            entity=cfg.logging.entity,
            name=f"{cfg.experiment_name}_{int(time.time())}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.logging.tags,
            mode="offline" if cfg.logging.offline else "online"
        )
    
    # Get device
    device = get_device(cfg.training.device)
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        compiled_results_dir=cfg.data.compiled_results_dir,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        random_seed=cfg.data.random_seed,
        max_models=cfg.data.max_models,
        disable_scaling=cfg.data.disable_scaling,
        skip_validation=cfg.training.skip_validation,
        skip_testing=cfg.training.skip_testing
    )
    
    # Create model
    print("Creating model...")
    model_config = {k: v for k, v in cfg.model.items() if k != '_target_'}
    # Override config values with actual data dimensions
    model_config['num_muscles'] = metadata['num_muscles']
    
    # Extract model name from _target_ (e.g., "models.DualEncoderModel" -> "dual_encoder")
    model_class_name = cfg.model._target_.split('.')[-1]
    model_name_map = {
        'DualEncoderModel': 'dual_encoder',
        'CrossAttentionModel': 'cross_attention',
        'PhysicsInformedModel': 'physics_informed',
        'HierarchicalModel': 'hierarchical',
        'VariationalModel': 'variational'
    }
    model_name = model_name_map.get(model_class_name, model_class_name.lower())
    
    model = get_model(model_name, **model_config)
    model = model.to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Model info: {model.get_model_info()}")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(
        model, cfg.training.optimizer, 
        cfg.training.learning_rate, 
        cfg.training.weight_decay
    )
    
    scheduler = get_scheduler(
        optimizer, cfg.training.scheduler, 
        epochs=cfg.training.epochs
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
        monitor=cfg.training.early_stopping.monitor
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"Starting training for {cfg.training.epochs} epochs...")
    
    for epoch in range(cfg.training.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            cfg.training.loss_type, cfg.training.gradient_clip_norm,
            debug_mode=cfg.training.debug_mode
        )
        
        # Validate (skip if configured)
        if not cfg.training.skip_validation and epoch % cfg.training.validate_every_n_epochs == 0:
            val_metrics = validate_epoch(model, val_loader, device, cfg.training.loss_type)
        else:
            val_metrics = {'loss': 0.0, 'mse': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
        
        epoch_time = time.time() - epoch_start_time
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau) and epoch % cfg.training.validate_every_n_epochs == 0:
                scheduler.step(val_metrics['loss'])
            elif not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
        
        # Logging
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_mse': train_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'train_r2': train_metrics['r2'],
            'val_loss': val_metrics['loss'],
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | "
                  f"Train Loss: {train_metrics['loss']:.6f} | "
                  f"Val Loss: {val_metrics['loss']:.6f} | "
                  f"Val R²: {val_metrics['r2']:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Log to wandb
        if cfg.logging.use_wandb and epoch % cfg.logging.log_every_n_steps == 0:
            wandb.log(metrics)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss and cfg.training.save_best_model:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': cfg,
                'scalers': metadata['scalers']
            }, output_dir / "best_model.pth")
        
        # Save checkpoint
        if epoch % cfg.training.checkpoint_every_n_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': cfg,
                'scalers': metadata['scalers']
            }, output_dir / f"checkpoint_epoch_{epoch}.pth")
        
        # Early stopping (only check when we actually computed validation metrics)
        if epoch % cfg.training.validate_every_n_epochs == 0:
            # Create metrics dict with the key that early stopping expects
            early_stop_metrics = {'val_loss': val_metrics['loss']}
            if early_stopping(early_stop_metrics):
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Save final model
    if cfg.training.save_last_model:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'config': cfg,
            'scalers': metadata['scalers']
        }, output_dir / "last_model.pth")
    
    # Final evaluation on test set (skip if configured)
    if not cfg.training.skip_testing:
        print("Evaluating on test set...")
        test_metrics = validate_epoch(model, test_loader, device, cfg.training.loss_type)
        
        print(f"Test Results:")
        print(f"  Loss: {test_metrics['loss']:.6f}")
        print(f"  MSE: {test_metrics['mse']:.6f}")
        print(f"  MAE: {test_metrics['mae']:.6f}")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        print(f"  R²: {test_metrics['r2']:.4f}")
        
        # Log final test metrics
        if cfg.logging.use_wandb:
            wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
    else:
        print("Skipping test evaluation (overfitting mode)")
    
    if cfg.logging.use_wandb:
        wandb.finish()
    
    print(f"Training completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()