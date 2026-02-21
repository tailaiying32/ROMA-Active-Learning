#!/usr/bin/env python3
"""
Base class for reach prediction models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseReachPredictor(nn.Module, ABC):
    """Base class for all reach prediction models."""
    
    def __init__(self, num_muscles: int = 40, pose_dim: int = 3):
        super().__init__()
        self.num_muscles = num_muscles
        self.pose_dim = pose_dim
        
    @abstractmethod
    def forward(self, muscle_params: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            muscle_params: (batch_size, num_muscles) muscle parameter values
            target_pose: (batch_size, 3) target pose coordinates
            
        Returns:
            predicted_distance: (batch_size,) predicted reach distances
        """
        pass
    
    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with batch dictionary format."""
        return self.forward(batch['muscle_params'], batch['target_pose'])
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    loss_type: str = 'mse') -> torch.Tensor:
        """Compute loss between predictions and targets."""
        if loss_type == 'mse':
            return nn.MSELoss()(predictions, targets)
        elif loss_type == 'mae':
            return nn.L1Loss()(predictions, targets)
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            'model_class': self.__class__.__name__,
            'num_muscles': self.num_muscles,
            'pose_dim': self.pose_dim,
            'num_parameters': self.get_num_parameters(),
        }