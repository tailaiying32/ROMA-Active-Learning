#!/usr/bin/env python3
"""
Dual-encoder model: separate encoders for muscle parameters and pose, then concatenate.
"""

import torch
import torch.nn as nn
from .base import BaseReachPredictor


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, 
                 dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        
        # Choose activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Add activation and dropout for all layers except the last
            if i < len(dims) - 2:
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DualEncoderModel(BaseReachPredictor):
    """
    Dual-encoder architecture:
    - Muscle Encoder: MLP (40 → muscle_hidden → muscle_embed_dim)
    - Pose Encoder: MLP (3 → pose_hidden → pose_embed_dim)  
    - Predictor: Concatenate embeddings → MLP (muscle_embed_dim + pose_embed_dim → predictor_hidden → 1)
    """
    
    def __init__(self, 
                 num_muscles: int = 40,
                 pose_dim: int = 3,
                 muscle_hidden_dims: list = [64, 32],
                 muscle_embed_dim: int = 32,
                 pose_hidden_dims: list = [16],
                 pose_embed_dim: int = 8,
                 predictor_hidden_dims: list = [20],
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        
        super().__init__(num_muscles, pose_dim)
        
        self.muscle_embed_dim = muscle_embed_dim
        self.pose_embed_dim = pose_embed_dim
        
        # Muscle parameter encoder
        self.muscle_encoder = MLP(
            input_dim=num_muscles,
            hidden_dims=muscle_hidden_dims, 
            output_dim=muscle_embed_dim,
            dropout=dropout,
            activation=activation
        )
        
        # Target pose encoder
        self.pose_encoder = MLP(
            input_dim=pose_dim,
            hidden_dims=pose_hidden_dims,
            output_dim=pose_embed_dim, 
            dropout=dropout,
            activation=activation
        )
        
        # Distance predictor
        self.predictor = MLP(
            input_dim=muscle_embed_dim + pose_embed_dim,
            hidden_dims=predictor_hidden_dims,
            output_dim=1,
            dropout=dropout,
            activation=activation
        )
        
    def forward(self, muscle_params: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            muscle_params: (batch_size, num_muscles) 
            target_pose: (batch_size, 3)
            
        Returns:
            predicted_distance: (batch_size,)
        """
        # Encode inputs
        muscle_embed = self.muscle_encoder(muscle_params)  # (batch_size, muscle_embed_dim)
        pose_embed = self.pose_encoder(target_pose)        # (batch_size, pose_embed_dim)
        
        # Concatenate embeddings
        combined_embed = torch.cat([muscle_embed, pose_embed], dim=1)  # (batch_size, muscle_embed_dim + pose_embed_dim)
        
        # Predict distance
        distance = self.predictor(combined_embed).squeeze(-1)  # (batch_size,)
        
        return distance
    
    def get_embeddings(self, muscle_params: torch.Tensor, target_pose: torch.Tensor):
        """Get intermediate embeddings for analysis."""
        with torch.no_grad():
            muscle_embed = self.muscle_encoder(muscle_params)
            pose_embed = self.pose_encoder(target_pose)
            return {
                'muscle_embed': muscle_embed,
                'pose_embed': pose_embed
            }
    
    def get_model_info(self):
        """Get extended model information."""
        info = super().get_model_info()
        info.update({
            'muscle_embed_dim': self.muscle_embed_dim,
            'pose_embed_dim': self.pose_embed_dim,
            'muscle_encoder_params': sum(p.numel() for p in self.muscle_encoder.parameters()),
            'pose_encoder_params': sum(p.numel() for p in self.pose_encoder.parameters()),
            'predictor_params': sum(p.numel() for p in self.predictor.parameters()),
        })
        return info


if __name__ == "__main__":
    # Test the model
    model = DualEncoderModel()
    
    batch_size = 16
    muscle_params = torch.randn(batch_size, 40)
    target_pose = torch.randn(batch_size, 3)
    
    output = model(muscle_params, target_pose)
    print(f"Output shape: {output.shape}")
    print(f"Model info: {model.get_model_info()}")
    
    # Test embeddings
    embeddings = model.get_embeddings(muscle_params, target_pose)
    print(f"Muscle embedding shape: {embeddings['muscle_embed'].shape}")
    print(f"Pose embedding shape: {embeddings['pose_embed'].shape}")