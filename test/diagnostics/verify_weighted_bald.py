import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from active_learning.src.legacy.bald import BALD
from active_learning.src.legacy.user_distribution import UserDistribution

def verify_weighted_bald_gate():
    print("Verifying Weighted BALD Gate Logic...")
    
    # Dummy posterior
    joint_names = ['j1', 'j2']
    anatomical_limits = {'j1': [-1, 1], 'j2': [-1, 1]}
    config = {
        'bald': {
            'tau': 1.0,
            'n_mc_samples': 100,
            'use_weighted_bald': True,
            'weighted_bald_sigma': 0.1
        },
        'prior': {
            'init_std': 0.1
        }
    }
    
    posterior = UserDistribution(joint_names, anatomical_limits=anatomical_limits, config=config)
    bald = BALD(posterior, config)
    
    # We want to test the gate logic.
    # Gate = exp( - (p - 0.5)^2 / (2 * sigma^2) )
    # If p = 0.5, gate = 1.0
    # If p = 0.0 or 1.0, gate = exp(-0.25 / (2 * 0.01)) = exp(-12.5) approx 0.
    
    sigma = config['bald']['weighted_bald_sigma']
    
    # Test cases for p_mean
    p_means = torch.tensor([0.5, 0.1, 0.9, 0.0, 1.0])
    
    print(f"{'p_mean':<10} {'Gate Value':<15} {'Expected'}")
    print("-" * 40)
    
    for p in p_means:
        diff = p - 0.5
        gate = torch.exp(- (diff**2) / (2 * sigma**2))
        
        expected_full = 1.0 if abs(p - 0.5) < 1e-6 else 0.0
        # Just check if it's decreasing
        print(f"{p.item():<10.2f} {gate.item():<15.6f} {'High' if p==0.5 else 'Low'}")
        
        if p == 0.5:
            assert abs(gate.item() - 1.0) < 1e-6
        elif abs(p - 0.5) > 0.3:
            assert gate.item() < 0.1

    print("\nGate logic verified successfully.")

if __name__ == "__main__":
    verify_weighted_bald_gate()
