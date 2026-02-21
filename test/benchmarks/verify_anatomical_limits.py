
import sys
import os
import torch
import numpy as np
import yaml

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, DEVICE, get_bounds_from_config
from infer_params.training.model import LevelSetDecoder
from infer_params.training.dataset import LevelSetDataset

def verify_limits():
    print("=== Verifying Anatomical Limits Mismatch ===")
    
    # 1. Load Config
    config_path = os.path.join(os.path.dirname(__file__), '../configs/default.yaml')
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    bounds = get_bounds_from_config(config, DEVICE)
    bounds_np = bounds.cpu().numpy()
    
    print("\nConfigured Anatomical Limits:")
    joint_names = config['prior']['joint_names']
    for i, name in enumerate(joint_names):
        print(f"  {name}: [{bounds_np[i,0]:.4f}, {bounds_np[i,1]:.4f}] (rad)")
        print(f"           [{np.rad2deg(bounds_np[i,0]):.1f}, {np.rad2deg(bounds_np[i,1]):.1f}] (deg)")

    # 2. Load Decoder
    model_path = os.path.join(os.path.dirname(__file__), '../../models/best_model.pt')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"\nLoading decoder from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    train_config = checkpoint['config']
    model_cfg = train_config['model']
    embeddings = checkpoint['embeddings']
    
    decoder = LevelSetDecoder(
        num_samples=embeddings.shape[0],
        latent_dim=model_cfg['latent_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        num_blocks=model_cfg['num_blocks'],
        num_slots=model_cfg.get('num_slots', 18),
        params_per_slot=model_cfg.get('params_per_slot', 6),
    ).to(DEVICE)
    decoder.load_state_dict(checkpoint['model_state_dict'])
    decoder.eval()
    
    # 3. Decode all embeddings
    print(f"\nDecoding {embeddings.shape[0]} samples...")
    batch_size = 100
    num_samples = embeddings.shape[0]
    
    all_lower = []
    all_upper = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_emb = embeddings[i:min(i+batch_size, num_samples)].to(DEVICE)
            lower, upper, _, _, _ = decoder.decode_from_embedding(batch_emb)
            all_lower.append(lower.cpu().numpy())
            all_upper.append(upper.cpu().numpy())
            
    all_lower = np.concatenate(all_lower, axis=0) # (N, 4)
    all_upper = np.concatenate(all_upper, axis=0) # (N, 4)
    
    # 4. Check Violations
    print(f"\nChecking violations against configured limits...")
    
    total_samples = num_samples
    any_violation_count = 0
    
    for i, name in enumerate(joint_names):
        limit_lower = bounds_np[i, 0]
        limit_upper = bounds_np[i, 1]
        
        # Check lower bound violations (decoder lower < limit lower)
        # Note: We want decoder's lower to be INSIDE the limits? 
        # Actually, if decoder lower < limit lower, it means the feasible range extends outside the limit to the left.
        # So yes, that's a violation if we consider limits as "hard physical limits".
        
        # Violations:
        # 1. Decoder lower is BELOW config lower
        v_lower = all_lower[:, i] < limit_lower
        # 2. Decoder upper is ABOVE config upper
        v_upper = all_upper[:, i] > limit_upper
        
        n_v_lower = np.sum(v_lower)
        n_v_upper = np.sum(v_upper)
        
        max_v_lower = np.min(all_lower[:, i]) - limit_lower if n_v_lower > 0 else 0
        max_v_upper = np.max(all_upper[:, i]) - limit_upper if n_v_upper > 0 else 0
        
        print(f"\nJoint: {name}")
        print(f"  Lower Violations: {n_v_lower}/{total_samples} ({n_v_lower/total_samples*100:.1f}%)")
        if n_v_lower > 0:
             print(f"    Max violation: {np.rad2deg(max_v_lower):.1f} deg (Decoder min: {np.rad2deg(np.min(all_lower[:, i])):.1f})")
             
        print(f"  Upper Violations: {n_v_upper}/{total_samples} ({n_v_upper/total_samples*100:.1f}%)")
        if n_v_upper > 0:
             print(f"    Max violation: {np.rad2deg(max_v_upper):.1f} deg (Decoder max: {np.rad2deg(np.max(all_upper[:, i])):.1f})")

        any_violation_count += np.sum(v_lower | v_upper)

    print(f"\nTotal samples with at least one violation: {any_violation_count} (Note: may double count if multiple joints violated)")
    # Correct calculation
    any_violation_mask = np.zeros(total_samples, dtype=bool)
    for i in range(len(joint_names)):
        limit_lower = bounds_np[i, 0]
        limit_upper = bounds_np[i, 1]
        any_violation_mask |= (all_lower[:, i] < limit_lower)
        any_violation_mask |= (all_upper[:, i] > limit_upper)
        
    print(f"Unique samples with violations: {np.sum(any_violation_mask)}/{total_samples} ({np.sum(any_violation_mask)/total_samples*100:.1f}%)")
    
    print("\n" + "="*60)
    print("SUGGESTED LIMITS (based on decoder min/max over dataset)")
    print("="*60)
    print("anatomical_limits:")
    units = config.get('prior', {}).get('units', 'degrees')
    
    for i, name in enumerate(joint_names):
        # Calculate min/max from decoder outputs
        # Add a small buffer (e.g. 5 degrees)
        dec_min = np.min(all_lower[:, i])
        dec_max = np.max(all_upper[:, i])
        
        if units == 'degrees':
            dec_min_val = np.rad2deg(dec_min)
            dec_max_val = np.rad2deg(dec_max)
            # Round to nearest 5
            suggested_min = np.floor(dec_min_val / 5) * 5
            suggested_max = np.ceil(dec_max_val / 5) * 5
        else:
            suggested_min = dec_min
            suggested_max = dec_max
            
        print(f"    {name}: [{suggested_min:.0f}, {suggested_max:.0f}]")


if __name__ == "__main__":
    verify_limits()
