
import sys
import os
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from active_learning.src.config import load_config, get_bounds_from_config, DEVICE

def test_bounds_toggle():
    with open("verification_output.txt", "w") as f:
        def log(msg):
            print(msg)
            f.write(msg + "\n")
            
        log("Testing anatomical bounds toggle...")
        
        # 1. Load default config
        config = load_config()
        
        # 2. Check default behavior (should be enabled)
        log("\nTest Case 1: Default config (use_anatomical_bounds=True)")
        if 'bald' not in config: config['bald'] = {}
        config['bald']['use_anatomical_bounds'] = True
        
        bounds_default = get_bounds_from_config(config, DEVICE)
        log(f"Bounds shape: {bounds_default.shape}")
        log(f"Bounds:\n{bounds_default.cpu().numpy()}")
        
        # Check if they look like anatomical limits (not just -pi, pi)
        # Note: Depending on the actual anatomical limits, they might be different.
        # But wide limits are distinctly [-pi, pi]
        is_wide_default = torch.allclose(bounds_default[:, 0], torch.tensor(-np.pi, device=DEVICE)) and \
                          torch.allclose(bounds_default[:, 1], torch.tensor(np.pi, device=DEVICE))
        
        if not is_wide_default:
            log("PASS: Default bounds are NOT just [-pi, pi] (likely anatomical limits).")
        else:
            log("FAIL: Default bounds seem to be wide [-pi, pi] unexpectedly.")
            
        # 3. Check disabled behavior
        log("\nTest Case 2: Disabled anatomical bounds (use_anatomical_bounds=False)")
        config['bald']['use_anatomical_bounds'] = False
        
        bounds_wide = get_bounds_from_config(config, DEVICE)
        log(f"Bounds:\n{bounds_wide.cpu().numpy()}")
        
        is_wide = torch.allclose(bounds_wide[:, 0], torch.tensor(-np.pi, device=DEVICE)) and \
                  torch.allclose(bounds_wide[:, 1], torch.tensor(np.pi, device=DEVICE))
                  
        if is_wide:
            log("PASS: Bounds are wide [-pi, pi] as expected.")
        else:
            log("FAIL: Bounds are NOT wide [-pi, pi].")
            
        # 4. Check robustness (missing key)
        log("\nTest Case 3: Missing config key (should contain default True)")
        if 'use_anatomical_bounds' in config['bald']:
            del config['bald']['use_anatomical_bounds']
            
        bounds_missing = get_bounds_from_config(config, DEVICE)
        if torch.equal(bounds_missing, bounds_default):
             log("PASS: Missing key defaults to anatomical limits.")
        else:
             log("FAIL: Missing key did not default to anatomical limits.")

if __name__ == "__main__":
    test_bounds_toggle()
