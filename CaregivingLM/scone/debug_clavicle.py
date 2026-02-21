#!/usr/bin/env python3
"""Debug script to trace clavicle_protraction_r sampling."""

import numpy as np

def debug_clavicle_sampling():
    """Debug the sampling for clavicle_protraction_r specifically."""

    # Parameters from the script
    deg_to_rad = np.pi / 180.0
    base_min = -30 * deg_to_rad
    base_max = 10 * deg_to_rad
    base_range = base_max - base_min

    coverage_factor = 1.0  # User set this
    min_range_factor = 0.05  # User set this

    min_range = min_range_factor * base_range
    effective_range = coverage_factor * base_range
    effective_min = base_min + (base_range - effective_range) / 2
    effective_max = effective_min + effective_range

    print(f"=== CLAVICLE PROTRACTION DEBUG ===")
    print(f"Base range: {base_min*180/np.pi:.1f}° to {base_max*180/np.pi:.1f}°")
    print(f"Base range span: {base_range*180/np.pi:.1f}°")
    print(f"Min range: {min_range*180/np.pi:.1f}°")
    print(f"Effective range: {effective_range*180/np.pi:.1f}°")
    print(f"Effective min: {effective_min*180/np.pi:.1f}°")
    print(f"Effective max: {effective_max*180/np.pi:.1f}°")
    print()

    # Strategy 1: Lower first
    print("=== STRATEGY 1: LOWER FIRST ===")
    lower_max_possible = effective_max - min_range
    print(f"Lower can range from: {effective_min*180/np.pi:.1f}° to {lower_max_possible*180/np.pi:.1f}°")
    print(f"Lower range span: {(lower_max_possible - effective_min)*180/np.pi:.1f}°")

    # Test with extreme values
    for u_val, u_name in [(0.0, "u=0"), (1.0, "u=1")]:
        if lower_max_possible > effective_min:
            lower_limit = effective_min + u_val * (lower_max_possible - effective_min)
        else:
            lower_limit = effective_min

        upper_min_possible = lower_limit + min_range
        print(f"  {u_name}: lower={lower_limit*180/np.pi:.1f}°, upper_min={upper_min_possible*180/np.pi:.1f}°, upper_max={effective_max*180/np.pi:.1f}°")
    print()

    # Strategy 2: Upper first
    print("=== STRATEGY 2: UPPER FIRST ===")
    upper_min_possible = effective_min + min_range
    print(f"Upper can range from: {upper_min_possible*180/np.pi:.1f}° to {effective_max*180/np.pi:.1f}°")
    print(f"Upper range span: {(effective_max - upper_min_possible)*180/np.pi:.1f}°")

    # Test with extreme values
    for u_val, u_name in [(0.0, "u=0"), (1.0, "u=1")]:
        if effective_max > upper_min_possible:
            upper_limit = upper_min_possible + u_val * (effective_max - upper_min_possible)
        else:
            upper_limit = effective_max

        lower_max_possible = upper_limit - min_range
        print(f"  {u_name}: upper={upper_limit*180/np.pi:.1f}°, lower_min={effective_min*180/np.pi:.1f}°, lower_max={lower_max_possible*180/np.pi:.1f}°")

if __name__ == "__main__":
    debug_clavicle_sampling()