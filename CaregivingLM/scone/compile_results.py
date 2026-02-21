#!/usr/bin/env python3
"""
Compile results from evaluation_results.json files.
Groups data by model and creates .txt files with target poses and distances.
Each line format: <x y z dist>
"""

import os
import re
import json
import argparse
import shutil
from collections import defaultdict


def extract_model_from_folder_name(folder_name):
    """Extract model name from folder like 'single_arm_c0_random_1_20250827...'"""
    # Look for the pattern single_arm_cX or single_arm_cX_Y
    match = re.match(r'^(condition_\d+(?:_\d+)?)', folder_name)
    return match.group(1) if match else "unknown"


def extract_muscles_from_hfd(hfd_file_path):
    """Extract all muscle names from .hfd file"""
    muscles = []
    
    try:
        with open(hfd_file_path, 'r') as f:
            content = f.read()
            
        # Find all point_path_muscle entries
        muscle_pattern = r'point_path_muscle\s*{\s*name\s*=\s*([^\s\n]+)'
        matches = re.findall(muscle_pattern, content)
        
        for match in matches:
            muscle_name = match.strip()
            if muscle_name:
                muscles.append(muscle_name)
                
    except Exception as e:
        print(f"Warning: Could not read HFD file {hfd_file_path}: {e}")
    
    return sorted(list(set(muscles)))  # Remove duplicates and sort


def parse_config_scone(config_file_path):
    """Extract muscle max_isometric_force.factor values from config.scone file"""
    muscle_factors = {}
    
    try:
        with open(config_file_path, 'r') as f:
            content = f.read()
        
        # First check if Properties is empty (Properties = "")
        empty_properties = re.search(r'Properties\s*=\s*["\']["\']', content)
        if empty_properties:
            return muscle_factors  # Return empty dict
        
        # Find the Properties section with nested braces
        # Use a more robust approach to handle nested braces
        properties_start = re.search(r'Properties\s*{', content)
        if properties_start:
            start_pos = properties_start.end() - 1  # Include the opening brace
            brace_count = 0
            end_pos = start_pos
            
            for i, char in enumerate(content[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            
            if brace_count == 0:
                properties_section = content[properties_start.end():end_pos]
                
                # Find individual muscle entries
                # Pattern: muscle_name { max_isometric_force.factor = value }
                muscle_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*{\s*max_isometric_force\.factor\s*=\s*([0-9.eE+-]+)\s*}'
                matches = re.findall(muscle_pattern, properties_section, re.MULTILINE | re.DOTALL)
            else:
                matches = []
        else:
            matches = []
        
        for muscle_name, factor_str in matches:
            try:
                factor = float(factor_str)
                muscle_factors[muscle_name.strip()] = factor
            except ValueError:
                print(f"Warning: Could not parse factor '{factor_str}' for muscle '{muscle_name}'")
        
    except Exception as e:
        print(f"Warning: Could not parse config file {config_file_path}: {e}")
    
    return muscle_factors


def create_complete_muscle_params(all_muscles, muscle_factors):
    """Create complete muscle parameter dict with defaults for missing muscles"""
    complete_params = {}

    for muscle in all_muscles:
        if muscle in muscle_factors:
            complete_params[muscle] = muscle_factors[muscle]
        else:
            complete_params[muscle] = 1.0  # Default factor

    return complete_params


def extract_hand_trajectory(sto_file_path):
    """Extract hand_r.pos.x, hand_r.pos.y, hand_r.pos.z trajectory from .sto file"""
    trajectory_points = []

    try:
        with open(sto_file_path, 'r') as f:
            lines = f.readlines()

        # Find the header line to locate hand_r.pos columns
        header_idx = -1
        hand_x_col = -1
        hand_y_col = -1
        hand_z_col = -1

        for i, line in enumerate(lines):
            if line.startswith('time\t') or 'hand_r.pos.x' in line:
                header_idx = i
                columns = line.strip().split('\t')
                for j, col in enumerate(columns):
                    if col == 'hand_r.pos.x':
                        hand_x_col = j
                    elif col == 'hand_r.pos.y':
                        hand_y_col = j
                    elif col == 'hand_r.pos.z':
                        hand_z_col = j
                break

        # Extract trajectory data
        if header_idx >= 0 and hand_x_col >= 0 and hand_y_col >= 0 and hand_z_col >= 0:
            for line in lines[header_idx+1:]:
                line = line.strip()
                if line and not line.startswith('#'):
                    data = line.split('\t')
                    if len(data) > max(hand_x_col, hand_y_col, hand_z_col):
                        try:
                            x = float(data[hand_x_col])
                            y = float(data[hand_y_col])
                            z = float(data[hand_z_col])
                            trajectory_points.append((x, y, z))
                        except (ValueError, IndexError):
                            continue

    except Exception as e:
        print(f"Warning: Could not extract hand trajectory from {sto_file_path}: {e}")

    return trajectory_points


def collect_sto_files(model_data, distance_threshold, output_dir):
    """
    Collect best .sto files for each condition that meet the distance threshold.
    For each target position, only keep the best (lowest distance) result.
    """
    print(f"Distance threshold: {distance_threshold:.6f}")

    # Group results by condition and target position
    condition_targets = defaultdict(lambda: defaultdict(list))  # condition -> target_key -> [results]

    for condition, results in model_data.items():
        for result in results:
            if result["has_sto"] and result["distance"] <= distance_threshold:
                # Create a target key based on position (rounded to avoid floating point issues)
                target_key = f"{result['x']:.6f},{result['y']:.6f},{result['z']:.6f}"
                condition_targets[condition][target_key].append(result)

    # Process each condition
    total_collected = 0
    for condition in sorted(condition_targets.keys()):
        targets = condition_targets[condition]
        if not targets:
            print(f"{condition:<20} No results under threshold")
            continue

        # Create condition directory
        condition_dir = os.path.join(output_dir, f"{condition}_sto_files")
        os.makedirs(condition_dir, exist_ok=True)

        condition_collected = 0
        for target_key, target_results in targets.items():
            # Find the best result for this target (lowest distance)
            best_result = min(target_results, key=lambda x: x["distance"])

            # Copy the .sto file
            sto_filename = os.path.basename(best_result["sto_file_path"])
            dest_path = os.path.join(condition_dir, f"target_{target_key.replace(',', '_')}_{sto_filename}")

            try:
                shutil.copy2(best_result["sto_file_path"], dest_path)
                condition_collected += 1
                total_collected += 1
            except Exception as e:
                print(f"  Error copying {sto_filename}: {e}")

        print(f"{condition:<20} {condition_collected} .sto files collected to {condition}_sto_files/")

    print(f"\nTotal .sto files collected: {total_collected}")
    if total_collected > 0:
        print(f"Files organized in subdirectories under: {output_dir}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compile SCONE evaluation results by model')
    parser.add_argument('--results-dir', type=str, 
                       default='/home/ziang/Workspace/CaregivingLM/scone/results',
                       help='Path to results directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/ziang/Workspace/CaregivingLM/scone/compiled_results',
                       help='Directory to save compiled .txt files')
    parser.add_argument('--collect-sto', type=float, metavar='THRESHOLD',
                       help='Collect best .sto files for each target with distance < THRESHOLD')
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    output_dir = args.output_dir
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Data collection by model
    model_data = defaultdict(list)
    model_muscle_params = defaultdict(list)  # Store muscle params for consistency checking
    model_sto_counts = defaultdict(int)  # Count .sto files for each model
    model_trajectories = defaultdict(list)  # Store hand trajectories for each model
    
    # Expected models
    # expected_models = [
    #     "single_arm_c0", "single_arm_c1", "single_arm_c2", "single_arm_c3", 
    #     "single_arm_c4", "single_arm_c5", "single_arm_c6_1", "single_arm_c6_2", "single_arm_c6_3"
    # ]

    expected_models = [
        "condition_1"
    ]
    
    # Track HFD files to ensure consistency
    hfd_files_found = set()
    hfd_file_path = None
    all_muscles = []
    
    print(f"Scanning results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Process each folder
    total_folders_processed = 0
    total_valid_results = 0
    
    for folder_name in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
            
        # Skip if doesn't match pattern
        if not folder_name.startswith("condition_"):
            continue
            
        total_folders_processed += 1
        
        # Extract model name
        model = extract_model_from_folder_name(folder_name)
        
        # Check for evaluation_results.json and config.scone
        eval_file = os.path.join(folder_path, "evaluation_results.json")
        config_file = os.path.join(folder_path, "config.scone")
        
        # Extract and track HFD file paths
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_content = f.read()
                hfd_match = re.search(r'model_file\s*=\s*["\']([^"\']+)["\']', config_content)
                if hfd_match:
                    hfd_relative_path = hfd_match.group(1)
                    
                    # First check if the HFD file exists in the same folder as config.scone (local copy)
                    hfd_filename = os.path.basename(hfd_relative_path)
                    local_hfd_path = os.path.join(folder_path, hfd_filename)
                    
                    if os.path.exists(local_hfd_path):
                        # Use the local copy in the results folder
                        hfd_full_path = local_hfd_path
                    else:
                        # Resolve the relative path from the config.scone location
                        hfd_full_path = os.path.normpath(os.path.join(folder_path, hfd_relative_path))
                        
                        # If still not found, try resolving from the scone directory root
                        if not os.path.exists(hfd_full_path):
                            # Assume we're in scone/ directory, resolve relative to that
                            scone_root = os.path.dirname(results_dir)  # Go up from results/ to scone/
                            hfd_full_path = os.path.normpath(os.path.join(scone_root, hfd_relative_path.lstrip('../')))
                    
                    # Track all HFD files found
                    hfd_files_found.add(hfd_full_path)
                    
                    # Set the first valid HFD file we find
                    if hfd_file_path is None and os.path.exists(hfd_full_path):
                        hfd_file_path = hfd_full_path
            except Exception as e:
                print(f"Warning: Could not extract HFD path from {config_file}: {e}")
        
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)

                # Extract muscle parameters from config.scone
                muscle_factors = {}
                if os.path.exists(config_file):
                    muscle_factors = parse_config_scone(config_file)
                    model_muscle_params[model].append(muscle_factors)

                # Check for .sto analysis file and extract hand trajectory
                has_sto_file = False
                sto_file_path = None
                if "best_par_file" in eval_data:
                    par_filename = eval_data["best_par_file"]
                    sto_filename = par_filename + ".sto"
                    sto_file_path = os.path.join(folder_path, sto_filename)
                    if os.path.exists(sto_file_path):
                        has_sto_file = True
                        model_sto_counts[model] += 1

                        # Extract hand trajectory
                        trajectory_points = extract_hand_trajectory(sto_file_path)
                        if trajectory_points:
                            model_trajectories[model].extend(trajectory_points)

                # Extract required data
                if "target_pose" in eval_data and "distance" in eval_data:
                    target_pose = eval_data["target_pose"]
                    distance = eval_data["distance"]

                    if all(key in target_pose for key in ["x", "y", "z"]):
                        x = target_pose["x"]
                        y = target_pose["y"]
                        z = target_pose["z"]

                        # Add to model data
                        model_data[model].append({
                            "x": x,
                            "y": y,
                            "z": z,
                            "distance": distance,
                            "folder": folder_name,
                            "folder_path": folder_path,
                            "has_sto": has_sto_file,
                            "sto_file_path": sto_file_path if has_sto_file else None
                        })

                        total_valid_results += 1
                        muscle_info = f"muscles: {len(muscle_factors)}" if muscle_factors else "no muscles"
                        sto_info = "sto: yes" if has_sto_file else "sto: no"
                        print(f"  {folder_name[:50]:<50} -> {model} (x={x:.3f}, y={y:.3f}, z={z:.3f}, dist={distance:.6f}, {muscle_info}, {sto_info})")
                    else:
                        print(f"  {folder_name[:50]:<50} -> Missing target pose coordinates")
                else:
                    print(f"  {folder_name[:50]:<50} -> Missing target_pose or distance")
                    
            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"  {folder_name[:50]:<50} -> Error reading evaluation file: {e}")
        else:
            print(f"  {folder_name[:50]:<50} -> No evaluation_results.json")
    
    print("=" * 60)
    print("HFD FILE ANALYSIS")
    print("=" * 60)
    
    # Check HFD file consistency
    if len(hfd_files_found) == 0:
        print("No HFD files found in any config.scone files")
    elif len(hfd_files_found) == 1:
        hfd_file_path = list(hfd_files_found)[0]
        print(f"All runs use the same HFD file: {hfd_file_path}")
        if os.path.exists(hfd_file_path):
            print("Extracting muscles from HFD file...")
            all_muscles = extract_muscles_from_hfd(hfd_file_path)
            print(f"Found {len(all_muscles)} muscles in HFD file")
        else:
            print(f"ERROR: HFD file does not exist: {hfd_file_path}")
    else:
        print(f"WARNING: Found {len(hfd_files_found)} different HFD files:")
        for hfd_file in sorted(hfd_files_found):
            print(f"  {hfd_file}")
        print("Using the first valid HFD file found")
        if hfd_file_path and os.path.exists(hfd_file_path):
            print(f"Extracting muscles from: {hfd_file_path}")
            all_muscles = extract_muscles_from_hfd(hfd_file_path)
            print(f"Found {len(all_muscles)} muscles in HFD file")

    print()
    print("=" * 60)
    print("COMPILATION SUMMARY")
    print("=" * 60)
    print(f"Total folders processed: {total_folders_processed}")
    print(f"Total valid results: {total_valid_results}")
    print()
    
    # Write data files for each model
    print("WRITING MODEL DATA FILES")
    print("=" * 60)
    
    # Show stats for expected models first, then any others found
    all_models = set(expected_models + list(model_data.keys()))
    
    for model in sorted(all_models):
        data_points = model_data[model]
        
        if data_points:
            output_file = os.path.join(output_dir, f"{model}_results.txt")
            
            # Keep original order (no sorting)
            
            with open(output_file, 'w') as f:
                # Write header comment
                f.write(f"# {model} evaluation results\n")
                f.write(f"# Format: x y z distance\n")
                f.write(f"# Total data points: {len(data_points)}\n")
                f.write("#\n")
                
                # Write data points
                for point in data_points:
                    f.write(f"{point['x']:.6f} {point['y']:.6f} {point['z']:.6f} {point['distance']:.6f}\n")
            
            # Calculate statistics
            distances = [p["distance"] for p in data_points]
            min_dist = min(distances)
            max_dist = max(distances)
            avg_dist = sum(distances) / len(distances)

            sto_count = model_sto_counts[model]
            print(f"{model:<20} {len(data_points):<8} points  Min: {min_dist:.6f}  Max: {max_dist:.6f}  Avg: {avg_dist:.6f}  STO: {sto_count}/{len(data_points)}")
        else:
            sto_count = model_sto_counts[model]
            print(f"{model:<20} {'0':<8} points  (no data)  STO: {sto_count}/0")
    
    print()
    print("PROCESSING MUSCLE PARAMETERS")
    print("=" * 60)

    if all_muscles and model_muscle_params:
        print(f"Creating muscle parameter files for {len(all_muscles)} muscles")
        
        for model in sorted(model_muscle_params.keys()):
            muscle_param_sets = model_muscle_params[model]
            
            if muscle_param_sets:
                # Sanity check: all parameter sets for the same model should be identical
                first_params = muscle_param_sets[0]
                all_identical = True
                
                for params in muscle_param_sets[1:]:
                    if params != first_params:
                        all_identical = False
                        break
                
                if not all_identical:
                    print(f"WARNING: {model} has inconsistent muscle parameters across runs!")
                    print("  Using parameters from first run found")
                
                # Create complete muscle parameter dict with defaults
                complete_muscle_params = create_complete_muscle_params(all_muscles, first_params)
                
                # Write JSON file
                muscle_json_file = os.path.join(output_dir, f"{model}_muscle_params.json")
                with open(muscle_json_file, 'w') as f:
                    json.dump(complete_muscle_params, f, indent=2, sort_keys=True)
                
                # Count non-default parameters
                non_default_count = sum(1 for factor in complete_muscle_params.values() if factor != 1.0)
                print(f"{model:<20} {len(complete_muscle_params)} muscles  ({non_default_count} modified, {len(complete_muscle_params) - non_default_count} default)")
            
            else:
                print(f"{model:<20} No muscle parameter data found")
    
    else:
        print("No muscle parameter data to process (missing HFD file or config files)")

    print()
    print("PROCESSING HAND TRAJECTORIES")
    print("=" * 60)

    if model_trajectories:
        print(f"Creating trajectory files for hand movements")

        for model in sorted(model_trajectories.keys()):
            trajectory_points = model_trajectories[model]

            if trajectory_points:
                # Write trajectory file
                trajectory_file = os.path.join(output_dir, f"{model}_hand_trajectories.txt")
                with open(trajectory_file, 'w') as f:
                    # Write header comment
                    f.write(f"# {model} hand trajectory data\n")
                    f.write(f"# Format: x y z\n")
                    f.write(f"# Total trajectory points: {len(trajectory_points)}\n")
                    f.write("#\n")

                    # Write trajectory points
                    for x, y, z in trajectory_points:
                        f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

                print(f"{model:<20} {len(trajectory_points):<8} trajectory points")
            else:
                print(f"{model:<20} {'0':<8} trajectory points (no data)")
    else:
        print("No hand trajectory data to process")

    print()
    print("=" * 60)
    print("COMPILATION COMPLETE")
    print("=" * 60)
    
    if total_valid_results > 0:
        print(f"Data files saved to: {output_dir}")
        print("Files created:")
        for model in sorted(model_data.keys()):
            if model_data[model]:
                print(f"  {model}_results.txt ({len(model_data[model])} data points)")
        
        if all_muscles and model_muscle_params:
            print("Muscle parameter files:")
            for model in sorted(model_muscle_params.keys()):
                if model_muscle_params[model]:
                    print(f"  {model}_muscle_params.json")

        if model_trajectories:
            print("Hand trajectory files:")
            for model in sorted(model_trajectories.keys()):
                if model_trajectories[model]:
                    print(f"  {model}_hand_trajectories.txt")
    else:
        print("No valid results found to compile.")

    # Handle .sto file collection if requested
    if args.collect_sto is not None:
        print()
        print("=" * 60)
        print("COLLECTING .STO FILES")
        print("=" * 60)

        collect_sto_files(model_data, args.collect_sto, output_dir)


if __name__ == "__main__":
    main()