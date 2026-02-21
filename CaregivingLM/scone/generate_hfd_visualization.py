#!/usr/bin/env python3
"""
Generate SCONE .hfd model files for 3D visualization of target positions.
Creates small spheres at each target position from the compiled data files.
"""

import os
import argparse
import colorsys


def read_data_file(data_file):
    """Read target positions and distances from data file"""
    data_points = []
    
    with open(data_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            try:
                parts = line.split()
                if len(parts) != 4:
                    print(f"Warning: Line {line_num} has {len(parts)} values, expected 4. Skipping.")
                    continue
                    
                x, y, z, distance = map(float, parts)
                data_points.append({
                    'x': x, 'y': y, 'z': z, 
                    'distance': distance,
                    'index': len(data_points)
                })
                
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num}: '{line}'. Error: {e}")
                continue
    
    return data_points


def generate_color_from_distance(distance, threshold=0.1):
    """Generate RGB color with smooth green->yellow->red transition based on distance"""
    
    # Define color zones using the threshold as reference
    excellent_zone = threshold * 0.5    # Best performance: pure green
    good_zone = threshold               # Good performance: green to yellow transition
    poor_zone = threshold * 3.0         # Poor performance: yellow to red transition
    
    if distance <= excellent_zone:
        # Excellent: Pure green
        return 0.2, 0.8, 0.2
    
    elif distance <= good_zone:
        # Good: Green to yellow transition
        # Normalize within this zone (0 to 1)
        factor = (distance - excellent_zone) / (good_zone - excellent_zone)
        
        # Transition from green (0.2, 0.8, 0.2) to yellow (0.8, 0.8, 0.2)
        r = 0.2 + (0.8 - 0.2) * factor    # 0.2 -> 0.8
        g = 0.8                           # Stay at 0.8
        b = 0.2                           # Stay at 0.2
        
        return r, g, b
    
    elif distance <= poor_zone:
        # Poor: Yellow to red transition
        # Normalize within this zone (0 to 1)
        factor = (distance - good_zone) / (poor_zone - good_zone)
        
        # Transition from yellow (0.8, 0.8, 0.2) to red (0.8, 0.2, 0.2)
        r = 0.8                           # Stay at 0.8
        g = 0.8 - (0.8 - 0.2) * factor   # 0.8 -> 0.2
        b = 0.2                           # Stay at 0.2
        
        return r, g, b
    
    else:
        # Very poor: Dark red with intensity based on how far over
        # Cap at 10x threshold to avoid extreme values
        max_distance = threshold * 10
        clamped_distance = min(distance, max_distance)
        
        # Scale from red to darker red
        factor = (clamped_distance - poor_zone) / (max_distance - poor_zone)
        
        # Darken the red as distance increases
        r = 0.8 + 0.2 * factor    # 0.8 -> 1.0 (brighter red)
        g = 0.2 * (1.0 - factor)  # 0.2 -> 0.0 (less green)
        b = 0.2 * (1.0 - factor)  # 0.2 -> 0.0 (less blue)
        
        return r, g, b


def generate_target_bodies(data_points, model_name, distance_threshold=0.1, filter_threshold=None):
    """Generate target body definitions to insert into existing .hfd file"""
    
    if not data_points:
        raise ValueError("No data points to visualize")
    
    # Filter data points if filter_threshold is specified
    if filter_threshold is not None:
        original_count = len(data_points)
        data_points = [p for p in data_points if p['distance'] <= filter_threshold]
        filtered_count = len(data_points)
        print(f"Filtered {original_count - filtered_count} points with distance > {filter_threshold:.3f}")
        print(f"Showing {filtered_count} points with distance <= {filter_threshold:.3f}")
        
        if not data_points:
            print(f"WARNING: No data points remain after filtering with threshold {filter_threshold}")
            print("Proceeding to generate empty visualization file...")
    
    # Calculate distance statistics
    if data_points:
        distances = [p['distance'] for p in data_points]
        min_dist = min(distances)
        max_dist = max(distances)
        excellent_count = sum(1 for d in distances if d <= distance_threshold * 0.5)
        good_count = sum(1 for d in distances if d <= distance_threshold)
        poor_count = sum(1 for d in distances if d <= distance_threshold * 3.0)
    else:
        distances = []
        min_dist = max_dist = 0.0
        excellent_count = good_count = poor_count = 0
    
    # Start with coordinate axis indicators
    target_content = f"""	# Visualization targets for {model_name}
	# Total target positions: {len(data_points)}
	# Distance range: {min_dist:.6f} to {max_dist:.6f}
	# Color zones based on threshold {distance_threshold:.6f}:
	#   Green (excellent): distance <= {distance_threshold * 0.5:.6f} ({excellent_count} targets)
	#   Yellow (good): distance <= {distance_threshold:.6f} ({good_count} targets)
	#   Red (poor): distance > {distance_threshold:.6f} ({len(data_points) - good_count} targets)
	# Color coding: Smooth Green -> Yellow -> Red transition
	
	# Coordinate axes for reference
	body {{
		name = axis_x
		mass = 0
		inertia {{ x = 0 y = 0 z = 0 }}
		mesh {{
			shape {{ type = capsule radius = 0.002 height = 0.2 }}
			color {{ r = 1 g = 0 b = 0 a = 1 }}
			pos {{ x = 0.1 y = 0 z = 0 }}
			ori {{ x = 0 y = 0 z = 90 }}
		}}
	}}
	body {{
		name = axis_y
		mass = 0
		inertia {{ x = 0 y = 0 z = 0 }}
		mesh {{
			shape {{ type = capsule radius = 0.002 height = 0.2 }}
			color {{ r = 0 g = 1 b = 0 a = 1 }}
			pos {{ x = 0 y = 0.1 z = 0 }}
		}}
	}}
	body {{
		name = axis_z
		mass = 0
		inertia {{ x = 0 y = 0 z = 0 }}
		mesh {{
			shape {{ type = capsule radius = 0.002 height = 0.2 }}
			color {{ r = 0 g = 0 b = 1 a = 1 }}
			pos {{ x = 0 y = 0 z = 0.1 }}
			ori {{ x = 90 y = 0 z = 0 }}
		}}
	}}
	
"""
    
    # Add sphere for each data point
    for point in data_points:
        x, y, z = point['x'], point['y'], point['z']
        distance = point['distance']
        index = point['index']
        
        # Generate color based on distance threshold
        r, g, b = generate_color_from_distance(distance, distance_threshold)
        
        # Create a body with mesh for this target position
        target_content += f"""	body {{
		name = target_{index:03d}
		mass = 0
		inertia {{ x = 0 y = 0 z = 0 }}
		mesh {{
			shape {{ type = sphere radius = 0.03 }}
			color {{ r = {r:.3f} g = {g:.3f} b = {b:.3f} a = 0.8 }}
			pos {{ x = {x:.6f} y = {y:.6f} z = {z:.6f} }}
		}}
	}}
"""
    
    # Add summary comments
    if data_points:
        best_idx = distances.index(min_dist)
        worst_idx = distances.index(max_dist)
        target_content += f"""
	# Best result: distance {min_dist:.6f} at ({data_points[best_idx]['x']:.3f}, {data_points[best_idx]['y']:.3f}, {data_points[best_idx]['z']:.3f})
	# Worst result: distance {max_dist:.6f} at ({data_points[worst_idx]['x']:.3f}, {data_points[worst_idx]['y']:.3f}, {data_points[worst_idx]['z']:.3f})

"""
    else:
        target_content += """
	# No target positions to display

"""
    
    return target_content


def generate_hfd_content(data_points, model_name, base_hfd_file, distance_threshold=0.1, filter_threshold=None):
    """Generate the complete .hfd file by adding targets to existing base model"""
    
    # Read the base .hfd file
    if not os.path.exists(base_hfd_file):
        raise FileNotFoundError(f"Base .hfd file not found: {base_hfd_file}")
    
    with open(base_hfd_file, 'r') as f:
        base_content = f.read()
    
    # Generate target bodies
    target_bodies = generate_target_bodies(data_points, model_name, distance_threshold, filter_threshold)
    
    # Find the closing brace of the model block
    # We want to insert our targets before the final closing brace
    last_brace = base_content.rfind('}')
    if last_brace == -1:
        raise ValueError("Could not find closing brace in base .hfd file")
    
    # Insert target bodies before the final closing brace
    modified_content = base_content[:last_brace] + target_bodies + base_content[last_brace:]
    
    return modified_content


def main():
    parser = argparse.ArgumentParser(description='Generate SCONE .hfd visualization from data file')
    parser.add_argument('input_file', help='Input .txt data file (format: x y z distance)')
    parser.add_argument('output_file', help='Output .hfd model file')
    parser.add_argument('base_hfd', help='Base .hfd model file to add targets to')
    parser.add_argument('--model-name', help='Model name for comments (inferred from filename if not provided)')
    parser.add_argument('--threshold', type=float, default=0.1, 
                       help='Distance threshold for success/failure coloring (default: 0.1)')
    parser.add_argument('--filter', type=float, default=None,
                       help='Only show points with reaching distance <= this value (default: show all points)')
    
    args = parser.parse_args()
    
    # Check input files exist
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    if not os.path.exists(args.base_hfd):
        print(f"Error: Base .hfd file '{args.base_hfd}' not found")
        return 1
    
    # Infer model name if not provided
    model_name = args.model_name
    if not model_name:
        basename = os.path.basename(args.input_file)
        model_name = os.path.splitext(basename)[0].replace('_results', '')
    
    print(f"Reading data from: {args.input_file}")
    print(f"Model name: {model_name}")
    
    try:
        # Read data points
        data_points = read_data_file(args.input_file)
        
        if not data_points:
            print("Error: No valid data points found in input file")
            return 1
        
        print(f"Found {len(data_points)} data points")
        
        # Calculate statistics
        distances = [p['distance'] for p in data_points]
        min_dist = min(distances)
        max_dist = max(distances)
        avg_dist = sum(distances) / len(distances)
        
        print(f"Distance range: {min_dist:.6f} to {max_dist:.6f}")
        print(f"Average distance: {avg_dist:.6f}")
        
        # Calculate performance statistics by color zones
        excellent_count = sum(1 for d in distances if d <= args.threshold * 0.5)
        good_count = sum(1 for d in distances if d <= args.threshold)
        poor_count = len(distances) - good_count
        
        print(f"Performance breakdown:")
        print(f"  Green (≤{args.threshold * 0.5:.3f}): {excellent_count}/{len(distances)} ({excellent_count/len(distances)*100:.1f}%)")
        print(f"  Yellow (≤{args.threshold:.3f}): {good_count - excellent_count}/{len(distances)} ({(good_count - excellent_count)/len(distances)*100:.1f}%)")
        print(f"  Red (>{args.threshold:.3f}): {poor_count}/{len(distances)} ({poor_count/len(distances)*100:.1f}%)")
        
        # Generate HFD content
        if args.filter is not None:
            print(f"Generating .hfd content with coloring threshold {args.threshold} and filter threshold {args.filter}...")
        else:
            print(f"Generating .hfd content with threshold {args.threshold}...")
        hfd_content = generate_hfd_content(data_points, model_name, args.base_hfd, args.threshold, args.filter)
        
        # Write output file
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output_file, 'w') as f:
            f.write(hfd_content)
        
        print(f"Generated visualization file: {args.output_file}")
        print(f"\\nTo view in SCONE:")
        print(f"1. Open SCONE Studio")
        print(f"2. Load model: {args.output_file}")
        print(f"3. Spheres show target positions with color coding:")
        print(f"   - Green: Good results (low distance)")
        print(f"   - Red: Poor results (high distance)")
        print(f"4. Red/green/blue axes show coordinate system")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())