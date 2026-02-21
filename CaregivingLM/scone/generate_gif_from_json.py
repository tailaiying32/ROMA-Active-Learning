#!/usr/bin/env python3
"""
Generate GIF from JSON

Creates rotating 3D GIF visualizations from reachability JSON files produced by
batch_reachability_processor.py. Supports batch processing and selective visualization.

Usage:
    # Single file
    python generate_gif_from_json.py reachability_results/reachability_set_0001.json

    # Multiple files
    python generate_gif_from_json.py reachability_results/reachability_set_*.json --output-dir gifs/

    # Select interesting cases based on criteria
    python generate_gif_from_json.py reachability_results/*.json --select-criteria low_reachability --output-dir gifs/
"""

import numpy as np
import json
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def load_reachability_json(json_file):
    """Load reachability analysis results from JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error in {json_file}: {e}")

    # Validate required fields
    required_fields = ['reachable_positions', 'unreachable_positions', 'summary', 'metadata']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in {json_file}")

    # Convert lists back to numpy arrays
    try:
        reachable_positions = np.array(data['reachable_positions']) if data['reachable_positions'] else np.array([]).reshape(0, 3)
        unreachable_positions = np.array(data['unreachable_positions']) if data['unreachable_positions'] else np.array([]).reshape(0, 3)
    except Exception as e:
        raise ValueError(f"Error converting position data in {json_file}: {e}")

    return {
        'metadata': data['metadata'],
        'joint_limits': data['joint_limits'],
        'summary': data['summary'],
        'reachable_positions': reachable_positions,
        'unreachable_positions': unreachable_positions,
        'violation_statistics': data.get('violation_statistics', {})
    }

def set_equal_aspect_3d(ax, x_data, y_data, z_data):
    """Set equal aspect ratio for 3D plot."""
    if len(x_data) == 0:
        return

    # Get the range of each axis
    x_range = np.max(x_data) - np.min(x_data)
    y_range = np.max(y_data) - np.min(y_data)
    z_range = np.max(z_data) - np.min(z_data)

    # Get the maximum range
    max_range = max(x_range, y_range, z_range)

    # Get the center of each axis
    x_center = (np.max(x_data) + np.min(x_data)) / 2
    y_center = (np.max(y_data) + np.min(y_data)) / 2
    z_center = (np.max(z_data) + np.min(z_data)) / 2

    # Set the limits
    # ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    # ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    # ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.8, 0.4)
    ax.set_zlim(0.5, 1.8)

def create_static_image_from_reachability_data(reachability_data, output_path, title_suffix=""):
    """Create static 3D PNG image from reachability data."""

    reachable_positions = reachability_data['reachable_positions']
    unreachable_positions = reachability_data['unreachable_positions']
    summary = reachability_data['summary']

    # Apply SCONE coordinate transformation: Y-up to Z-up
    if len(reachable_positions) > 0:
        reach_x = reachable_positions[:, 0]   # X stays X (right)
        reach_y = -reachable_positions[:, 2]  # -Z becomes Y (forward)
        reach_z = reachable_positions[:, 1]   # Y becomes Z (up)
    else:
        reach_x = reach_y = reach_z = np.array([])

    if len(unreachable_positions) > 0:
        unreach_x = unreachable_positions[:, 0]   # X stays X (right)
        unreach_y = -unreachable_positions[:, 2]  # -Z becomes Y (forward)
        unreach_z = unreachable_positions[:, 1]   # Y becomes Z (up)
    else:
        unreach_x = unreach_y = unreach_z = np.array([])

    # Use only reachable points for aspect ratio calculation
    all_x = reach_x if len(reach_x) > 0 else np.array([0])
    all_y = reach_y if len(reach_y) > 0 else np.array([0])
    all_z = reach_z if len(reach_z) > 0 else np.array([0])

    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot reachable positions with inferno colormap
    if len(reach_x) > 0:
        ax.scatter(reach_x, reach_y, reach_z, c=reach_z, cmap='inferno', alpha=0.7, s=20,
                  label=f'Reachable ({len(reach_x)})')

    # Skip unreachable positions for cleaner visualization

    # Add shoulder reference point using the same position as kinematic_reach_visualizer.py
    # Original shoulder position from SCONE model: [0.0, 1.25, 0.15]
    shoulder_x_orig, shoulder_y_orig, shoulder_z_orig = 0.0, 1.25, 0.15

    # Apply the same coordinate transformation: Y-up to Z-up
    shoulder_x_rot = shoulder_x_orig      # x stays x (right)
    shoulder_y_rot = -shoulder_z_orig     # -z becomes y (forward)
    shoulder_z_rot = shoulder_y_orig      # y becomes z (up)

    ax.scatter([shoulder_x_rot], [shoulder_y_rot], [shoulder_z_rot],
               c='blue', s=100, marker='o', label='Shoulder Center')

    # Set equal aspect ratio
    set_equal_aspect_3d(ax, all_x, all_y, all_z)

    # Labels and title
    ax.set_xlabel('X (Right) [m]')
    ax.set_ylabel('Y (Forward) [m]')
    ax.set_zlabel('Z (Up) [m]')

    set_id = reachability_data['metadata']['joint_limit_set_id']
    reachability_rate = summary['reachability_rate']

    title = f'Joint Limit Set {set_id:04d} - Reachable Workspace ({reachability_rate:.1%}){title_suffix}'
    ax.set_title(title)
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set optimal viewing angle for static image
    ax.view_init(elev=20, azim=45)

    # Save as PNG
    print(f"Creating static image: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True


def create_gif_from_reachability_data(reachability_data, output_path, title_suffix=""):
    """Create rotating 3D GIF from reachability data."""

    reachable_positions = reachability_data['reachable_positions']
    unreachable_positions = reachability_data['unreachable_positions']
    summary = reachability_data['summary']

    # Apply SCONE coordinate transformation: Y-up to Z-up
    if len(reachable_positions) > 0:
        reach_x = reachable_positions[:, 0]   # X stays X (right)
        reach_y = -reachable_positions[:, 2]  # -Z becomes Y (forward)
        reach_z = reachable_positions[:, 1]   # Y becomes Z (up)
    else:
        reach_x = reach_y = reach_z = np.array([])

    if len(unreachable_positions) > 0:
        unreach_x = unreachable_positions[:, 0]   # X stays X (right)
        unreach_y = -unreachable_positions[:, 2]  # -Z becomes Y (forward)
        unreach_z = unreachable_positions[:, 1]   # Y becomes Z (up)
    else:
        unreach_x = unreach_y = unreach_z = np.array([])

    # Use only reachable points for aspect ratio calculation
    all_x = reach_x if len(reach_x) > 0 else np.array([0])
    all_y = reach_y if len(reach_y) > 0 else np.array([0])
    all_z = reach_z if len(reach_z) > 0 else np.array([0])

    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot reachable positions with inferno colormap
    if len(reach_x) > 0:
        ax.scatter(reach_x, reach_y, reach_z, c=reach_z, cmap='inferno', alpha=0.7, s=20,
                  label=f'Reachable ({len(reach_x)})')

    # Skip unreachable positions for cleaner visualization

    # Add shoulder reference point using the same position as kinematic_reach_visualizer.py
    # Original shoulder position from SCONE model: [0.0, 1.25, 0.15]
    shoulder_x_orig, shoulder_y_orig, shoulder_z_orig = 0.0, 1.25, 0.15

    # Apply the same coordinate transformation: Y-up to Z-up
    shoulder_x_rot = shoulder_x_orig      # x stays x (right)
    shoulder_y_rot = -shoulder_z_orig     # -z becomes y (forward)
    shoulder_z_rot = shoulder_y_orig      # y becomes z (up)

    ax.scatter([shoulder_x_rot], [shoulder_y_rot], [shoulder_z_rot],
               c='blue', s=100, marker='o', label='Shoulder Center')

    # Set equal aspect ratio
    set_equal_aspect_3d(ax, all_x, all_y, all_z)

    # Labels and title
    ax.set_xlabel('X (Right) [m]')
    ax.set_ylabel('Y (Forward) [m]')
    ax.set_zlabel('Z (Up) [m]')

    set_id = reachability_data['metadata']['joint_limit_set_id']
    reachability_rate = summary['reachability_rate']

    title = f'Joint Limit Set {set_id:04d} - Reachable Workspace ({reachability_rate:.1%}){title_suffix}'
    ax.set_title(title)
    ax.legend()

    # Add grid
    ax.grid(True, alpha=0.3)

    # Animation function
    def animate(frame):
        ax.view_init(elev=20, azim=frame * 2)  # Rotate 2 degrees per frame
        return []

    # Create animation (180 frames for 360 degrees)
    print(f"Creating rotating GIF: {output_path}")
    anim = animation.FuncAnimation(fig, animate, frames=180, interval=50, blit=False)

    # Save as GIF
    anim.save(output_path, writer='pillow', fps=20, dpi=100)
    plt.close()

    return True

def select_interesting_cases(json_files, criteria='low_reachability', n_select=10):
    """Select interesting cases based on specified criteria."""

    print(f"Analyzing {len(json_files)} files to select {n_select} cases using '{criteria}' criteria...")

    file_data = []
    error_count = 0

    # Load summary data from all files
    for json_file in json_files:
        try:
            # Quick check: skip files that are too small (likely incomplete)
            file_size = Path(json_file).stat().st_size
            if file_size < 100:  # Less than 100 bytes is definitely incomplete
                error_count += 1
                if error_count <= 3:
                    print(f"Warning: Skipping {Path(json_file).name}: file too small ({file_size} bytes)")
                continue

            data = load_reachability_json(json_file)
            file_data.append({
                'file': json_file,
                'set_id': data['metadata']['joint_limit_set_id'],
                'reachability_rate': data['summary']['reachability_rate'],
                'reachable_count': data['summary']['reachable_count'],
                'total_samples': data['summary']['total_samples']
            })
        except Exception as e:
            error_count += 1
            if error_count <= 3:  # Only show first 3 errors to avoid spam
                print(f"Warning: Could not load {Path(json_file).name}: {str(e)[:100]}...")
            elif error_count == 4:
                print("... (suppressing additional error messages)")

    if error_count > 0:
        print(f"Total files with errors: {error_count}/{len(json_files)}")

    if not file_data:
        print("No valid files found!")
        return []

    # Apply selection criteria
    if criteria == 'low_reachability':
        # Select files with lowest reachability rates
        selected = sorted(file_data, key=lambda x: x['reachability_rate'])[:n_select]
        print(f"Selected {len(selected)} files with lowest reachability rates:")
        for item in selected:
            print(f"  Set {item['set_id']:04d}: {item['reachability_rate']:.1%} reachable")

    elif criteria == 'high_reachability':
        # Select files with highest reachability rates
        selected = sorted(file_data, key=lambda x: x['reachability_rate'], reverse=True)[:n_select]
        print(f"Selected {len(selected)} files with highest reachability rates:")
        for item in selected:
            print(f"  Set {item['set_id']:04d}: {item['reachability_rate']:.1%} reachable")

    elif criteria == 'diverse_reachability':
        # Select files spanning the range of reachability rates
        sorted_data = sorted(file_data, key=lambda x: x['reachability_rate'])
        indices = np.linspace(0, len(sorted_data) - 1, n_select, dtype=int)
        selected = [sorted_data[i] for i in indices]
        print(f"Selected {len(selected)} files spanning reachability range:")
        for item in selected:
            print(f"  Set {item['set_id']:04d}: {item['reachability_rate']:.1%} reachable")

    else:
        print(f"Unknown criteria: {criteria}. Using first {n_select} files.")
        selected = file_data[:n_select]

    return [item['file'] for item in selected]

def main():
    """Generate GIFs or static images from reachability JSON files."""
    parser = argparse.ArgumentParser(description='Generate GIF visualizations or static images from reachability JSON files')
    parser.add_argument('json_files', nargs='+', help='JSON file(s) to process (supports wildcards)')
    parser.add_argument('--output-dir', default='gifs', help='Output directory for output files (default: gifs)')
    parser.add_argument('--select-criteria', choices=['low_reachability', 'high_reachability', 'diverse_reachability'],
                       help='Selection criteria for interesting cases')
    parser.add_argument('--n-select', type=int, default=10, help='Number of files to select (default: 10)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--static', action='store_true', help='Generate static PNG images instead of rotating GIFs (much faster)')

    args = parser.parse_args()

    mode_name = "Static Image" if args.static else "GIF"
    print(f"=== {mode_name} Generator from JSON ===")

    # Expand wildcards and filter out summary files
    all_json_files = []
    for pattern in args.json_files:
        matching_files = glob.glob(pattern)
        if matching_files:
            # Filter out batch summary files
            filtered_files = [f for f in matching_files if not Path(f).name.startswith('batch_summary')]
            all_json_files.extend(filtered_files)
        else:
            # If no wildcards matched, try as literal filename
            if Path(pattern).exists() and not Path(pattern).name.startswith('batch_summary'):
                all_json_files.append(pattern)

    if not all_json_files:
        print("No JSON files found!")
        return

    print(f"Found {len(all_json_files)} JSON files")

    # Apply selection criteria if specified
    if args.select_criteria:
        selected_files = select_interesting_cases(all_json_files, args.select_criteria, args.n_select)
    else:
        selected_files = all_json_files

    print(f"Processing {len(selected_files)} files...")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Process each file
    success_count = 0
    file_extension = ".png" if args.static else ".gif"
    
    for i, json_file in enumerate(selected_files):
        try:
            # Load reachability data
            reachability_data = load_reachability_json(json_file)
            set_id = reachability_data['metadata']['joint_limit_set_id']

            # Generate output filename
            output_filename = f"reachability_set_{set_id:04d}{file_extension}"
            output_path = output_dir / output_filename

            # Check if file exists and skip if not forcing overwrite
            if output_path.exists() and not args.force:
                print(f"Skipping {output_filename} (already exists, use --force to overwrite)")
                continue

            # Create title suffix with selection criteria if applicable
            title_suffix = ""
            if args.select_criteria:
                reachability_rate = reachability_data['summary']['reachability_rate']
                title_suffix = f" [{args.select_criteria}: {reachability_rate:.1%}]"

            # Generate output file (static image or GIF)
            if args.static:
                create_static_image_from_reachability_data(reachability_data, output_path, title_suffix)
            else:
                create_gif_from_reachability_data(reachability_data, output_path, title_suffix)
            
            success_count += 1

            print(f"Generated: {output_filename} ({i+1}/{len(selected_files)})")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"\n=== GENERATION COMPLETE ===")
    file_type = "PNG files" if args.static else "GIF files"
    print(f"Successfully generated {success_count}/{len(selected_files)} {file_type}")
    print(f"Output directory: {output_dir}/")

if __name__ == "__main__":
    main()