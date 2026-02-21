#!/usr/bin/env python3
"""
Generate .scone files from .txt muscle parameter files.
Uses scenes/condition_trajectory_template.scone as template and replaces Properties section
with muscle parameters from muscle_params/*.txt files.
Supports both regular and gravity compensation modes.
"""

import os
import glob
import argparse

def read_template_scone():
    """Read the template .scone file."""
    with open('scenes/condition_trajectory_template.scone', 'r') as f:
        return f.readlines()

def extract_muscle_properties(txt_file):
    """Extract muscle properties from a .txt file."""
    properties = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                properties.append(f"\t\t\t\t{line}")
    return properties

def generate_scone_file(template_lines, muscle_properties, output_file, mode="regular"):
    """Generate a .scone file with replaced muscle properties and mode-specific timing settings."""

    with open(output_file, 'w') as f:
        in_properties_section = False
        properties_written = False

        for line in template_lines:
            # Handle timing variables for gravity compensation mode
            if mode == "grav_comp":
                if '$rest_dur = ' in line:
                    f.write('\t\t$rest_dur = 0\n')
                    continue
                elif '$trg_dur = ' in line:
                    f.write('\t\t$trg_dur = 1.0\n')
                    continue
                elif '$react_dur = ' in line:
                    f.write('\t\t$react_dur = 0.0\n')
                    continue

            # Check if we're entering the Properties section
            if '\t\t\tProperties {' in line:
                f.write(line)
                in_properties_section = True
                # Write the muscle properties only if not written yet
                if not properties_written:
                    for prop in muscle_properties:
                        f.write(prop + '\n')
                    properties_written = True
                continue

            # Check if we're exiting the Properties section
            if in_properties_section and '\t\t\t}' in line:
                f.write(line)
                in_properties_section = False
                continue

            # Skip ALL original muscle property lines within Properties section
            if in_properties_section:
                continue

            # Write all other lines as-is
            f.write(line)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate .scone files from muscle parameter files')
    parser.add_argument('--template-dir', default='muscle_params/v7',
                        help='Directory containing template files (default: muscle_params/v7)')
    parser.add_argument('--mode', choices=['regular', 'grav_comp'], default='regular',
                        help='Generation mode: regular (default) or grav_comp (gravity compensation)')
    parser.add_argument('--output-prefix', default='condition',
                        help='Prefix for output files (default: condition)')

    args = parser.parse_args()

    # Find all .txt files in specified template directory
    txt_files = []

    # Check for both template_*.txt and numbered .txt files
    template_files = glob.glob(f"{args.template_dir}/template_*.txt")
    numbered_files = glob.glob(f"{args.template_dir}/[0-9]*.txt")

    if template_files:
        txt_files = template_files
        print(f"Found {len(txt_files)} template TXT files in {args.template_dir}")
    elif numbered_files:
        txt_files = numbered_files
        print(f"Found {len(txt_files)} numbered TXT files in {args.template_dir}")
    else:
        print(f"No TXT files found in {args.template_dir} directory")
        return

    # Read the template
    template_lines = read_template_scone()
    print("Template .scone file loaded")

    # Create scenes directory if it doesn't exist
    os.makedirs('scenes', exist_ok=True)

    # Generate .scone files
    for i, txt_file in enumerate(sorted(txt_files), start=0):
        print(f"Processing {txt_file}...")

        # Extract muscle properties
        muscle_properties = extract_muscle_properties(txt_file)
        print(f"  Found {len(muscle_properties)} muscle parameters")

        # Generate output filename
        output_file = f"scenes/{args.output_prefix}_{i}.scone"

        # Generate the .scone file
        generate_scone_file(template_lines, muscle_properties, output_file, mode=args.mode)
        print(f"  Created {output_file}")

    print(f"\nGenerated {len(txt_files)} .scone files in scenes/ directory")
    
    # Show a sample of the first generated file
    if txt_files:
        sample_file = f"scenes/{args.output_prefix}_1.scone"
        if os.path.exists(sample_file):
            print(f"\nSample from {sample_file} (timing and Properties section):")
            with open(sample_file, 'r') as f:
                lines = f.readlines()
                in_properties = False
                for line in lines:
                    # Show timing variables
                    if any(var in line for var in ['$rest_dur', '$trg_dur', '$react_dur']):
                        print(f"  {line.strip()}")
                    if 'Properties {' in line:
                        in_properties = True
                        print(f"  {line.strip()}")
                        continue
                    if in_properties:
                        print(f"  {line.strip()}")
                        if '}' in line and 'Properties' not in line:
                            break

if __name__ == "__main__":
    main()