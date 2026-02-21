#!/usr/bin/env python3
"""
Convert JSON muscle parameter files to TXT format.
Converts files from muscle_params/*.json to muscle_params/*.txt
"""

import json
import os
import glob

def convert_json_to_txt(json_file, txt_file, factor=0.1):
    """Convert a JSON muscle parameter file to TXT format."""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    muscles = data.get('muscles', [])
    
    with open(txt_file, 'w') as f:
        for muscle in muscles:
            f.write(f"{muscle} {{max_isometric_force.factor = {factor}}}\n")

def main():
    # Find all JSON files in muscle_params directory (excluding v0 subdirectory)
    json_files = glob.glob("muscle_params/template_*.json")
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    for json_file in sorted(json_files):
        # Create corresponding TXT filename
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        txt_file = f"muscle_params/{base_name}.txt"
        
        print(f"Converting {json_file} -> {txt_file}")
        convert_json_to_txt(json_file, txt_file)
    
    print("Conversion complete!")
    
    # Verify the conversion
    print("\nVerification:")
    txt_files = glob.glob("muscle_params/template_*.txt")
    print(f"Created {len(txt_files)} TXT files")
    
    # Show a sample
    if txt_files:
        sample_file = sorted(txt_files)[0]
        print(f"\nSample content from {sample_file}:")
        with open(sample_file, 'r') as f:
            lines = f.readlines()[:5]  # Show first 5 lines
            for line in lines:
                print(f"  {line.strip()}")
        if len(lines) == 5:
            print("  ...")

if __name__ == "__main__":
    main()