#!/usr/bin/env python3
"""
Test script to demonstrate both query modes.
"""

import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scripts.query import query_pipeline, load_config

def test_default_mode():
    """Test the default query mode."""
    print("=" * 80)
    print("TESTING DEFAULT MODE")
    print("=" * 80)
    
    config = load_config("config/config.json")
    query = "What are the key principles of occupational therapy for stroke rehabilitation?"
    
    results = query_pipeline(
        query=query,
        config=config,
        n_results=3,
        debug=True,
        mode="default"
    )
    
    print(f"\nResponse saved to: {results['output_path']}")
    return results

def test_functionality_mode():
    """Test the functionality assessment mode."""
    print("\n" + "=" * 80)
    print("TESTING FUNCTIONALITY MODE")
    print("=" * 80)
    
    config = load_config("config/config.json")
    query = "Patient with left hemiplegia following ischemic stroke affecting the right middle cerebral artery, 3 months post-stroke"
    
    results = query_pipeline(
        query=query,
        config=config,
        n_results=5,
        debug=True,
        mode="functionality"
    )
    
    print(f"\nResponse saved to: {results['output_path']}")
    return results

def main():
    """Run both test modes."""
    print("Testing CaregivingLM Query Modes")
    print("This script will test both default and functionality modes.")
    print("Responses will be saved to the 'outputs' directory.\n")
    
    # Test default mode
    default_results = test_default_mode()
    
    # Test functionality mode
    functionality_results = test_functionality_mode()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Default mode response: {default_results['output_path']}")
    print(f"Functionality mode response: {functionality_results['output_path']}")
    print("\nCheck the 'outputs' directory for the saved response files.")

if __name__ == "__main__":
    main() 