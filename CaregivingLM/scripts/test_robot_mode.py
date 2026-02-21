#!/usr/bin/env python3
"""
Test script to demonstrate the robot mode.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scripts.query import query_pipeline, load_config

def test_robot_mode():
    """Test the robot adaptation mode."""
    print("=" * 80)
    print("TESTING ROBOT MODE")
    print("=" * 80)
    
    config = load_config("config/config.json")
    
    # First, create a functionality assessment
    print("Step 1: Creating functionality assessment...")
    functionality_query = "Patient with left hemiplegia following ischemic stroke affecting the right middle cerebral artery, 3 months post-stroke"
    
    functionality_results = query_pipeline(
        query=functionality_query,
        config=config,
        n_results=5,
        debug=True,
        mode="functionality"
    )
    
    print(f"Functionality assessment saved to: {functionality_results['cache_path']}")
    
    # Now use the functionality assessment for robot adaptation
    print("\nStep 2: Creating robot adaptation guidelines...")
    robot_query = "Generate robot adaptation guidelines for dressing task"
    
    robot_results = query_pipeline(
        query=robot_query,
        config=config,
        n_results=5,
        debug=True,
        mode="robot",
        functionality_cache=functionality_results['cache_path'],
        task_name="dressing"
    )
    
    print(f"Robot adaptation guidelines saved to: {robot_results['output_path']}")
    
    return functionality_results, robot_results

def test_multiple_tasks():
    """Test robot mode with multiple tasks."""
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE TASKS")
    print("=" * 80)
    
    config = load_config("config/config.json")
    
    # Use the same functionality assessment for multiple tasks
    functionality_cache = "data/cache/interaction_20231201_143022.json"  # This would be from previous run
    
    tasks = ["feeding", "hygiene", "transferring"]
    
    for task in tasks:
        print(f"\nTesting robot adaptation for {task} task...")
        
        robot_results = query_pipeline(
            query=f"Generate robot adaptation guidelines for {task} task",
            config=config,
            n_results=3,
            debug=False,
            mode="robot",
            functionality_cache=functionality_cache,
            task_name=task
        )
        
        print(f"{task.capitalize()} adaptation saved to: {robot_results['output_path']}")

def main():
    """Run robot mode tests."""
    print("Testing CaregivingLM Robot Mode")
    print("This script will test the robot adaptation mode.")
    print("Responses will be saved to the 'outputs' directory.\n")
    
    # Test robot mode
    functionality_results, robot_results = test_robot_mode()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Functionality assessment: {functionality_results['output_path']}")
    print(f"Robot adaptation guidelines: {robot_results['output_path']}")
    print("\nCheck the 'outputs' directory for the saved response files.")
    
    # Note: Uncomment the following line to test multiple tasks
    # test_multiple_tasks()

if __name__ == "__main__":
    main() 