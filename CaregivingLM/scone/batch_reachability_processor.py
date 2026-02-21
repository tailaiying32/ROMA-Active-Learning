#!/usr/bin/env python3
"""
Batch Reachability Processor

Processes joint limit sets with online SCONE evaluation to determine reachability.
For each joint limit set, performs coverage sampling within those limits and
evaluates hand positions using SCONE. Outputs individual JSON files for each
joint limit set with reachable/unreachable positions and joint configurations.

Usage:
    python batch_reachability_processor.py joint_limit_sets.json --n-samples 1000
"""

import numpy as np
import json
import glob
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sconetools import sconepy

def halton_sequence(index, base):
    """Generate the index-th number in the Halton sequence for given base."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result

def generate_halton_samples(n_samples, n_dims):
    """Generate n_samples of n_dims using Halton sequence with different prime bases."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if n_dims > len(primes):
        raise ValueError(f"Too many dimensions ({n_dims}). Max supported: {len(primes)}")

    samples = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        for j in range(n_dims):
            samples[i, j] = halton_sequence(i + 1, primes[j])
    return samples

def read_joint_angles_from_zml(filepath):
    """Extract joint angles from a .zml file."""
    joint_angles = {}

    with open(filepath, 'r') as f:
        in_values_section = False

        for line in f:
            line = line.strip()

            if line == "values {":
                in_values_section = True
                continue
            elif line == "}":
                if in_values_section:
                    break
                continue

            if in_values_section and "=" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    joint_name = parts[0].strip()
                    joint_value = float(parts[1].strip())
                    joint_angles[joint_name] = joint_value

    return joint_angles

def check_joint_limits_compliance(joint_angles, joint_limits):
    """Check if joint angles are within specified limits."""
    violations = {}
    within_limits = True

    for joint_name, angle in joint_angles.items():
        if joint_name in joint_limits:
            min_limit, max_limit = joint_limits[joint_name]

            if angle < min_limit or angle > max_limit:
                violations[joint_name] = {
                    'angle': angle,
                    'limits': [min_limit, max_limit],
                    'violation': angle - max_limit if angle > max_limit else min_limit - angle
                }
                within_limits = False

    return within_limits, violations


def check_min_distance(new_position, existing_positions, min_distance):
    """
    Check if new position maintains minimum distance from all existing positions.

    Args:
        new_position: New hand position [x, y, z]
        existing_positions: List of existing hand positions
        min_distance: Minimum required distance threshold

    Returns:
        bool: True if new position is far enough from all existing positions
    """
    if len(existing_positions) == 0:
        return True

    new_pos = np.array(new_position)
    for existing_pos in existing_positions:
        existing_pos = np.array(existing_pos)
        distance = np.linalg.norm(new_pos - existing_pos)
        if distance < min_distance:
            return False
    return True


def check_collision_forces(model, collision_threshold=0.001):
    """
    Check if current model pose has collision forces above threshold.

    Args:
        model: SCONE model instance
        collision_threshold: Contact force threshold for collision detection

    Returns:
        tuple: (collision_detected: bool, total_force: float, max_force: float)
    """
    total_contact_force = 0.0
    max_body_contact_force = 0.0
    collision_detected = False

    # Check all bodies for contact forces
    for body in model.bodies():
        body_name = body.name()
        # Focus on arm-related bodies
        if any(part in body_name for part in ['hand_r', 'radius_r', 'ulna_r', 'humerus_r', 'scapula_r', 'clavicle_r']):
            try:
                contact_force = body.contact_force()
                force_magnitude = (contact_force.x**2 + contact_force.y**2 + contact_force.z**2)**0.5

                if force_magnitude > collision_threshold:
                    collision_detected = True

                total_contact_force += force_magnitude
                max_body_contact_force = max(max_body_contact_force, force_magnitude)

            except Exception:
                # Skip if contact force unavailable
                pass

    # Also check model-level contact forces
    try:
        model_contact_force = model.contact_force()
        model_force_magnitude = (model_contact_force.x**2 + model_contact_force.y**2 + model_contact_force.z**2)**0.5
        if model_force_magnitude > collision_threshold:
            collision_detected = True
        total_contact_force += model_force_magnitude
    except Exception:
        pass

    return collision_detected, total_contact_force, max_body_contact_force

class SCONEEvaluator:
    """SCONE model evaluator for computing hand positions from joint configurations."""

    def __init__(self, model_file="models/HSA13T_hfd.scone"):
        """Initialize with model file path."""
        self.model_file = model_file
        self.model = None
        self.joint_names = []

    def load_model(self):
        """Load the SCONE model and extract joint information."""
        print(f"Loading SCONE model: {self.model_file}")
        self.model = sconepy.load_model(self.model_file)

        # Extract joint names from DOFs
        self.joint_names = []
        for dof in self.model.dofs():
            self.joint_names.append(dof.name())

        print(f"Found {len(self.joint_names)} DOFs: {self.joint_names}")
        return self.joint_names

    def load_init_values(self, init_file="init/InitHSA13_bs.zml"):
        """Load initial joint values from init file."""
        init_values = {}
        with open(init_file, 'r') as f:
            in_values_section = False
            for line in f:
                line = line.strip()
                if line == "values {":
                    in_values_section = True
                    continue
                elif line == "}":
                    if in_values_section:
                        break
                    continue

                if in_values_section and "=" in line:
                    parts = line.split("=")
                    if len(parts) == 2:
                        joint_name = parts[0].strip()
                        joint_value = float(parts[1].strip())
                        init_values[joint_name] = joint_value

        return init_values

    def evaluate_joint_configuration(self, joint_config, init_values):
        """Evaluate a joint configuration and return hand position."""
        # Set all joints to init values first
        joint_values = init_values.copy()

        # Override with the sampled joint values
        joint_values.update(joint_config)

        # Reset model and set DOF positions
        self.model.reset()

        dof_positions = np.zeros(len(self.joint_names))
        for idx, joint_name in enumerate(self.joint_names):
            dof_positions[idx] = joint_values.get(joint_name, 0.0)

        self.model.set_dof_positions(dof_positions)
        self.model.init_state_from_dofs()

        # Get hand position
        hand_body = None
        for body in self.model.bodies():
            if body.name() == "hand_r":
                hand_body = body
                break

        if hand_body is not None:
            hand_pos = hand_body.com_pos()
            return [hand_pos.x, hand_pos.y, hand_pos.z]
        else:
            print("Warning: hand_r body not found")
            return [0.0, 0.0, 0.0]

def load_joint_limit_sets(json_file):
    """Load joint limit sets from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data['joint_limit_sets'])} joint limit sets from {json_file}")
    print(f"Joints: {', '.join(data['metadata']['joint_names'])}")

    return data['joint_limit_sets'], data['metadata']

def filter_joint_limit_sets_by_batch(joint_limit_sets, batch_selection=None, random_sample=None, random_seed=42):
    """
    Filter joint limit sets by batch selection criteria.

    Args:
        joint_limit_sets: List of joint limit sets
        batch_selection: Batch number to select (1-based), or None for all batches
        random_sample: Number of sets to randomly sample across all batches, or None
        random_seed: Random seed for reproducible sampling

    Returns:
        Filtered list of joint limit sets
    """
    # Check if we have batch metadata
    has_batch_metadata = any('metadata' in jls and 'batch_number' in jls.get('metadata', {})
                           for jls in joint_limit_sets)

    if batch_selection is not None and not has_batch_metadata:
        print("Warning: Batch selection requested but no batch metadata found. Using all sets.")
        batch_selection = None

    # Apply batch filtering first
    if batch_selection is not None:
        # Filter by specific batch number
        filtered_sets = [jls for jls in joint_limit_sets
                        if jls.get('metadata', {}).get('batch_number') == batch_selection]

        if not filtered_sets:
            print(f"Warning: No joint limit sets found for batch {batch_selection}")
            return []

        print(f"Selected batch {batch_selection}: {len(filtered_sets)} sets")

        # Show batch info if available
        if filtered_sets:
            first_set = filtered_sets[0]
            n_limited_joints = first_set.get('metadata', {}).get('n_limited_joints', 'unknown')
            print(f"  Batch {batch_selection} limits {n_limited_joints} joints per set")

    else:
        filtered_sets = joint_limit_sets

    # Apply random sampling if requested
    if random_sample is not None:
        if random_sample >= len(filtered_sets):
            print(f"Random sample size ({random_sample}) >= available sets ({len(filtered_sets)}). Using all sets.")
        else:
            import random
            random.seed(random_seed)
            filtered_sets = random.sample(filtered_sets, random_sample)
            print(f"Randomly sampled {len(filtered_sets)} sets from {len(joint_limit_sets)} total sets (seed: {random_seed})")

    return filtered_sets

def analyze_batch_structure(joint_limit_sets):
    """Analyze and display the batch structure of joint limit sets."""
    # Check if we have batch metadata
    sets_with_metadata = [jls for jls in joint_limit_sets if 'metadata' in jls and 'batch_number' in jls.get('metadata', {})]

    if not sets_with_metadata:
        print("No batch metadata found in joint limit sets.")
        return

    # Group by batch number
    batch_info = {}
    for jls in sets_with_metadata:
        metadata = jls['metadata']
        batch_num = metadata['batch_number']
        n_limited = metadata['n_limited_joints']

        if batch_num not in batch_info:
            batch_info[batch_num] = {
                'count': 0,
                'n_limited_joints': n_limited,
                'limited_joints': metadata.get('limited_joints', []),
                'set_ids': []
            }

        batch_info[batch_num]['count'] += 1
        batch_info[batch_num]['set_ids'].append(jls['id'])

    print(f"\n=== BATCH STRUCTURE ANALYSIS ===")
    print(f"Found {len(batch_info)} batches with {len(sets_with_metadata)} total sets")

    for batch_num in sorted(batch_info.keys()):
        info = batch_info[batch_num]
        set_id_range = f"{min(info['set_ids'])}-{max(info['set_ids'])}" if len(info['set_ids']) > 1 else str(info['set_ids'][0])
        print(f"  Batch {batch_num}: {info['count']:4d} sets (IDs: {set_id_range}) - limits {info['n_limited_joints']:2d}/10 joints")

    return batch_info

def generate_joint_space_samples(joint_limits, n_samples, main_dofs_only=False, 
                                evaluator=None, init_values=None, min_distance=0.0, max_tries=10000,
                                check_collisions=False, collision_threshold=0.001):
    """
    Generate joint space samples within specified limits using Halton sampling with optional distance filtering and collision checking.
    
    Args:
        joint_limits: Dictionary of joint limits
        n_samples: Target number of samples to generate
        main_dofs_only: Whether to filter to main DOFs only
        evaluator: SCONE evaluator for computing hand positions (required if min_distance > 0 or check_collisions)
        init_values: Initial joint values (required if min_distance > 0 or check_collisions)
        min_distance: Minimum distance threshold between hand positions (0.0 = disabled)
        max_tries: Maximum attempts when distance filtering or collision checking is enabled
        check_collisions: Whether to reject samples with collision forces
        collision_threshold: Contact force threshold for collision detection
    
    Returns:
        List of joint configurations
    """
    # Filter to only main DOFs if requested
    if main_dofs_only:
        main_dofs = ['shoulder_flexion_r', 'shoulder_abduction_r', 'shoulder_rotation_r', 'elbow_flexion_r']
        joint_limits = {name: limits for name, limits in joint_limits.items() if name in main_dofs}

    joint_names_ordered = list(joint_limits.keys())
    n_joints = len(joint_names_ordered)

    if n_joints == 0:
        return []

    # If no filtering is enabled, use original simple approach
    if min_distance <= 0.0 and not check_collisions:
        # Generate Halton samples
        halton_samples = generate_halton_samples(n_samples, n_joints)

        joint_configs = []
        for i in range(n_samples):
            joint_config = {}
            for j, joint_name in enumerate(joint_names_ordered):
                min_val, max_val = joint_limits[joint_name]
                # Scale [0,1] Halton sample to [min_val, max_val]
                joint_config[joint_name] = min_val + halton_samples[i, j] * (max_val - min_val)
            joint_configs.append(joint_config)

        return joint_configs

    # Filtering enabled - generate samples with distance and/or collision constraints
    if evaluator is None:
        print("Warning: Evaluator required for distance filtering or collision checking. Using simple sampling.")
        return generate_joint_space_samples(joint_limits, n_samples, main_dofs_only, None, None, 0.0, max_tries, False, collision_threshold)
    
    filter_msg = []
    if min_distance > 0.0:
        filter_msg.append(f"min_distance={min_distance:.3f}m")
    if check_collisions:
        filter_msg.append(f"collision_check={collision_threshold:.3f}")
    print(f"  Using filtered sampling ({', '.join(filter_msg)}, max_tries={max_tries})")
    
    joint_configs = []
    hand_positions = []
    attempt = 0
    accepted_count = 0
    
    # Generate a large pool of Halton samples to avoid running out
    pool_size = max(max_tries, n_samples * 5)
    halton_samples = generate_halton_samples(pool_size, n_joints)
    
    while accepted_count < n_samples and attempt < max_tries:
        # Use Halton sample from pool
        halton_idx = attempt % pool_size
        halton_sample = halton_samples[halton_idx]
        
        # Generate joint configuration
        joint_config = {}
        for j, joint_name in enumerate(joint_names_ordered):
            min_val, max_val = joint_limits[joint_name]
            joint_config[joint_name] = min_val + halton_sample[j] * (max_val - min_val)
        
        # Evaluate hand position using SCONE
        try:
            hand_position = evaluator.evaluate_joint_configuration(joint_config, init_values)
            
            # Check collision constraint first (if enabled)
            collision_rejected = False
            if check_collisions:
                collision_detected, total_force, max_force = check_collision_forces(evaluator.model, collision_threshold)
                if collision_detected:
                    collision_rejected = True
            
            # Check distance constraint (if enabled and not collision rejected)
            distance_rejected = False
            if not collision_rejected and min_distance > 0.0:
                if not check_min_distance(hand_position, hand_positions, min_distance):
                    distance_rejected = True
            
            # Accept sample if it passes all enabled constraints
            if not collision_rejected and not distance_rejected:
                joint_configs.append(joint_config)
                hand_positions.append(hand_position)
                accepted_count += 1
            
        except Exception as e:
            # Skip this sample if SCONE evaluation fails
            pass
        
        attempt += 1
        
        # Progress indicator for filtered sampling
        if attempt % 1000 == 0:
            print(f"    Filtered sampling progress: {accepted_count}/{n_samples} samples, {attempt} attempts")

    if accepted_count < n_samples:
        print(f"  Warning: Only found {accepted_count}/{n_samples} samples meeting filtering constraints after {attempt} attempts")
    else:
        print(f"  Successfully generated {accepted_count} samples with filtering constraints after {attempt} attempts")

    return joint_configs

def process_joint_limit_set(joint_limit_set, evaluator, init_values, n_samples=1000, main_dofs_only=False,
                           min_distance=0.0, max_tries=10000, check_collisions=False, collision_threshold=0.001):
    """Process a single joint limit set with online SCONE evaluation."""
    joint_limits = {name: tuple(limits) for name, limits in joint_limit_set['joint_limits'].items()}

    # Generate joint configurations within the specified limits
    joint_configs = generate_joint_space_samples(
        joint_limits, n_samples, main_dofs_only, 
        evaluator, init_values, min_distance, max_tries, check_collisions, collision_threshold
    )

    if not joint_configs:
        return {
            'reachable_positions': np.array([]),
            'unreachable_positions': np.array([]),
            'reachable_configs': [],
            'unreachable_configs': [],
            'violation_stats': {},
            'violation_examples': {},
            'summary': {
                'total_samples': 0,
                'reachable_count': 0,
                'unreachable_count': 0,
                'reachability_rate': 0.0
            }
        }

    # Evaluate each joint configuration
    reachable_positions = []
    unreachable_positions = []
    reachable_configs = []
    unreachable_configs = []
    violation_stats = {}
    joint_violation_examples = {}

    # Progress bar setup
    actual_samples = len(joint_configs)
    progress_interval = max(1, actual_samples // 20)  # Show progress 20 times

    # If filtering was used, we don't need to re-evaluate - positions were already computed and validated
    if (min_distance > 0.0 or check_collisions) and evaluator is not None:
        print(f"  Evaluating {actual_samples} filtered joint configurations...")
        # For filtered samples, re-evaluate to get positions for JSON output
        for i, joint_config in enumerate(joint_configs):
            try:
                hand_position = evaluator.evaluate_joint_configuration(joint_config, init_values)
                reachable_positions.append(hand_position)
                reachable_configs.append(joint_config)
            except Exception as e:
                unreachable_positions.append([0.0, 0.0, 0.0])  # Default position
                unreachable_configs.append(joint_config)

            # Simple progress bar
            if (i + 1) % progress_interval == 0 or i == actual_samples - 1:
                progress = (i + 1) / actual_samples
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '▒' * (bar_length - filled_length)
                print(f"\r  [{bar}] {i+1}/{actual_samples} ({progress*100:.1f}%)", end='', flush=True)
    else:
        print(f"  Evaluating {actual_samples} joint configurations...")
        for i, joint_config in enumerate(joint_configs):
            # Check if joint configuration is within limits (should always be true for generated samples)
            within_limits, violations = check_joint_limits_compliance(joint_config, joint_limits)

            if within_limits:
                # Evaluate hand position using SCONE
                try:
                    hand_position = evaluator.evaluate_joint_configuration(joint_config, init_values)
                    reachable_positions.append(hand_position)
                    reachable_configs.append(joint_config)
                except Exception as e:
                    unreachable_positions.append([0.0, 0.0, 0.0])  # Default position
                    unreachable_configs.append(joint_config)
            else:
                unreachable_positions.append([0.0, 0.0, 0.0])
                unreachable_configs.append(joint_config)

            # Simple progress bar
            if (i + 1) % progress_interval == 0 or i == actual_samples - 1:
                progress = (i + 1) / actual_samples
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '▒' * (bar_length - filled_length)
                print(f"\r  [{bar}] {i+1}/{actual_samples} ({progress*100:.1f}%)", end='', flush=True)

    print()  # New line after progress bar

    # Convert to numpy arrays
    reachable_positions = np.array(reachable_positions) if reachable_positions else np.array([]).reshape(0, 3)
    unreachable_positions = np.array(unreachable_positions) if unreachable_positions else np.array([]).reshape(0, 3)

    total_samples = len(joint_configs)
    reachable_count = len(reachable_configs)
    unreachable_count = len(unreachable_configs)

    return {
        'reachable_positions': reachable_positions,
        'unreachable_positions': unreachable_positions,
        'reachable_configs': reachable_configs,
        'unreachable_configs': unreachable_configs,
        'violation_stats': violation_stats,
        'violation_examples': joint_violation_examples,
        'summary': {
            'total_samples': total_samples,
            'reachable_count': reachable_count,
            'unreachable_count': unreachable_count,
            'reachability_rate': reachable_count / total_samples if total_samples > 0 else 0.0
        }
    }

def save_reachability_json(joint_limit_set, results, output_path):
    """Save reachability analysis results to JSON file."""

    # Prepare data for JSON serialization
    output_data = {
        'metadata': {
            'joint_limit_set_id': joint_limit_set['id'],
            'analysis_type': 'kinematic_reachability',
            'description': 'Reachability analysis results for specific joint limit set'
        },
        'joint_limits': joint_limit_set['joint_limits'],
        'summary': results['summary'],
        'reachable_positions': results['reachable_positions'].tolist() if len(results['reachable_positions']) > 0 else [],
        'unreachable_positions': results['unreachable_positions'].tolist() if len(results['unreachable_positions']) > 0 else [],
        'reachable_joint_configs': [
            {joint: float(angle) for joint, angle in config.items()}
            for config in results['reachable_configs']
        ],
        'unreachable_joint_configs': [
            {joint: float(angle) for joint, angle in config.items()}
            for config in results['unreachable_configs']
        ],
        'violation_statistics': {
            joint: {
                'count': stats['count'],
                'violation_rate': stats['violation_rate'],
                'average_violation_rad': stats['average_violation'],
                'average_violation_deg': stats['average_violation'] * 180.0 / np.pi
            }
            for joint, stats in results['violation_stats'].items()
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def debug_joint_violations(joint_limit_set, results):
    """Print debugging information about joint violations."""
    set_id = joint_limit_set['id']
    reachability_rate = results['summary']['reachability_rate']

    print(f"\n=== DEBUG: Joint Limit Set {set_id} (Reachability: {reachability_rate:.1%}) ===")

    # Show violation rates by joint
    violation_stats = results['violation_stats']
    print("Joint violation rates:")
    for joint, stats in violation_stats.items():
        if stats['count'] > 0:
            print(f"  {joint:25s}: {stats['count']:4d} violations ({stats['violation_rate']:5.1%}) - "
                  f"avg violation: {stats['average_violation']*180/np.pi:4.1f}°")

    # Show most problematic joints
    worst_joints = sorted(violation_stats.items(), key=lambda x: x[1]['violation_rate'], reverse=True)[:3]
    print(f"\nTop 3 most violating joints:")
    for joint, stats in worst_joints:
        if stats['count'] > 0:
            print(f"  {joint}: {stats['violation_rate']:.1%} violation rate")

            # Show examples if available
            if joint in results['violation_examples'] and results['violation_examples'][joint]:
                examples = results['violation_examples'][joint][:2]  # Show first 2 examples
                for ex in examples:
                    print(f"    Sample {ex['sample_idx']}: {ex['angle_deg']:.1f}° (limits: [{ex['limits_deg'][0]:.1f}°, {ex['limits_deg'][1]:.1f}°])")

def visualize_reachability_distribution(summary_results, output_dir):
    """Create visualization of reachability rate distribution across all joint limit sets."""

    reachability_rates = [r['reachability_rate'] for r in summary_results]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Histogram of reachability rates
    ax1.hist(reachability_rates, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Reachability Rate')
    ax1.set_ylabel('Count of Joint Limit Sets')
    ax1.set_title('Distribution of Reachability Rates')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(reachability_rates), color='red', linestyle='--',
                label=f'Mean: {np.mean(reachability_rates):.2%}')
    ax1.axvline(np.median(reachability_rates), color='orange', linestyle='--',
                label=f'Median: {np.median(reachability_rates):.2%}')
    ax1.legend()

    # Plot 2: Scatter plot of set_id vs reachability_rate
    set_ids = [r['set_id'] for r in summary_results]
    ax2.scatter(set_ids, reachability_rates, alpha=0.6, s=20)
    ax2.set_xlabel('Joint Limit Set ID')
    ax2.set_ylabel('Reachability Rate')
    ax2.set_title('Reachability Rate by Joint Limit Set')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Box plot of reachability rates
    ax3.boxplot(reachability_rates, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7))
    ax3.set_ylabel('Reachability Rate')
    ax3.set_title('Reachability Rate Distribution\n(Box Plot)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative distribution
    sorted_rates = np.sort(reachability_rates)
    cumulative = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
    ax4.plot(sorted_rates, cumulative, 'b-', linewidth=2)
    ax4.set_xlabel('Reachability Rate')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution of Reachability Rates')
    ax4.grid(True, alpha=0.3)

    # Add percentile lines
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    for p in percentiles:
        rate_at_p = np.percentile(reachability_rates, p * 100)
        ax4.axvline(rate_at_p, color='red', alpha=0.5, linestyle=':')
        ax4.text(rate_at_p, p, f'{p:.0%}', rotation=90, va='bottom')

    plt.tight_layout()

    # Save the plot
    plot_path = output_dir / 'reachability_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Reachability distribution plot saved to: {plot_path}")

    # Print summary statistics
    print(f"\n=== REACHABILITY DISTRIBUTION SUMMARY ===")
    print(f"Mean reachability rate: {np.mean(reachability_rates):.2%}")
    print(f"Median reachability rate: {np.median(reachability_rates):.2%}")
    print(f"Std deviation: {np.std(reachability_rates):.2%}")
    print(f"Min reachability rate: {np.min(reachability_rates):.2%}")
    print(f"Max reachability rate: {np.max(reachability_rates):.2%}")
    print(f"25th percentile: {np.percentile(reachability_rates, 25):.2%}")
    print(f"75th percentile: {np.percentile(reachability_rates, 75):.2%}")

    # Identify extreme cases
    low_threshold = np.percentile(reachability_rates, 10)
    high_threshold = np.percentile(reachability_rates, 90)

    low_reachability_sets = [r for r in summary_results if r['reachability_rate'] <= low_threshold]
    high_reachability_sets = [r for r in summary_results if r['reachability_rate'] >= high_threshold]

    print(f"\nLow reachability sets (≤{low_threshold:.1%}): {len(low_reachability_sets)}")
    print(f"High reachability sets (≥{high_threshold:.1%}): {len(high_reachability_sets)}")

    if low_reachability_sets:
        lowest_sets = [f"Set {r['set_id']} ({r['reachability_rate']:.1%})" for r in sorted(low_reachability_sets, key=lambda x: x['reachability_rate'])[:3]]
        print(f"Lowest 3 sets: {lowest_sets}")

    if high_reachability_sets:
        highest_sets = [f"Set {r['set_id']} ({r['reachability_rate']:.1%})" for r in sorted(high_reachability_sets, key=lambda x: x['reachability_rate'], reverse=True)[:3]]
        print(f"Highest 3 sets: {highest_sets}")

    return {
        'mean': np.mean(reachability_rates),
        'median': np.median(reachability_rates),
        'std': np.std(reachability_rates),
        'min': np.min(reachability_rates),
        'max': np.max(reachability_rates),
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'low_reachability_sets': low_reachability_sets,
        'high_reachability_sets': high_reachability_sets
    }

def main():
    """Process joint limit sets and generate reachability JSON files with online SCONE evaluation."""
    parser = argparse.ArgumentParser(description='Process joint limit sets for reachability analysis using online SCONE evaluation')
    parser.add_argument('joint_limit_sets_file', help='JSON file containing joint limit sets')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of joint space samples to generate per joint limit set (default: 1000)')
    parser.add_argument('--model-file', default='models/HSA13T_hfd.scone',
                       help='SCONE model file path (default: models/HSA13T_hfd.scone)')
    parser.add_argument('--output-dir', default='reachability_results',
                       help='Output directory for JSON results (default: reachability_results)')
    parser.add_argument('--limit', type=int, help='Process only first N joint limit sets')
    parser.add_argument('--visualize', action='store_true', help='Create reachability distribution visualization')
    parser.add_argument('--debug', action='store_true', help='Show detailed debugging information for joint violations')
    parser.add_argument('--main-dofs-only', action='store_true', help='Sample only 3 shoulder DOFs + elbow')

    # Batch selection arguments
    parser.add_argument('--batch', type=int, help='Process only sets from specific batch number (1=most constrained, 10=least constrained)')
    parser.add_argument('--random-sample', type=int, help='Randomly sample N sets across all batches')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--show-batches', action='store_true', help='Show batch structure analysis and exit')
    
    # Distance-based sampling arguments
    parser.add_argument('--min-distance', type=float, default=0.0, help='Minimum distance threshold between hand positions for 3D coverage (default: 0.0, disabled)')
    parser.add_argument('--max-tries', type=int, default=10000, help='Maximum attempts to find samples meeting distance threshold (default: 10000)')
    
    # Collision detection arguments
    parser.add_argument('--check-collisions', action='store_true', help='Enable collision detection and reject samples with body collisions')
    parser.add_argument('--collision-threshold', type=float, default=0.001, help='Contact force threshold for collision detection (default: 0.001)')

    args = parser.parse_args()

    print("=== Batch Reachability Processor (Online SCONE Evaluation) ===")

    # Set SCONE log level
    sconepy.set_log_level(3)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load joint limit sets
    joint_limit_sets, metadata = load_joint_limit_sets(args.joint_limit_sets_file)

    # Show batch structure analysis and exit if requested
    if args.show_batches:
        analyze_batch_structure(joint_limit_sets)
        return

    # Apply batch filtering and random sampling
    joint_limit_sets = filter_joint_limit_sets_by_batch(
        joint_limit_sets,
        batch_selection=args.batch,
        random_sample=args.random_sample,
        random_seed=args.random_seed
    )

    if not joint_limit_sets:
        print("No joint limit sets to process after filtering.")
        return

    # Apply limit if specified (after batch filtering)
    if args.limit:
        original_count = len(joint_limit_sets)
        joint_limit_sets = joint_limit_sets[:args.limit]
        print(f"Applied limit: processing first {len(joint_limit_sets)} of {original_count} filtered sets")

    # Initialize SCONE evaluator
    print("\nInitializing SCONE evaluator...")
    evaluator = SCONEEvaluator(args.model_file)
    evaluator.load_model()
    init_values = evaluator.load_init_values()

    # Process each joint limit set
    print(f"\nProcessing {len(joint_limit_sets)} joint limit sets with {args.n_samples} samples each...")

    summary_results = []

    for i, joint_limit_set in enumerate(joint_limit_sets):
        # Batch progress indicator
        batch_progress = (i + 1) / len(joint_limit_sets)
        batch_bar_length = 50
        batch_filled_length = int(batch_bar_length * batch_progress)
        batch_bar = '█' * batch_filled_length + '▒' * (batch_bar_length - batch_filled_length)

        print(f"\nSet {joint_limit_set['id']:04d} [{batch_bar}] {i+1}/{len(joint_limit_sets)} ({batch_progress*100:.1f}%)")

        # Analyze reachability for this joint limit set using online SCONE evaluation
        results = process_joint_limit_set(joint_limit_set, evaluator, init_values,
                                        n_samples=args.n_samples, main_dofs_only=args.main_dofs_only,
                                        min_distance=args.min_distance, max_tries=args.max_tries,
                                        check_collisions=args.check_collisions, collision_threshold=args.collision_threshold)

        # Print summary
        summary = results['summary']
        print(f"  Reachable: {summary['reachable_count']}/{summary['total_samples']} ({summary['reachability_rate']:.1%})")

        # Show debugging information if requested (for online evaluation, this is less relevant)
        # if args.debug:
        #     debug_joint_violations(joint_limit_set, results)

        # Save to JSON
        output_filename = f"reachability_set_{joint_limit_set['id']:04d}.json"
        output_path = output_dir / output_filename
        save_reachability_json(joint_limit_set, results, output_path)

        # Track summary for batch analysis
        summary_results.append({
            'set_id': joint_limit_set['id'],
            'reachability_rate': summary['reachability_rate'],
            'reachable_count': summary['reachable_count'],
            'total_samples': summary['total_samples']
        })


    # Save batch summary
    batch_summary = {
        'metadata': {
            'n_joint_limit_sets': len(joint_limit_sets),
            'total_samples_per_set': args.n_samples,
            'output_directory': str(output_dir),
            'evaluation_method': 'online_scone_evaluation'
        },
        'results': summary_results,
        'statistics': {
            'mean_reachability_rate': np.mean([r['reachability_rate'] for r in summary_results]),
            'std_reachability_rate': np.std([r['reachability_rate'] for r in summary_results]),
            'min_reachability_rate': np.min([r['reachability_rate'] for r in summary_results]),
            'max_reachability_rate': np.max([r['reachability_rate'] for r in summary_results])
        }
    }

    with open(output_dir / 'batch_summary.json', 'w') as f:
        json.dump(batch_summary, f, indent=2)

    print(f"\n=== BATCH PROCESSING COMPLETE ===")
    print(f"Processed {len(joint_limit_sets)} joint limit sets")
    print(f"Results saved to: {output_dir}/")
    print(f"Reachability rate: {batch_summary['statistics']['mean_reachability_rate']:.1%} ± {batch_summary['statistics']['std_reachability_rate']:.1%}")
    print(f"Range: {batch_summary['statistics']['min_reachability_rate']:.1%} - {batch_summary['statistics']['max_reachability_rate']:.1%}")

    # Create reachability distribution visualization if requested
    if args.visualize:
        print(f"\nCreating reachability distribution visualization...")
        viz_stats = visualize_reachability_distribution(summary_results, output_dir)

        # Add visualization stats to batch summary
        batch_summary['visualization_statistics'] = viz_stats

        # Re-save batch summary with visualization stats
        with open(output_dir / 'batch_summary.json', 'w') as f:
            json.dump(batch_summary, f, indent=2)

if __name__ == "__main__":
    main()